import os
import argparse
import random

from config.global_name import MODEL_PATH_2_5, ROMA_BENCH_DIR, ROMA_BENCH_QA_PATH, ROMA_BENCH_RESULTS_DIR

parser = argparse.ArgumentParser(description="ROMA-Bench evaluation of ROMA-7B (batched active perception on a single GPU)")
parser.add_argument('--gpu', type=int, default=0, help='CUDA device id to bind this process to.')
parser.add_argument('--shard_id', type=int, default=None, help='Which shard of the QA set this process handles (defaults to --gpu).')
parser.add_argument('--num_shards', type=int, default=1, help='Total number of shards the QA set is split into.')
parser.add_argument('--batch_size', type=int, default=4, help='Number of QA items processed in parallel on this single GPU.')
parser.add_argument('--save_every', type=int, default=50, help='Save the results every this many finished QA on this GPU (<=0: only save at the end).')
parser.add_argument('--model_path', default=MODEL_PATH_2_5, help='ROMA-7B checkpoint directory.')
parser.add_argument('--bench_dir', default=ROMA_BENCH_DIR, help='ROMA-Bench scene directory (paths in the QA json are relative to it).')
parser.add_argument('--qa_path', default=ROMA_BENCH_QA_PATH, help='ROMA-Bench QA json (roma_bench_tasks.json).')
parser.add_argument('--output_dir', default=ROMA_BENCH_RESULTS_DIR, help='Directory for the per-shard result / error json files.')
cli_args = parser.parse_args()
if cli_args.shard_id is None:
    cli_args.shard_id = cli_args.gpu

os.environ['CUDA_VISIBLE_DEVICES'] = str(cli_args.gpu)

import json
import re
import csv
import copy
import time
from collections import deque
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import is_flash_attn_2_available

from my_qwen_omni_utils import process_mm_info
from qwen25omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from qwen25omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from utils.util import smart_tokenizer_and_embedding_resize_with_init

# ============================================================
# ROMA-Bench evaluation with single-GPU batching.
#
# Several QA items are packed into one batch (bs>1) and run through a single
# model.thinker.generate() call:
#   - Every QA keeps its own multi-round active-perception state
#     (assistant_parts / action_results / round_i / current_object ...).
#   - Each round, the conversation text + multimodal inputs of every QA that is
#     still running are passed through the processor one by one, then
#     left-padded and concatenated into a batch. The multimodal tensors
#     (pixel_values / pixel_values_tactile / image_grid_thw / input_features /
#     feature_attention_mask) are simply concatenated in batch order: the model
#     slices them according to the placeholder tokens in input_ids, regardless
#     of how many inputs of each modality a sample has.
#   - The action tokens (<lift>/<collide>/...) are added to eos_token_id together
#     with the real eos, so generate()'s built-in per-row early stopping handles
#     each row independently: a row stops as soon as it emits any of these ids
#     (and is padded with pad_token_id afterwards) while the other rows keep
#     generating, until the whole batch stops or max_new_tokens is hit.
#   - After generation, each row is checked: interrupted by an action token
#     (load the observation, continue with the next round) or ended naturally
#     (finalize and produce the result).
#   - Continuous batching: whenever a slot frees up (a QA finished), a new QA is
#     taken from the pending queue right away, keeping the batch as full as
#     possible to maximize GPU utilization.
# ============================================================


# ============================================================
# Settings
# ============================================================

BENCH_DIR = cli_args.bench_dir
TACTILE_BG_PATH = os.path.join(BENCH_DIR, "touch_bg.png")

os.makedirs(cli_args.output_dir, exist_ok=True)
OUTPUT_FILE = os.path.join(cli_args.output_dir, f"eval_roma_bench_{cli_args.shard_id}.json")
ERROR_FILE = os.path.join(cli_args.output_dir, f"error_roma_bench_{cli_args.shard_id}.json")

MAX_ROUNDS = 20
THINKER_MAX_NEW_TOKENS = 8192
SHUFFLE_QA = True

# A list loaded from some previous eval result json (each element has at least
# id / level / qtype). When not empty, only the QA that appear in reference_list
# are evaluated (matched by id+level; if the same id+level has a different qtype
# in reference_list and in the current QA, the two refer to different QA data
# and an error is raised).
# e.g. reference_list = json.load(open("/path/to/some_eval_result.json"))
reference_list = None

ACTION_FOLDER_MAP = {
    "lift": "lift",
    "squeeze": "clench",
    "shake": "shake",
    "rotate": "rotate",
    "collide": "collide",
    "press": "press",
}

SYS_PROMPT = (
    "You are ROMA, an embodied robot assistant based on the Qwen model. "
    "You perceive the environment through vision, audition, touch, and force, and interact with it by "
    "generating actions and natural language. Your goal is to answer the user's questions by combining "
    "perception, reasoning, and interaction. If the available observations are sufficient, answer the "
    "user's question directly. Otherwise, decide which modality or combination of modalities is needed "
    "for the task before actions, and interact with the environment to acquire the missing information. "
    "Reason step by step with a chain of modality. If options are provided, "
    "you should choose the option(s) that is most likely to be the answer."
)
# New special tokens (added in this order) and the words whose embeddings initialize them
SPECIAL_TOKEN_INIT_WORDS = {
    '<|tactile_bos|>': 'tactile start', '<|tactile_eos|>': 'tactile end', '<|TACTILE|>': 'tactile',
    '<grasp_start>': 'grasp start', '<grasp_end>': 'grasp end', '<lift>': 'lift', '<collide>': 'collide',
    '<press>': 'press', '<rotate>': 'rotate', '<squeeze>': 'squeeze', '<shake>': 'shake',
    '<force_start>': 'gravity force start', '<force_end>': 'gravity force end',
}

CHAT_TEMPLATE = "{% set audio_count = namespace(value=0) %}{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% set tactile_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_bos|><|IMAGE|><|vision_eos|>{% elif content['type'] == 'audio' or 'audio' in content or 'audio_url' in content %}{% set audio_count.value = audio_count.value + 1 %}{% if add_audio_id %}Audio {{ audio_count.value }}: {% endif %}<|audio_bos|><|AUDIO|><|audio_eos|>{% elif content['type'] == 'tactile' or 'tactile' in content %}{% set tactile_count.value = tactile_count.value + 1 %}{% if add_audio_id %}Tactile {{ tactile_count.value }}: {% endif %}<|tactile_bos|><|TACTILE|><|tactile_eos|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_bos|><|VIDEO|><|vision_eos|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"


# ============================================================
# Model loading (same as demo.py)
# ============================================================

def load_anytouch(ckpt, tactile_tower):
    """Load the pretrained AnyTouch encoder weights into the thinker's tactile tower."""
    new_ckpt = {}
    for key, item in ckpt.items():
        if "touch_mae_model" in key and 'decoder' not in key and 'mask_token' not in key:
            new_ckpt[key.replace('touch_mae_model.', '')] = copy.deepcopy(item)

    for k, v in tactile_tower.named_parameters():
        if k not in new_ckpt:
            new_ckpt[k] = v.detach().cpu()

    tactile_tower.load_state_dict(new_ckpt, strict=True)
    return tactile_tower


def load_finetune_weights(thinker, path, name):
    """Overlay a finetune_weights.bin (partial state dict) onto the thinker."""
    ckpt = torch.load(path, map_location='cpu')
    for n, p in thinker.named_parameters():
        if n not in ckpt:
            ckpt[n] = p.data
        else:
            print(f"Loading {n} from {name} checkpoint.")
    thinker.load_state_dict(ckpt, strict=False)


roma_dir = cli_args.model_path
# Use flash attention 2 when the flash_attn package is installed, otherwise the default attention
attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
print(f"attn_implementation: {attn_implementation or 'default'}")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    roma_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation=attn_implementation,
)
model.disable_talker()

tactile_ckpt = torch.load(os.path.join(roma_dir, "anytouch2.pth"), map_location='cpu')
model.thinker.tactile_tower = load_anytouch(tactile_ckpt, model.thinker.tactile_tower)

processor = Qwen2_5OmniProcessor.from_pretrained(roma_dir)
smart_tokenizer_and_embedding_resize_with_init(
    {'additional_special_tokens': list(SPECIAL_TOKEN_INIT_WORDS)},
    SPECIAL_TOKEN_INIT_WORDS, processor.tokenizer, model.thinker,
)

tactile_ids = processor.tokenizer.convert_tokens_to_ids(['<|TACTILE|>', '<|tactile_bos|>', '<|tactile_eos|>'])
for cfg in (model.config, model.thinker.config):
    cfg.tactile_token_id, cfg.tactile_start_token_id, cfg.tactile_end_token_id = tactile_ids
processor.chat_template = CHAT_TEMPLATE

load_finetune_weights(model.thinker, os.path.join(roma_dir, "audio.bin"), "audio adapter")
load_finetune_weights(model.thinker, os.path.join(roma_dir, "tactile.bin"), "tactile adapter")
load_finetune_weights(model.thinker, os.path.join(roma_dir, "ROMA-LLM.bin"), "ROMA LLM")

model.eval()
print('model loaded!', torch.cuda.is_available())


# ============================================================
# bbox utilities
# ============================================================

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def parse_bbox(text):
    pattern = re.compile(
        r"\(\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s*,\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s*,\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s*,\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s*\)"
    )
    match = pattern.search(text)
    if match is None:
        return None, "bbox not found"

    raw = list(match.groups())
    values = [float(x) for x in raw]

    all_integer = all(re.fullmatch(r"[+-]?\d+", x.strip()) is not None for x in raw)
    if all_integer:
        return values, None

    for value in values:
        if value < 0.0 or value > 1.0:
            return None, f"floating bbox coordinate out of [0,1]: {values}"

    bbox = [values[0] * 640.0, values[1] * 480.0, values[2] * 640.0, values[3] * 480.0]
    return bbox, None


def extract_objects(save_data):
    objects = []

    def recursive_search(obj):
        if isinstance(obj, dict):
            if "bbox_int" in obj and "folder_name" in obj:
                objects.append(obj)
            for value in obj.values():
                recursive_search(value)
        elif isinstance(obj, list):
            for item in obj:
                recursive_search(item)

    recursive_search(save_data)
    return objects


def resolve_question_content(content_list):
    """Image paths in roma_bench_tasks.json are relative to the ROMA-Bench directory; make them usable."""
    resolved = []
    for item in content_list:
        if item.get("type") == "image":
            item = dict(item, image=os.path.join(BENCH_DIR, item["image"]))
        resolved.append(item)
    return resolved


def get_initial_image(question_content):
    for item in question_content:
        if item.get("type") == "image":
            image_path = item.get("image")
            if not image_path or not os.path.exists(image_path):
                raise Exception(f"Question image not found: {image_path}")
            return image_path
    raise Exception("No image found in question.")


def load_scene_data(image_path):
    """The initial image lies directly in the scene folder (e.g. ROMA-Bench/b1/Initial_180.png)."""
    scene_folder = os.path.dirname(image_path)
    save_path = os.path.join(scene_folder, "save.json")
    if not os.path.exists(save_path):
        raise Exception(f"save.json not found: {save_path}")
    with open(save_path, "r") as f:
        save_data = json.load(f)
    objects = extract_objects(save_data)
    if len(objects) == 0:
        raise Exception(f"No bbox_int/folder_name objects found in {save_path}")
    return scene_folder, objects


def match_object(action_bbox, objects):
    """Object with the largest intersection area, which must cover >= 70% of the object's bbox_int."""
    if action_bbox is None:
        return None, "action bbox is None"

    best_object = None
    best_intersection = 0.0
    for obj in objects:
        bbox_int = obj.get("bbox_int")
        if not isinstance(bbox_int, (list, tuple)) or len(bbox_int) != 4:
            continue
        obj_bbox = [float(x) for x in bbox_int]
        if bbox_area(obj_bbox) <= 0:
            continue
        inter = intersection_area(action_bbox, obj_bbox)
        if inter > best_intersection:
            best_intersection = inter
            best_object = obj

    if best_object is None:
        return None, "No object has positive intersection."

    best_bbox = [float(x) for x in best_object["bbox_int"]]
    best_area = bbox_area(best_bbox)
    if best_area <= 0:
        return None, "Matched object bbox has zero area."

    overlap_ratio = best_intersection / best_area
    if overlap_ratio < 0.70:
        return None, (
            f"Best intersection ratio {overlap_ratio:.4f} < 0.70; "
            f"action_bbox={action_bbox}; object_bbox={best_bbox}"
        )

    return best_object, None


def parse_actions(text):
    """
    Parse <grasp_start>...<grasp_end> and the following action tokens in text order;
    each action inherits the bbox of the latest grasp.
    """
    events = []

    grasp_pattern = re.compile(r"<grasp_start>\s*(.*?)\s*<grasp_end>", re.DOTALL)
    for match in grasp_pattern.finditer(text):
        grasp_text = match.group(1).strip()
        bbox, error = parse_bbox(grasp_text)
        events.append({"position": match.start(), "action": "grasp", "bbox": bbox, "error": error, "text": grasp_text, "modality": None})

    # Action tokens usually follow the "(use modality) <action>" pattern from the training data; parse the modality too
    action_pattern = re.compile(r"(?:\(\s*use\s+([^)]*?)\s*\)\s*)?<(lift|squeeze|shake|rotate|collide|press)>", re.IGNORECASE)
    for match in action_pattern.finditer(text):
        modality = match.group(1).strip() if match.group(1) else None
        events.append({"position": match.start(2), "action": match.group(2), "bbox": None, "error": None, "text": "", "modality": modality})

    events.sort(key=lambda x: x["position"])

    current_bbox = None
    parsed_actions = []
    for event in events:
        if event["action"] == "grasp":
            current_bbox = event["bbox"]
            parsed_actions.append({"action": "grasp", "bbox": event["bbox"], "error": event["error"], "text": event["text"], "modality": None})
        else:
            if current_bbox is None:
                parsed_actions.append({"action": event["action"], "bbox": None, "error": "No previous grasp bbox for this action.", "text": "", "modality": event["modality"]})
            else:
                parsed_actions.append({"action": event["action"], "bbox": current_bbox, "error": None, "text": "", "modality": event["modality"]})

    return parsed_actions


def get_gelsight_folder(object_folder):
    gelsight_folder = "GelSight"
    test_path = os.path.join(object_folder, "clench", gelsight_folder, "0.png")
    if not os.path.exists(test_path):
        gelsight_folder = "GelSightL"
    return gelsight_folder


def read_force_observation(action_folder):
    force_csv = os.path.join(action_folder, "force", "data.csv")
    if not os.path.exists(force_csv):
        return None

    with open(force_csv, "r") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return None

    values = []
    for row in rows[1:]:
        if len(row) <= 3:
            continue
        try:
            values.append(float(row[3]))
        except (ValueError, TypeError):
            continue

    if len(values) == 0:
        return None

    object_force_mean = round(float(np.mean(values)), 2)
    return f"<force_start> mean gravity force {object_force_mean} N <force_end>"


def parse_modality_keywords(modality_text):
    """
    Parse text like "(use vision and touch)" / "(use force)" into a set of modality keywords.
    None means the model did not specify a modality (fall back to every modality that exists for the action).
    """
    if not modality_text:
        return None

    text = modality_text.lower()
    keywords = set()
    if 'vision' in text or 'visual' in text or 'image' in text:
        keywords.add('image')
    if 'touch' in text or 'tactile' in text:
        keywords.add('tactile')
    if 'audio' in text or 'sound' in text or 'audit' in text:
        keywords.add('audio')
    if 'force' in text or 'gravity' in text or 'weight' in text:
        keywords.add('force')

    return keywords if keywords else None


def load_action_observation(object_folder, action_name, modality=None):
    """
    Load the observations that exist under the action folder, in our model's message content format.

    Only the modalities the model explicitly requested in "(use modality) <action>" are loaded; if no
    modality can be parsed (e.g. the model did not follow the format), every modality that exists is loaded.
    """
    if action_name not in ACTION_FOLDER_MAP:
        raise Exception(f"Unsupported action: {action_name}")

    action_folder = os.path.join(object_folder, ACTION_FOLDER_MAP[action_name])
    if not os.path.isdir(action_folder):
        raise Exception(f"Action folder does not exist: {action_folder}")

    requested = parse_modality_keywords(modality)

    observations = []

    # image
    if requested is None or 'image' in requested:
        image_path = os.path.join(action_folder, "WristCamera", "25.png")
        if os.path.exists(image_path):
            observations.append({"type": "image", "image": image_path})

    # tactile
    if requested is None or 'tactile' in requested:
        gelsight_folder = get_gelsight_folder(object_folder)
        tactile_path0 = os.path.join(action_folder, gelsight_folder, "0.png")
        tactile_path1 = os.path.join(action_folder, gelsight_folder, "25.png")
        if os.path.exists(tactile_path0) and os.path.exists(tactile_path1):
            observations.append({"type": "tactile", "tactile": [tactile_path0, tactile_path1], "tactile_bg": TACTILE_BG_PATH})

    # audio
    if requested is None or 'audio' in requested:
        audio_path = os.path.join(action_folder, "mic", "audio_enhanced_cut.wav")
        if os.path.exists(audio_path):
            observations.append({"type": "audio", "audio": audio_path})

    # force
    if requested is None or 'force' in requested:
        force_text = read_force_observation(action_folder)
        if force_text is not None:
            observations.append({"type": "text", "text": force_text})

    if len(observations) == 0:
        raise Exception(f"No observation found in {action_folder} for requested modality={modality}")

    print(f"  [OBS] action={action_name} modality={modality} folder={action_folder} -> {observations}")

    return observations


# ============================================================
# Action token / eos token ids. The action tokens are passed to generate() as
# "pseudo eos" ids, so its built-in per-row early stopping interrupts every
# sample in the batch independently.
# ============================================================

ACTION_NAMES = ['collide', 'shake', 'press', 'rotate', 'squeeze', 'lift']
ACTION_ID_TO_NAME = {}
ACTION_ID_SET = set()
for _name in ACTION_NAMES:
    _tid = processor.tokenizer.convert_tokens_to_ids(f'<{_name}>')
    ACTION_ID_TO_NAME[_tid] = _name
    ACTION_ID_SET.add(_tid)

EOS_TOKEN_IDS = set()
_gen_cfg_eos = getattr(getattr(model.thinker, "generation_config", None), "eos_token_id", None)
if _gen_cfg_eos is not None:
    if isinstance(_gen_cfg_eos, (list, tuple, set)):
        EOS_TOKEN_IDS.update(_gen_cfg_eos)
    else:
        EOS_TOKEN_IDS.add(_gen_cfg_eos)
for _tok in ['<|im_end|>', '<|endoftext|>']:
    _tid = processor.tokenizer.convert_tokens_to_ids(_tok)
    if _tid is not None and _tid >= 0:
        EOS_TOKEN_IDS.add(_tid)

STOP_ID_SET = ACTION_ID_SET | EOS_TOKEN_IDS
STOP_ID_LIST = list(STOP_ID_SET)

# Same id used to pad input_ids during training (<|im_end|>). It is also in
# EOS_TOKEN_IDS, so using it as generate()'s fill value after a row stops
# introduces no ambiguity.
PAD_ID = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')


def render_placeholder_text(assistant_parts):
    """Render assistant_parts as human-readable text, with placeholders for the multimodal data."""
    full_text = ""
    for p in assistant_parts:
        if p["type"] == "text":
            full_text += p["text"]
        elif p["type"] == "image":
            full_text += "[image]"
        elif p["type"] == "tactile":
            full_text += "[tactile]"
        elif p["type"] == "audio":
            full_text += "[audio]"
    return full_text


def render_placeholder_text_raw(assistant_parts):
    """Concatenate only the text parts of assistant_parts (for the parse_actions regexes, no placeholders)."""
    return "".join(p["text"] for p in assistant_parts if p["type"] == "text")


def find_stop_index(new_ids, stop_id_set):
    """Find the first stop id (action token or real eos) among the newly generated tokens of a row."""
    for idx, tid in enumerate(new_ids.tolist()):
        if tid in stop_id_set:
            return idx, tid
    return None, None


# ============================================================
# Active-perception state of a single QA
# ============================================================

class ActiveItem:
    def __init__(self, qa):
        self.qa = qa
        self.level = qa["level"]
        self.scene_folder = None
        self.objects = None
        self.initial_messages = None
        self.assistant_parts = []
        self.action_results = []
        self.recorded_count = 0
        self.current_object = None
        self.round_i = 0
        self.finished = False
        self.result = None
        self.error_id = None


def init_item(item):
    question_content = resolve_question_content(item.qa["question"]["content"])
    image_path = get_initial_image(question_content)
    item.scene_folder, item.objects = load_scene_data(image_path)
    item.initial_messages = [
        {"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]},
        {"role": "user", "content": question_content},
    ]


def build_conv(item):
    if item.assistant_parts:
        return item.initial_messages + [{"role": "assistant", "content": list(item.assistant_parts)}]
    return item.initial_messages


def get_text_qa(content_list):
    all_text = ""
    for pp in content_list:
        t = pp.get('type')
        if t == 'text':
            all_text += pp['text']
        elif t == 'image':
            all_text += '[IMAGE]'
        elif t == 'tactile':
            all_text += '[TACTILE]'
        elif t == 'audio':
            all_text += '[AUDIO]'
    return all_text


def extract_final_answer(full_text):
    matches = re.findall(r"answer is\s*\[?\s*([A-Za-z]+)\s*\]?", full_text, re.IGNORECASE)
    if len(matches) == 0:
        return None
    return matches[-1]


def build_result(item):
    qa = item.qa
    now_id = qa["id"]
    full_text = render_placeholder_text(item.assistant_parts)
    answer_pred = extract_final_answer(full_text)
    answer_gt = qa.get("answer_text")
    print(f"[DONE] {item.level} qa {now_id}: pred={answer_pred} gt={answer_gt}")
    return {
        "id": now_id,
        "level": item.level,
        "qtype": qa.get("type"),
        "question_text": get_text_qa(qa["question"]["content"]),
        "question": qa.get("question"),
        "answer_llm": full_text,
        "answer_final": answer_pred,
        "answer_gt": answer_gt,
        # Rough exact-match only; check_eval_results.py gives the actual accuracy
        "correct": (answer_pred is not None and answer_pred.upper() == str(answer_gt).upper()),
        "action_results": item.action_results,
    }


def finalize_success(item):
    item.result = build_result(item)
    item.error_id = None
    item.finished = True


def finalize_with_error(item, exc):
    qa = item.qa
    now_id = qa["id"]
    print(f"[ERROR] {item.level} qa {now_id}: {repr(exc)}")
    item.result = {
        "id": now_id,
        "level": item.level,
        "qtype": qa.get("type"),
        "question_text": get_text_qa(qa["question"]["content"]),
        "question": qa.get("question"),
        "answer_llm": None,
        "answer_final": None,
        "answer_gt": qa.get("answer_text"),
        "correct": False,
        "action_results": item.action_results,
        "error": repr(exc),
    }
    item.error_id = f"{item.level}_{now_id}"
    item.finished = True


# ============================================================
# Batch collation of the multimodal inputs: pixel_values / pixel_values_tactile /
# image_grid_thw / input_features / feature_attention_mask are concatenated in
# batch order; input_ids / attention_mask are additionally left-padded, since
# generate() needs the new tokens of every row to be aligned in the same column.
# ============================================================

def collate_batch(per_item_inputs):
    input_ids_list = [inp['input_ids'][0] for inp in per_item_inputs]
    max_len = max(x.shape[0] for x in input_ids_list)
    bsz = len(input_ids_list)

    input_ids = torch.full((bsz, max_len), PAD_ID, dtype=input_ids_list[0].dtype)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        L = ids.shape[0]
        input_ids[i, max_len - L:] = ids
        attention_mask[i, max_len - L:] = 1

    batch = {'input_ids': input_ids, 'attention_mask': attention_mask}

    for key in ['pixel_values', 'image_grid_thw', 'pixel_values_tactile', 'input_features', 'feature_attention_mask']:
        tensors = [inp[key] for inp in per_item_inputs if key in inp]
        if len(tensors) > 0:
            batch[key] = torch.cat(tensors, dim=0)

    return batch


def prepare_round_inputs(active):
    """Build this round's conversation text + multimodal inputs for every QA in the batch that is still running."""
    valid_items = []
    per_item_inputs = []
    for item in active:
        try:
            conv = build_conv(item)
            if item.assistant_parts:
                text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                for suffix in ['<|im_end|>\n', '<|im_end|>']:
                    if text.endswith(suffix):
                        text = text[:-len(suffix)]
                        break
            else:
                text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            audios, images, videos, tactiles = process_mm_info(conv, use_audio_in_video=True)
            inp = processor(
                text=text, audio=audios, images=images, videos=videos,
                tactile=tactiles, return_tensors="pt", padding=True, use_audio_in_video=True,
            )
            valid_items.append(item)
            per_item_inputs.append(inp)
        except Exception as e:
            finalize_with_error(item, e)
    return valid_items, per_item_inputs


def process_generation_result(item, new_ids, stop_idx, stop_id):
    if stop_idx is not None:
        trimmed_ids = new_ids[:stop_idx + 1]
    else:
        trimmed_ids = new_ids
    new_text = processor.tokenizer.decode(trimmed_ids, skip_special_tokens=False)
    new_text = new_text.replace('<|im_end|>', '').replace('<|endoftext|>', '')

    if stop_id is None or stop_id not in ACTION_ID_SET:
        # Natural end (real eos, or in the extreme case max_new_tokens ran out without any stop id)
        if new_text.strip():
            item.assistant_parts.append({"type": "text", "text": new_text})
        finalize_success(item)
        return

    action_name = ACTION_ID_TO_NAME[stop_id]

    all_text = render_placeholder_text_raw(item.assistant_parts) + new_text
    parsed_so_far = parse_actions(all_text)
    new_events = parsed_so_far[item.recorded_count:]
    item.recorded_count = len(parsed_so_far)

    for event in new_events:
        if event["action"] == "grasp":
            if event["error"] is not None:
                raise Exception(f"grasp bbox error: {event['error']}")
            matched_object, match_error = match_object(event["bbox"], item.objects)
            if matched_object is None:
                raise Exception(f"grasp object matching failed: {match_error}")
            item.current_object = matched_object
            item.action_results.append({
                "action": "grasp",
                "bbox": event["bbox"],
                "folder_name": matched_object["folder_name"],
                "bbox_int": matched_object["bbox_int"],
            })
            print(f"  [PARSE] grasp -> bbox={event['bbox']} matched folder={matched_object['folder_name']} (bbox_int={matched_object['bbox_int']})")
        else:
            if item.current_object is None:
                raise Exception(f"action <{event['action']}> appears before any grasp.")
            item.action_results.append({
                "action": event["action"],
                "bbox": event["bbox"],
                "folder_name": item.current_object["folder_name"],
                "bbox_int": item.current_object["bbox_int"],
                "modality": event["modality"],
            })
            print(f"  [PARSE] action=<{event['action']}> modality={event['modality']} -> folder={item.current_object['folder_name']} (bbox_int={item.current_object['bbox_int']})")

    object_folder = os.path.join(item.scene_folder, item.current_object["folder_name"])
    if not os.path.isdir(object_folder):
        raise Exception(f"Object folder does not exist: {object_folder}")

    triggered_modality = new_events[-1]["modality"] if new_events else None
    observations = load_action_observation(object_folder, action_name, triggered_modality)

    item.assistant_parts.append({"type": "text", "text": new_text + " → observations: "})
    item.assistant_parts.extend(observations)

    item.round_i += 1
    if item.round_i >= MAX_ROUNDS:
        # Rounds exhausted: finalize, even if the last round was still interrupted by an action token
        finalize_success(item)


@torch.no_grad()
def run_one_round(active):
    """Run one generate() round over every QA in the batch that is still running."""
    valid_items, per_item_inputs = prepare_round_inputs(active)
    if len(valid_items) == 0:
        return

    print(f"\n[BATCH] running one round for {len(valid_items)} QA: ids={[(it.level, it.qa['id']) for it in valid_items]}")

    batch = collate_batch(per_item_inputs)
    batch = BatchFeature(data=batch)
    batch = batch.to(model.device).to(model.dtype)

    input_len = batch['input_ids'].shape[1]

    # Call the thinker directly (as demo.py does): model.generate() requires a talker speaker
    # from spk_dict.pt, which is never loaded since the talker is disabled.
    output = model.thinker.generate(
        **batch,
        use_audio_in_video=True,
        max_new_tokens=THINKER_MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=STOP_ID_LIST,
        pad_token_id=PAD_ID,
    )

    for row_idx, item in enumerate(valid_items):
        new_ids = output[row_idx, input_len:]
        stop_idx, stop_id = find_stop_index(new_ids, STOP_ID_SET)
        try:
            process_generation_result(item, new_ids, stop_idx, stop_id)
        except Exception as e:
            finalize_with_error(item, e)


# ============================================================
# reference_list filtering
# ============================================================

def filter_qa_by_reference(all_qa, reference_list):
    """
    When reference_list is not empty, keep only the QA whose (id, level) appears in it;
    raise if the same (id, level) has a different qtype in reference_list and in the current QA.
    """
    if not reference_list:
        return all_qa

    reference_qtype = {}
    for ref in reference_list:
        key = (ref.get("id"), ref.get("level"))
        reference_qtype[key] = ref.get("qtype")

    filtered = []
    for qa in all_qa:
        key = (qa.get("id"), qa.get("level"))
        if key not in reference_qtype:
            continue
        ref_qtype = reference_qtype[key]
        cur_qtype = qa.get("type")
        if ref_qtype != cur_qtype:
            raise Exception(
                f"reference_list mismatch for id={qa.get('id')} level={qa.get('level')}: "
                f"qtype in reference_list={ref_qtype!r} vs current qa={cur_qtype!r}"
            )
        filtered.append(qa)
    return filtered


# ============================================================
# Main loop: single-GPU continuous batching
#   - Keep an `active` list (QA occupying a batch slot) and a `queue` (QA not started yet).
#   - First fill `active` up to batch_size from the queue (initialize each new QA;
#     QA that fail to initialize are marked as errors right away).
#   - Run one batched generate() round over `active`; each QA decides whether to
#     continue (interrupted by an action token, load the observation) or finish
#     (natural end / rounds exhausted / error).
#   - Finished QA leave `active` and are recorded; their slots are refilled from the
#     queue in the next iteration, keeping the GPU running a full batch.
#   - Results are saved every --save_every finished QA, and once more at the end.
# ============================================================

with open(cli_args.qa_path, "r", encoding="utf-8") as f:
    all_qa = json.load(f)

all_qa = filter_qa_by_reference(all_qa, reference_list)
if reference_list:
    print(f"[REFERENCE] {len(all_qa)} QA remain after reference_list filtering.")

if SHUFFLE_QA:
    random.seed(42)
    random.shuffle(all_qa)
    print(f"[SHUFFLE] {len(all_qa)} QA shuffled.")

all_qa = all_qa[cli_args.shard_id::cli_args.num_shards]
print(f"[SHARD] gpu={cli_args.gpu} shard_id={cli_args.shard_id} num_shards={cli_args.num_shards} batch_size={cli_args.batch_size} -> {len(all_qa)} QA")

queue = deque(ActiveItem(qa) for qa in all_qa)
active = []

all_answer = []
error_list = []
last_saved = 0


def record(item):
    all_answer.append(item.result)
    if item.error_id is not None:
        error_list.append(item.error_id)


def save_results():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_answer, f, indent=4, ensure_ascii=False)
    with open(ERROR_FILE, "w", encoding="utf-8") as f:
        json.dump(error_list, f, indent=4, ensure_ascii=False)
    print(f"[SAVE] {len(all_answer)}/{len(all_qa)} QA saved to {OUTPUT_FILE}")


start_time = time.time()

while active or queue:
    while len(active) < cli_args.batch_size and queue:
        item = queue.popleft()
        try:
            print(f"[START] processing {item.level} qa {item.qa['id']}")
            init_item(item)
            active.append(item)
        except Exception as e:
            finalize_with_error(item, e)
            record(item)

    if active:
        run_one_round(active)

        still_active = []
        for item in active:
            if item.finished:
                record(item)
            else:
                still_active.append(item)
        active = still_active

    if cli_args.save_every > 0 and len(all_answer) - last_saved >= cli_args.save_every:
        save_results()
        last_saved = len(all_answer)

save_results()

elapsed = time.time() - start_time
success_count = len(all_answer) - len(error_list)
correct_count = sum(1 for r in all_answer if r.get("correct"))

print()
print("=" * 60)
print("Finished!")
print(f"Total QA: {len(all_answer)}")
print(f"Success: {success_count}")
print(f"Error: {len(error_list)}")
print(f"Correct: {correct_count} ({correct_count / max(1, len(all_answer)) * 100:.2f}%)")
print("NOTE: Success / Error / Correct above are NOT the final results. The model output is")
print("      sometimes not just the option letter(s), and 'Correct' here is only a rough exact")
print("      match. Run check_eval_results.py and use its results.")
print(f"Time: {elapsed:.2f} s")
print(f"Error IDs: {error_list}")
print(f"Output: {OUTPUT_FILE}")
print(f"Error File: {ERROR_FILE}")
print("=" * 60)
