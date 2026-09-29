import os
import argparse

parser = argparse.ArgumentParser(description="ROMA active-perception web demo")
parser.add_argument('--model_path', default='/share/project/robocoin/frx/ROMA-7B', help='Merged ROMA-7B checkpoint directory.')
parser.add_argument('--gpu', type=int, default=0, help='CUDA device id to bind this process to (default: leave CUDA_VISIBLE_DEVICES untouched).')
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--port', type=int, default=7860)
parser.add_argument('--share', action='store_true', help='Create a public gradio link.')
cli_args = parser.parse_args()
if cli_args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cli_args.gpu)

import csv
import html
import inspect
import json
import re
import threading
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from my_qwen_omni_utils import process_mm_info
from qwen25omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from qwen25omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor

# ============================================================
# Single-QA web version of the active-perception loop in eval.py:
#   generate -> stop at an action token (<lift>/<press>/...) -> load the real observation
#   for that action from example_data (wrist image / tactile / audio / force) and append it
#   to the context -> keep generating until the model ends naturally (gives the answer)
#   or MAX_ROUNDS is reached.
# Action parsing / object matching / observation loading follow eval.py (release_bbox_restrict=True).
# ============================================================

ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT / "resources"
SCENE_DIR = RES_DIR / "example_data" / "1"
SCENE_IMAGE = SCENE_DIR / "Initial_180.png"
TACTILE_BG_PATH = RES_DIR / "example_data" / "tactile_bg.png"
IMG_W, IMG_H = 640.0, 480.0  # image size that bbox_int in save.json refers to

MAX_ROUNDS = 20
THINKER_MAX_NEW_TOKENS = 8192

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
    "The available actions are: <lift>, <squeeze>, <shake>, <rotate>, <collide>, and <press>. "
    "You should specify the modalities you want to use for that action by using (use modality) before the action token."
    "You must grasp an object before performing any action on it by generating <grasp_start> object name at (x1,y1,x2,y2) <grasp_end> with the bounding box of the object. "
    "Reason step by step with a chain of modality. If options are provided, "
    "you should choose the option(s) that is most likely to be the answer."
)

SPECIAL_TOKENS = [
    '<|tactile_bos|>', '<|tactile_eos|>', '<|TACTILE|>',
    '<grasp_start>', '<grasp_end>', '<lift>', '<collide>', '<press>',
    '<rotate>', '<squeeze>', '<shake>', '<force_start>', '<force_end>',
]

CHAT_TEMPLATE = "{% set audio_count = namespace(value=0) %}{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% set tactile_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_bos|><|IMAGE|><|vision_eos|>{% elif content['type'] == 'audio' or 'audio' in content or 'audio_url' in content %}{% set audio_count.value = audio_count.value + 1 %}{% if add_audio_id %}Audio {{ audio_count.value }}: {% endif %}<|audio_bos|><|AUDIO|><|audio_eos|>{% elif content['type'] == 'tactile' or 'tactile' in content %}{% set tactile_count.value = tactile_count.value + 1 %}{% if add_audio_id %}Tactile {{ tactile_count.value }}: {% endif %}<|tactile_bos|><|TACTILE|><|tactile_eos|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_bos|><|VIDEO|><|vision_eos|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"


# ============================================================
# bbox / action parsing / observation loading (same as eval.py)
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
    num = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    match = re.search(r"\(\s*" + r"\s*,\s*".join([num] * 4) + r"\s*\)", text)
    if match is None:
        return None, "bbox not found"

    raw = list(match.groups())
    values = [float(x) for x in raw]
    if all(re.fullmatch(r"[+-]?\d+", x.strip()) is not None for x in raw):
        return values, None

    for value in values:
        if value < 0.0 or value > 1.0:
            return None, f"floating bbox coordinate out of [0,1]: {values}"
    return [values[0] * IMG_W, values[1] * IMG_H, values[2] * IMG_W, values[3] * IMG_H], None


def match_object(action_bbox, objects):
    """Object with the largest intersection area (no 70% area restriction, i.e. release_bbox_restrict=True in eval.py)."""
    if action_bbox is None:
        return None, "action bbox is None"

    best_object, best_intersection = None, 0.0
    for obj in objects:
        bbox_int = obj.get("bbox_int")
        if not isinstance(bbox_int, (list, tuple)) or len(bbox_int) != 4:
            continue
        obj_bbox = [float(x) for x in bbox_int]
        if bbox_area(obj_bbox) <= 0:
            continue
        inter = intersection_area(action_bbox, obj_bbox)
        if inter > best_intersection:
            best_intersection, best_object = inter, obj

    if best_object is None:
        return None, "No object has positive intersection."
    return best_object, None


def parse_actions(text):
    """Parse <grasp_start>...<grasp_end> and action tokens in text order; each action inherits the latest grasp bbox."""
    events = []

    for match in re.finditer(r"<grasp_start>\s*(.*?)\s*<grasp_end>", text, re.DOTALL):
        bbox, error = parse_bbox(match.group(1).strip())
        events.append({"position": match.start(), "action": "grasp", "bbox": bbox, "error": error, "modality": None})

    action_pattern = re.compile(r"(?:\(\s*use\s+([^)]*?)\s*\)\s*)?<(lift|squeeze|shake|rotate|collide|press)>", re.IGNORECASE)
    for match in action_pattern.finditer(text):
        modality = match.group(1).strip() if match.group(1) else None
        events.append({"position": match.start(2), "action": match.group(2), "bbox": None, "error": None, "modality": modality})

    events.sort(key=lambda x: x["position"])

    current_bbox = None
    parsed = []
    for event in events:
        if event["action"] == "grasp":
            current_bbox = event["bbox"]
            parsed.append(event)
        else:
            event["bbox"] = current_bbox
            if current_bbox is None:
                event["error"] = "No previous grasp bbox for this action."
            parsed.append(event)
    return parsed


def parse_modality_keywords(modality_text):
    """Parse text like "(use vision and touch)" into a set of modalities; None means unspecified (load every modality of the action)."""
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
    return keywords or None


def read_force_observation(action_folder):
    force_csv = action_folder / "force" / "data.csv"
    if not force_csv.exists():
        return None

    with open(force_csv, "r") as f:
        rows = list(csv.reader(f))

    values = []
    for row in rows[1:]:
        try:
            values.append(float(row[3]))
        except (ValueError, IndexError):
            continue
    if not values:
        return None

    return f"<force_start> mean gravity force {round(float(np.mean(values)), 2)} N <force_end>"


def load_action_observation(object_folder, action_name, modality=None):
    """Load the observations that exist under the action folder and were requested by the model in "(use modality)"."""
    if action_name not in ACTION_FOLDER_MAP:
        raise Exception(f"Unsupported action: {action_name}")

    action_folder = object_folder / ACTION_FOLDER_MAP[action_name]
    if not action_folder.is_dir():
        raise Exception(f"Action folder does not exist: {action_folder}")

    requested = parse_modality_keywords(modality)
    observations = []

    if requested is None or 'image' in requested:
        image_path = action_folder / "WristCamera" / "25.png"
        if image_path.exists():
            observations.append({"type": "image", "image": str(image_path)})

    if requested is None or 'tactile' in requested:
        gelsight = "GelSight" if (object_folder / "clench" / "GelSight" / "0.png").exists() else "GelSightL"
        path0, path1 = action_folder / gelsight / "0.png", action_folder / gelsight / "25.png"
        if path0.exists() and path1.exists():
            observations.append({"type": "tactile", "tactile": [str(path0), str(path1)], "tactile_bg": str(TACTILE_BG_PATH)})

    if requested is None or 'audio' in requested:
        audio_path = action_folder / "mic" / "audio_enhanced_cut.wav"
        if audio_path.exists():
            observations.append({"type": "audio", "audio": str(audio_path)})

    if requested is None or 'force' in requested:
        force_text = read_force_observation(action_folder)
        if force_text is not None:
            observations.append({"type": "text", "text": force_text})

    if not observations:
        raise Exception(f"No observation found in {action_folder} for requested modality={modality}")

    observations.append({"type": "text", "text": ". "})
    return observations


def extract_final_answer(full_text):
    matches = re.findall(r"answer is\s*[\[\(]?\s*([A-Za-z])(?![A-Za-z])", full_text, re.IGNORECASE)
    return matches[-1].upper() if matches else None


# ============================================================
# Model: loading + streaming multi-round active perception for a single QA
# ============================================================

class StopOnEvent(StoppingCriteria):
    """Once the event is set externally (Stop button / page disconnect), generate() halts at the next token."""

    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


class Run:
    """Full state of one question; blocks is the timeline shown on the page, parts is the assistant content fed back to the model."""

    def __init__(self, question):
        self.question = question
        self.blocks = []
        self.parts = []
        self.recorded = 0
        self.current_object = None
        self.grasp_bbox = None
        self.grasp_label = None
        self.round = 0
        self.status = "running"  # running / done / stopped / error


class RomaEngine:
    def __init__(self, model_path):
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # attn_implementation="flash_attention_2",
        )
        self.model.disable_talker()
        self.model.eval()
        print('model loaded!', torch.cuda.is_available())

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.processor.chat_template = CHAT_TEMPLATE

        # The merged checkpoint normally already carries these tokens; if the tokenizer lacks them, add them in training order.
        vocab = self.tokenizer.get_vocab()
        if any(t not in vocab for t in SPECIAL_TOKENS):
            print("WARNING: special tokens missing from tokenizer, adding them.")
            self.tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})

        tactile_ids = self.tokenizer.convert_tokens_to_ids(['<|TACTILE|>', '<|tactile_bos|>', '<|tactile_eos|>'])
        for cfg in (self.model.config, self.model.thinker.config):
            cfg.tactile_token_id, cfg.tactile_start_token_id, cfg.tactile_end_token_id = tactile_ids

        self.action_id_to_name = {self.tokenizer.convert_tokens_to_ids(f'<{n}>'): n for n in ACTION_FOLDER_MAP}
        self.action_ids = set(self.action_id_to_name)

        eos_ids = set()
        cfg_eos = getattr(getattr(self.model.thinker, "generation_config", None), "eos_token_id", None)
        if cfg_eos is not None:
            eos_ids.update(cfg_eos if isinstance(cfg_eos, (list, tuple, set)) else [cfg_eos])
        for tok in ['<|im_end|>', '<|endoftext|>']:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                eos_ids.add(tid)
        self.stop_ids = self.action_ids | eos_ids
        self.pad_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')

        with open(SCENE_DIR / "save.json", "r", encoding="utf-8") as f:
            self.objects = [o for o in json.load(f) if "bbox_int" in o and "folder_name" in o]
        self.stop_event = threading.Event()

    def request_stop(self):
        self.stop_event.set()

    def _build_inputs(self, messages, parts):
        conv = messages + ([{"role": "assistant", "content": list(parts)}] if parts else [])
        if parts:
            text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for suffix in ['<|im_end|>\n', '<|im_end|>']:
                if text.endswith(suffix):
                    text = text[:-len(suffix)]
                    break
        else:
            text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        audios, images, videos, tactiles = process_mm_info(conv, use_audio_in_video=True)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            tactile=tactiles, return_tensors="pt", padding=True, use_audio_in_video=True,
        )
        return inputs.to(self.model.device).to(self.model.dtype)

    @torch.no_grad()
    def _generate(self, inputs, streamer, stop_event, result):
        try:
            result["output"] = self.model.thinker.generate(
                **inputs,
                use_audio_in_video=True,
                max_new_tokens=THINKER_MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=list(self.stop_ids),
                pad_token_id=self.pad_id,
                streamer=streamer,
                stopping_criteria=StoppingCriteriaList([StopOnEvent(stop_event)]),
            )
        except Exception as e:
            result["error"] = e
            streamer.end()  # lets the main thread's `for chunk in streamer` loop exit

    def _apply_action(self, run, new_text, action_name):
        all_text = "".join(p["text"] for p in run.parts if p["type"] == "text") + new_text
        events = parse_actions(all_text)
        new_events, run.recorded = events[run.recorded:], len(events)

        for event in new_events:
            if event["error"] is not None:
                raise Exception(f"{event['action']}: {event['error']}")
            if event["action"] == "grasp":
                matched, err = match_object(event["bbox"], self.objects)
                if matched is None:
                    raise Exception(f"grasp object matching failed: {err}")
                run.current_object = matched
                run.grasp_bbox = event["bbox"]
                run.grasp_label = matched.get("object_name") or matched["folder_name"]
            elif run.current_object is None:
                raise Exception(f"action <{event['action']}> appears before any grasp.")

        object_folder = SCENE_DIR / run.current_object["folder_name"]
        if not object_folder.is_dir():
            raise Exception(f"Object folder does not exist: {object_folder}")

        modality = new_events[-1]["modality"] if new_events else None
        observations = load_action_observation(object_folder, action_name, modality)

        run.parts.append({"type": "text", "text": new_text + " → observations: "})
        run.parts.extend(observations)
        run.blocks.append({"kind": "obs", "action": action_name, "modality": modality, "items": observations})

    def stream(self, question):
        """Generator: yields the run every time its state is updated."""
        run = Run(question)
        self.stop_event = stop_event = threading.Event()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": str(SCENE_IMAGE)}, {"type": "text", "text": question}]},
        ]
        try:
            while run.round < MAX_ROUNDS:
                block = {"kind": "model", "round": run.round + 1, "text": "", "final": False}
                run.blocks.append(block)
                yield run

                inputs = self._build_inputs(messages, run.parts)
                input_len = inputs['input_ids'].shape[1]

                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
                result = {}
                worker = threading.Thread(target=self._generate, args=(inputs, streamer, stop_event, result), daemon=True)
                worker.start()
                for chunk in streamer:
                    block["text"] += chunk
                    yield run
                # generate() calls streamer.end() before it returns, so wait for the worker to publish its result
                worker.join()

                if "error" in result:
                    raise result["error"]
                if stop_event.is_set():
                    run.status = "stopped"
                    yield run
                    return

                new_ids = result["output"][0, input_len:].tolist()
                stop_idx = next((i for i, t in enumerate(new_ids) if t in self.stop_ids), None)
                stop_id = new_ids[stop_idx] if stop_idx is not None else None
                if stop_idx is not None:
                    new_ids = new_ids[:stop_idx + 1]
                new_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
                new_text = new_text.replace('<|im_end|>', '').replace('<|endoftext|>', '')
                block["text"] = new_text

                if stop_id not in self.action_ids:
                    # Natural end (eos, or in the extreme case max_new_tokens ran out)
                    if new_text.strip():
                        run.parts.append({"type": "text", "text": new_text})
                    block["final"] = True
                    run.status = "done"
                    yield run
                    return

                self._apply_action(run, new_text, self.action_id_to_name[stop_id])
                run.round += 1
                yield run

            run.status = "done"  # rounds exhausted: finish right away, same as eval.py
            yield run
        except Exception as e:
            print(f"[ERROR] {e!r}")
            run.blocks.append({"kind": "error", "text": repr(e)})
            run.status = "error"
            yield run
        finally:
            stop_event.set()  # on page disconnect / generator close, make sure the background generate thread stops too


# ============================================================
# Rendering: turn a Run into HTML (scene image + reasoning timeline)
# ============================================================

ACTION_ICON = {"lift": "⬆️", "squeeze": "🤏", "shake": "〰️", "rotate": "🔄", "collide": "💥", "press": "👇"}
MODALITY_ICON = {'image': '👁️ vision', 'tactile': '✋ touch', 'audio': '🔊 audio', 'force': '⚖️ force'}

_TOKEN_RE = re.compile(
    r"<grasp_start>\s*(?P<box>.*?)\s*(?:<grasp_end>|\Z)"
    r"|(?:\(\s*use\s+(?P<mod>[^)]*?)\s*\)\s*)?<(?P<act>lift|squeeze|shake|rotate|collide|press)>",
    re.S | re.I,
)


def file_url(path):
    return "/gradio_api/file=" + html.escape(Path(path).as_posix(), quote=True)


def format_model_text(text):
    """Escape the text and render grasp / action tokens as small chips."""
    text = text.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    out, pos = [], 0
    for m in _TOKEN_RE.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        if m.group("act"):
            act = m.group("act").lower()
            mod = f" <em>· {html.escape(m.group('mod'))}</em>" if m.group("mod") else ""
            out.append(f'<span class="chip chip-act">{ACTION_ICON[act]} {act}{mod}</span>')
        else:
            out.append(f'<span class="chip chip-grasp">🎯 grasp <code>{html.escape(m.group("box"))}</code></span>')
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def tile(caption, inner):
    return f'<div class="roma-tile">{inner}<div class="roma-cap">{caption}</div></div>'


def render_obs_item(item):
    t = item["type"]
    if t == "image":
        return tile("👁️ Wrist camera", f'<img src="{file_url(item["image"])}">')
    if t == "tactile":
        p0, p1 = item["tactile"]
        return (tile("✋ Tactile · before contact", f'<img src="{file_url(p0)}">')
                + tile("✋ Tactile · during contact", f'<img src="{file_url(p1)}">'))
    if t == "audio":
        return tile("🔊 Contact sound", f'<audio controls preload="metadata" src="{file_url(item["audio"])}"></audio>')
    m = re.search(r"([\d.]+)\s*N", item["text"])
    value = f'<div class="roma-force"><b>{m.group(1)}</b> N</div>' if m else f'<div>{html.escape(item["text"])}</div>'
    return tile("⚖️ Mean gravity force", value)


def render_scene(run):
    box = ""
    if run is not None and run.grasp_bbox is not None:
        x1, y1, x2, y2 = run.grasp_bbox
        x1, x2 = (min(max(v, 0.0), IMG_W) for v in (x1, x2))
        y1, y2 = (min(max(v, 0.0), IMG_H) for v in (y1, y2))
        box = (
            f'<div class="roma-bbox" style="left:{x1 / IMG_W * 100:.2f}%;top:{y1 / IMG_H * 100:.2f}%;'
            f'width:{(x2 - x1) / IMG_W * 100:.2f}%;height:{(y2 - y1) / IMG_H * 100:.2f}%">'
            f'<span>{html.escape(str(run.grasp_label))}</span></div>'
        )
    return (
        '<div class="roma-scene">'
        f'<div class="roma-scene-frame"><img src="{file_url(SCENE_IMAGE)}">{box}</div>'
        '<div class="roma-scene-cap">Initial scene</div>'
        '</div>'
    )


def render_trace(run):
    if run is None:
        return (
            '<div class="roma-empty">Ask a question about the scene. ROMA decides which senses to use, '
            'interacts with the objects, and reasons step by step — live, right here.</div>'
        )

    steps = []
    for i, b in enumerate(run.blocks):
        streaming = run.status == "running" and i == len(run.blocks) - 1
        if b["kind"] == "model":
            body = format_model_text(b["text"]) + ('<span class="roma-cursor"></span>' if streaming else "")
            if b["final"]:
                ans = extract_final_answer(b["text"])
                badge = f'<span class="roma-answer">Answer · {ans}</span>' if ans else ""
                steps.append(f'<div class="roma-step final"><div class="roma-step-h">✅ Final answer{badge}</div><div class="roma-step-b">{body}</div></div>')
            else:
                steps.append(f'<div class="roma-step"><div class="roma-step-h">🧠 Reasoning · round {b["round"]}</div><div class="roma-step-b">{body}</div></div>')
        elif b["kind"] == "obs":
            kws = parse_modality_keywords(b["modality"])
            mods = "".join(f'<span class="chip chip-mod">{MODALITY_ICON[k]}</span>' for k in ('image', 'tactile', 'audio', 'force') if kws and k in kws)
            grid = "".join(render_obs_item(it) for it in b["items"])
            steps.append(
                f'<div class="roma-step obs"><div class="roma-step-h">📡 Observation · {ACTION_ICON[b["action"]]} {b["action"]}{mods}</div>'
                f'<div class="roma-grid">{grid}</div></div>'
            )
        else:
            steps.append(f'<div class="roma-step err"><div class="roma-step-h">⚠️ Error</div><div class="roma-step-b">{html.escape(b["text"])}</div></div>')

    n_actions = sum(1 for b in run.blocks if b["kind"] == "obs")
    if run.status == "running":
        status = '<div class="roma-status"><div class="roma-spin"></div>Perceiving &amp; reasoning…</div>'
    elif run.status == "stopped":
        status = '<div class="roma-status">⏹ Stopped</div>'
    elif run.status == "error":
        status = '<div class="roma-status">Run aborted</div>'
    else:
        status = f'<div class="roma-status">✓ Finished · {n_actions} interaction{"s" if n_actions != 1 else ""}</div>'
    return f'<div class="roma-trace">{"".join(steps)}{status}</div>'


# ============================================================
# UI
# ============================================================

CSS = """
.gradio-container { max-width: 1000px !important; margin: 0 auto !important; }
.roma-header { text-align: center; padding: 22px 0 6px; }
.roma-title { font-size: 2.6rem; font-weight: 800; letter-spacing: .06em;
  background: linear-gradient(90deg, #6366f1, #ec4899, #f59e0b); -webkit-background-clip: text;
  background-clip: text; color: transparent; }
.roma-sub { opacity: .7; margin-top: 4px; }

.roma-scene { max-width: 760px; margin: 8px auto 4px; border-radius: 18px; overflow: hidden;
  border: 1px solid var(--border-color-primary); box-shadow: 0 10px 34px rgba(0,0,0,.14);
  background: var(--background-fill-secondary); }
.roma-scene-frame { position: relative; line-height: 0; overflow: hidden; }
.roma-scene-frame img { width: 100%; display: block; }
.roma-scene-cap { padding: 7px 14px; font-size: .85rem; opacity: .7; text-align: center; }
.roma-bbox { position: absolute; border: 2px solid #f43f5e; border-radius: 6px;
  box-shadow: 0 0 0 9999px rgba(0,0,0,.28); }
.roma-bbox span { position: absolute; top: 0; left: 0; background: #f43f5e; color: #fff; font-size: 12px;
  line-height: 1.5; padding: 0 7px; border-bottom-right-radius: 6px; }

.roma-empty { text-align: center; padding: 34px 20px; opacity: .6; border: 1px dashed var(--border-color-primary);
  border-radius: 16px; }

.roma-trace { position: relative; padding-left: 28px; margin-top: 6px; }
.roma-trace:before { content: ""; position: absolute; left: 8px; top: 8px; bottom: 30px; width: 2px;
  background: linear-gradient(#6366f1, #ec4899); opacity: .35; }
.roma-step { position: relative; margin: 0 0 14px; border: 1px solid var(--border-color-primary);
  border-radius: 14px; background: var(--block-background-fill); box-shadow: 0 2px 10px rgba(0,0,0,.05); }
.roma-step:before { content: ""; position: absolute; left: -26px; top: 15px; width: 12px; height: 12px;
  border-radius: 50%; background: #6366f1; box-shadow: 0 0 0 4px rgba(99,102,241,.2); }
.roma-step.obs:before { background: #10b981; box-shadow: 0 0 0 4px rgba(16,185,129,.2); }
.roma-step.final:before { background: #f59e0b; box-shadow: 0 0 0 4px rgba(245,158,11,.25); }
.roma-step.err:before { background: #ef4444; box-shadow: 0 0 0 4px rgba(239,68,68,.2); }
.roma-step.final { border-color: rgba(245,158,11,.6); }
.roma-step.err { border-color: rgba(239,68,68,.6); }
.roma-step-h { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; padding: 9px 14px; font-weight: 600;
  border-bottom: 1px solid var(--border-color-primary); border-radius: 14px 14px 0 0;
  background: var(--background-fill-secondary); }
.roma-step-b { padding: 12px 14px; white-space: pre-wrap; line-height: 1.7; word-break: break-word; }

.chip { display: inline-flex; align-items: center; gap: 4px; padding: 0 10px; border-radius: 999px;
  font-size: .85em; font-weight: 600; margin: 0 2px; }
.chip code { background: none; padding: 0; color: inherit; font-size: 1em; }
.chip em { font-style: normal; opacity: .8; font-weight: 500; }
.chip-grasp { background: rgba(244,63,94,.14); color: #e11d48; }
.chip-act { background: rgba(99,102,241,.16); color: #6366f1; }
.chip-mod { background: rgba(16,185,129,.16); color: #059669; }

.roma-grid { display: flex; flex-wrap: wrap; gap: 12px; padding: 12px 14px; }
.roma-tile { flex: 1 1 190px; max-width: 290px; min-width: 170px; }
.roma-tile img { width: 100%; border-radius: 10px; display: block; border: 1px solid var(--border-color-primary); }
.roma-tile audio { width: 100%; margin-top: 6px; }
.roma-cap { font-size: .8rem; opacity: .75; margin-top: 5px; }
.roma-force { padding: 16px 8px; text-align: center; font-size: 1.1rem; border-radius: 10px;
  background: rgba(245,158,11,.12); }
.roma-force b { font-size: 2rem; color: #d97706; }

.roma-answer { margin-left: auto; background: linear-gradient(90deg, #f59e0b, #ef4444); color: #fff;
  border-radius: 10px; padding: 1px 12px; font-weight: 700; }
.roma-cursor { display: inline-block; width: 8px; height: 1.1em; margin-left: 2px; background: #6366f1;
  vertical-align: text-bottom; animation: roma-blink 1s steps(2) infinite; }
.roma-status { display: flex; align-items: center; gap: 9px; opacity: .8; font-size: .9rem; padding: 2px 0 6px; }
.roma-spin { width: 14px; height: 14px; border: 2px solid rgba(99,102,241,.25); border-top-color: #6366f1;
  border-radius: 50%; animation: roma-rot .8s linear infinite; }
@keyframes roma-blink { 50% { opacity: 0; } }
@keyframes roma-rot { to { transform: rotate(360deg); } }
"""

HEADER = (
    '<div class="roma-header"><div class="roma-title">ROMA</div>'
    '<div class="roma-sub">Active multisensory perception · vision · touch · audio · force</div></div>'
)

EXAMPLES = [
    "Which object in the scene is the softest?",
    "Is there anything inside the red box? If so, what is it?",
    "Which is heavier, the can or the green cup?",
    "Which objects are made of metal?",
]


def build_ui(engine):
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="pink", radius_size="lg")
    # Gradio 6 moved theme / css from Blocks() to launch()
    style_in_launch = 'css' in inspect.signature(gr.Blocks.launch).parameters
    style = dict(theme=theme, css=CSS)

    def respond(question):
        question = (question or "").strip()
        if not question:
            gr.Warning("Please type a question first.")
            yield render_scene(None), render_trace(None)
            return
        for run in engine.stream(question):
            yield render_scene(run), render_trace(run)

    def clear():
        return "", render_scene(None), render_trace(None)

    with gr.Blocks(title="ROMA Demo", **({} if style_in_launch else style)) as demo:
        gr.HTML(HEADER)
        scene = gr.HTML(render_scene(None))
        with gr.Row(equal_height=True):
            question = gr.Textbox(
                placeholder="Ask a question about the scene, e.g. “Which object is the softest?”",
                lines=2, max_lines=6, show_label=False, container=False, scale=6,
            )
            with gr.Column(scale=1, min_width=170):
                run_btn = gr.Button("Ask ROMA", variant="primary")
                with gr.Row():
                    stop_btn = gr.Button("Stop", variant="stop")
                    clear_btn = gr.Button("Clear")
        gr.Examples(examples=[[q] for q in EXAMPLES], inputs=question, label="Try asking")
        trace = gr.HTML(render_trace(None))

        for trigger in (run_btn.click, question.submit):
            trigger(respond, inputs=question, outputs=[scene, trace], concurrency_limit=1, concurrency_id="gpu")
        stop_btn.click(engine.request_stop, queue=False)
        clear_btn.click(clear, outputs=[question, scene, trace], queue=False)

    return demo, (style if style_in_launch else {})


if __name__ == "__main__":
    engine = RomaEngine(cli_args.model_path)
    demo, launch_style = build_ui(engine)
    demo.queue(max_size=8).launch(
        server_name=cli_args.host,
        server_port=cli_args.port,
        share=cli_args.share,
        allowed_paths=[str(RES_DIR)],
        **launch_style,
    )
