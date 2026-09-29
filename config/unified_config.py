
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers

@dataclass
class ModelArguments:  
    # llm
    model_name_or_path: Optional[str] = field(default="resources/models/Qwen2.5-Omni-7B")
    freeze_backbone: bool = field(default=True, metadata={"help": "Whether to freeze the LLM backbone."})
    freeze_audio_encoder: bool = field(default=True, metadata={"help": "Whether to freeze the audio encoder."})
    freeze_audio_adapter: bool = field(default=True, metadata={"help": "Whether to freeze the audio adapter."})
    freeze_tactile_encoder: bool = field(default=True, metadata={"help": "Whether to freeze the tactile encoder."})
    freeze_tactile_adapter: bool = field(default=True, metadata={"help": "Whether to freeze the tactile adapter."})
    llm_name: str = field(default='qwen')
    debug_mode: bool = field(default=False)

@dataclass
class InferenceArguments:
    # used for inference
    ckpt_dir: str = field(default='')
    
    # for infer avs
    adapter_ckpt_path: str = field(default=None)
    test_name: str = field(default='test') # for ref-avs: test_u,test_s,test_n
    cut_folds:  int = field(default=2)

    device: str = field(default='cuda:0')
    

@dataclass
class DataArguments:
    # pretrain
    video_frame_nums: int = field(default=8)
    # image_size = ModelArguments.image_size
    # image_caption_task: bool = field(default=False)
    # video_caption_task: bool = field(default=False)
    # audio_caption_task: bool = field(default=False)
    # # fine-tune
    # avqa_task: bool = field(default=False)
    # ave_task: bool = field(default=False)
    # multi_frames: bool = field(default=False) # avs task input single frame



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=32,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 444
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    ## my
    reserved_modality: str = field(default=None)
    loramethod: str = field(default=None)
    blc_alpha: float = field(default=0.5)
    blc_weight: float = field(default=0.5)

    audio_branch: bool = field(default=False)
    visual_branch: bool = field(default=False)

    save_modules: str = field(default='vl_projector,al_projector,lora')

    exp_desc: str = field(default='exp')
    
    dataset_name: str = field(default='audio')

    audio_weight_path: str = field(default=None)
    tactile_weight_path: str = field(default=None)

    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.99)
    adam_epsilon: float = field(default=1e-8)
    max_grad_norm: float = field(default=0)