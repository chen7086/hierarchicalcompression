import torch

from model.qwen3 import (
    Qwen3BiForMNTPandSentEmbeddingsV2,
    Qwen3BiForMNTPandSentEmbeddingsV2_w_q_token
)

from args import parse_args

from munch import Munch
from transformers import AutoTokenizer, AutoConfig

from transformers import (
    Qwen3ForCausalLM
)

from typing import Optional, List

from util.util import SpecTokenType, SentenceEmbeddingType

from peft import get_peft_model, LoraConfig


class ModelType:
    QWEN3='qwen3'

MODEL_TYPE_2_CLS = {
    SentenceEmbeddingType.AVG: {
        ModelType.QWEN3: Qwen3BiForMNTPandSentEmbeddingsV2,
    },
}


def ensure_model_type(model_name_or_path):
    if 'qwen3' in model_name_or_path.lower():
        return ModelType.QWEN3
    else:
        raise ValueError('Unsupported model type: should be on of ["Qwen3"]')

class Qwen3ForCausalLMMock(Qwen3ForCausalLM):
    def __init__(self, config, model_body, lm_head):
        super().__init__(config)

        self.model = model_body
        self.lm_head = lm_head

def get_model_mock_class(model_type: int):
    if model_type == ModelType.QWEN3:
        return Qwen3ForCausalLMMock
    raise ValueError(f'Unsupported mock model type: {model_type}')


def build_model(config_path, tokenizer_name_or_path):
    model_args, data_args, training_args, custom_args = parse_args(
        config_path
    )
    return build_model_from_args(model_args, custom_args, tokenizer_name_or_path)


def build_model_from_args(model_args, custom_args, tokenizer_name_or_path):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        "padding_side": "left"
    }

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    # blank, eos, mask
    if tokenizer.mask_token is None:
        if custom_args.mask_token_type == "blank":
            tokenizer.mask_token = "_"
        elif custom_args.mask_token_type == "eos":
            tokenizer.mask_token = tokenizer.eos_token
        elif custom_args.mask_token_type == "mask":
            tokenizer.add_tokens(["<mask>"])
            tokenizer.mask_token = "<mask>"
        else:
            raise ValueError(
                f"mask_token_type {custom_args.mask_token_type} is not supported."
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # tokenizer.padding_side  = 'left'

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        raise NotImplementedError("Not implemented")

    config.contrastive_loss_scale = custom_args.contrastive_loss_scale

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model_typename = ensure_model_type(model_args.model_name_or_path)

    sentence_embedding_type = SentenceEmbeddingType.AVG

    model_cls = MODEL_TYPE_2_CLS[sentence_embedding_type][model_typename]

    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation=model_args.attn_implementation,
    )

    return Munch(
        # model stuff
        config=config,
        model=model,
        tokenizer=tokenizer,
        model_type=model_typename,
        # args
        model_args=model_args,
        custom_args=custom_args,
    )


def resize_model_embeddings_to_fit_tokenizer(model, tokenizer):
    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    return model


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "Qwen2Config",
        "Qwen3Config",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model
