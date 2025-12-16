# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modify state_dict and config for exporting speculative decoding in official format."""

import re
from copy import copy

import torch
import torch.nn as nn

LLAMA_EAGLE_SINGLE_LAYER = {
    "required": {
        "midlayer.self_attn.q_proj.weight",
        "midlayer.self_attn.k_proj.weight",
        "midlayer.self_attn.v_proj.weight",
        "midlayer.self_attn.o_proj.weight",
        "midlayer.mlp.gate_proj.weight",
        "midlayer.mlp.up_proj.weight",
        "midlayer.mlp.down_proj.weight",
        "midlayer.hidden_norm.weight",
        "midlayer.input_layernorm.weight",
        "midlayer.post_attention_layernorm.weight",
        "norm.weight",
        "fc.weight",
    },
    "optional": {"d2t", "lm_head.weight"},
}


def _check_valid_sd(state_dict: dict, num_hidden_layers: int):
    """Check the export state dict is valid, otherwise raise Exception."""
    # Check that export sd has required keys
    if num_hidden_layers == 1:
        for key in LLAMA_EAGLE_SINGLE_LAYER["required"]:
            assert key in state_dict, f"Missing required key: {key}"
    else:
        for key in LLAMA_EAGLE_SINGLE_LAYER["required"]:
            assert key.replace("midlayer", "midlayer.0") in state_dict, (
                f"Missing required key: {key}"
            )
        for i in range(1, num_hidden_layers):
            for key in LLAMA_EAGLE_SINGLE_LAYER["required"] - {
                "midlayer.hidden_norm.weight",
                "midlayer.input_layernorm.weight",
                "norm.weight",
                "fc.weight",
            }:
                assert key.replace("midlayer", f"midlayer.{i}") in state_dict, (
                    f"Missing required key: {key}"
                )

    # check that export sd has no unexpected keys
    allowed_keys_single_layer = (
        LLAMA_EAGLE_SINGLE_LAYER["required"] + LLAMA_EAGLE_SINGLE_LAYER["optional"]
    )
    if num_hidden_layers == 1:
        for key in state_dict:
            assert key in allowed_keys_single_layer, f"Unexpected key: {key}"
    else:
        for key in state_dict:
            assert re.sub(r"midlayers\.\d+\.", "", "layers.1212.a") in {
                k.replace("midlayer.", "") for k in allowed_keys_single_layer
            }, f"Unexpected key: {key}"


def spec_opt_only(model: nn.Module):
    """Check if the model have only speculative decoding optimization."""
    opt_modes = getattr(model, "_modelopt_state", None)
    return (
        isinstance(opt_modes, (list, tuple)) and len(opt_modes) == 1 and opt_modes[0][0] == "eagle"
    )


def export_spec_ckpt_state_dict(model: nn.Module):
    """Only return the state dict of the draft model in official format and ignore the base model."""
    # check the model has only speculative decoding
    assert spec_opt_only(model), "Not purely eagle model."

    # Rename layers to midlayer
    if model.eagle_config.num_hidden_layers == 1:
        model.eagle_module.midlayer = model.eagle_module._modules.pop("layers")[0]
    else:
        model.eagle_module.midlayer = model.eagle_module._modules.pop("layers")
    export_sd = copy(model.eagle_module.state_dict())

    # Use base model's lm head if draft model doesn't have one
    if "lm_head.weight" not in export_sd:
        export_sd["lm_head.weight"] = model.state_dict()["lm_head.weight"]

    # Rename parallel draft weights
    if model.eagle_config.parallel_draft_step > 1:
        for i in range(model.eagle_config.parallel_draft_step - 1):
            for j in range(model.eagle_config.parallel_draft_heads_num_layers):
                export_sd[f"parallel_draft_heads.{i}.medusa_layers.{j}.linear.weight"] = (
                    export_sd.pop(f"parallel_draft_heads.{i}.{j}.linear.weight")
                )
                if f"parallel_draft_heads.{i}.{j}.linear.bias" in export_sd:
                    export_sd[f"parallel_draft_heads.{i}.medusa_layers.{j}.linear.bias"] = (
                        export_sd.pop(f"parallel_draft_heads.{i}.{j}.linear.bias")
                    )
            export_sd[f"parallel_draft_heads.{i}.lm_head.weight"] = export_sd.pop(
                f"parallel_draft_heads.{i}.{model.eagle_config.parallel_draft_heads_num_layers}.weight"
            )

    _check_valid_sd(export_sd, model.eagle_config.num_hidden_layers)

    return export_sd


def export_spec_ckpt_config(model: nn.Module):
    """Return the config of draft model in official format."""
    assert spec_opt_only(model), "Not purely eagle model."

    # This is the config keys in official checkpoint.
    template_config = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "bos_token_id": None,
        "eos_token_id": None,
        "hidden_act": None,
        "hidden_size": None,
        "initializer_range": None,
        "intermediate_size": None,
        "max_position_embeddings": None,
        "model_type": "llama",
        "num_attention_heads": None,
        "num_key_value_heads": None,
        "num_hidden_layers": None,
        "pad_token_id": None,
        "rms_norm_eps": None,
        "tie_word_embeddings": False,
        "torch_dtype": None,
        "transformers_version": None,
        "use_cache": None,
        "vocab_size": None,
        "draft_vocab_size": None,
        "rope_scaling": None,
        "attention_bias": None,
        "attention_dropout": None,
        "head_dim": None,
        "mlp_bias": None,
        "pretraining_tp": None,
        "rope_theta": None,
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": None,
            "use_aux_hidden_state": None,
            "use_input_layernorm_in_first_layer": None,
            "use_last_layernorm": None,
            "use_mtp_layernorm": None,
            "next_layer_regular": True,
            "parallel_draft_step": None,
            "parallel_draft_heads_num_layers": None,
        },
    }

    def _get_config_from_eagle_config_or_base_config(key: str, model: nn.Module):
        if getattr(model.eagle_config, key, None) is not None:
            return getattr(model.eagle_config, key)
        elif getattr(model.config, key, None) is not None:
            return getattr(model.config, key)
        else:
            return None

    for key in template_config:
        value = template_config[key]
        if isinstance(value, dict):
            # for eagle config, we find it in model.eagle_config
            for sub_key in value:
                if value[sub_key] is None:
                    value[sub_key] = _get_config_from_eagle_config_or_base_config(sub_key, model)
        elif value is None:
            # First, we try to load fron eagle config.
            new_value = _get_config_from_eagle_config_or_base_config(key, model)
            # If the value is a torch.dtype, we convert to string for serialization.
            if isinstance(new_value, torch.dtype):
                new_value = str(new_value).replace("torch.", "")
            template_config[key] = new_value

    return template_config
