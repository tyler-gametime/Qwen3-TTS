# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

target_speaker_embedding = None


def compute_mtp_loss_weight(step: int, warmup_steps: int) -> float:
    """Compute MTP loss weight with curriculum warmup.

    The MTP loss weight (α) increases from 0 to 1 over the warmup period.
    This allows the model to first learn good AR representations before
    learning parallel prediction.

    Args:
        step: Current training step (global).
        warmup_steps: Number of steps for warmup.

    Returns:
        MTP loss weight in range [0, 1].
    """
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def compute_mtp_loss(
    mtp_module,
    talker_hidden_states: torch.Tensor,
    cb0_embeds: torch.Tensor,
    codec_ids: torch.Tensor,
    num_mtp_heads: int,
) -> torch.Tensor:
    """Compute MTP loss for parallel codebook prediction.

    Args:
        mtp_module: The MTPModule instance.
        talker_hidden_states: Hidden states from talker, shape (N, hidden_size).
        cb0_embeds: Embeddings of first codebook tokens, shape (N, hidden_size).
        codec_ids: Target codec IDs, shape (N, num_code_groups).
        num_mtp_heads: Number of MTP heads (codebooks to predict in parallel).

    Returns:
        MTP loss scalar.
    """
    # Labels are cb1 to cb(num_mtp_heads)
    mtp_labels = codec_ids[:, 1 : 1 + num_mtp_heads]

    # Forward through MTP module
    mtp_output = mtp_module(
        talker_hidden=talker_hidden_states,
        cb0_embed=cb0_embeds,
        labels=mtp_labels,
    )

    return mtp_output.loss


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    # MTP training arguments
    parser.add_argument("--use_mtp", action="store_true", help="Enable MTP training")
    parser.add_argument("--num_mtp_heads", type=int, default=4, help="Number of MTP prediction heads")
    parser.add_argument("--mtp_warmup_steps", type=int, default=1000, help="MTP loss warmup steps")
    parser.add_argument("--mtp_loss_weight", type=float, default=0.5, help="Max MTP loss weight after warmup")
    args = parser.parse_args()

    # Create output directory for logging
    os.makedirs(args.output_model_path, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_model_path,
    )

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # Initialize MTP module if enabled
    mtp_module = None
    if args.use_mtp:
        from qwen_tts.core.mtp import MTPConfig, MTPModule

        # Access talker config - may be object or dict depending on how it was loaded
        talker_cfg = config.talker_config
        if hasattr(talker_cfg, "hidden_size"):
            # It's an object with attributes
            hidden_size = talker_cfg.hidden_size
            hidden_act = talker_cfg.hidden_act
            rms_norm_eps = talker_cfg.rms_norm_eps
            code_pred_cfg = talker_cfg.code_predictor_config
            vocab_size = code_pred_cfg.vocab_size if hasattr(code_pred_cfg, "vocab_size") else code_pred_cfg["vocab_size"]
        else:
            # It's a dict
            hidden_size = talker_cfg["hidden_size"]
            hidden_act = talker_cfg["hidden_act"]
            rms_norm_eps = talker_cfg["rms_norm_eps"]
            vocab_size = talker_cfg["code_predictor_config"]["vocab_size"]

        mtp_config = MTPConfig(
            num_mtp_heads=args.num_mtp_heads,
            hidden_size=hidden_size,
            trunk_hidden_size=hidden_size,
            num_trunk_layers=2,
            vocab_size=vocab_size,
            hidden_act=hidden_act,
            rms_norm_eps=rms_norm_eps,
            warmup_steps=args.mtp_warmup_steps,
        )
        mtp_module = MTPModule(mtp_config)
        accelerator.print(f"MTP enabled with {args.num_mtp_heads} heads, warmup={args.mtp_warmup_steps} steps")

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    # Setup optimizer - include MTP module parameters if enabled
    params_to_optimize = list(qwen3tts.model.parameters())
    if mtp_module is not None:
        params_to_optimize.extend(mtp_module.parameters())

    optimizer = AdamW(params_to_optimize, lr=args.lr, weight_decay=0.01)

    # Prepare with accelerator
    if mtp_module is not None:
        model, mtp_module, optimizer, train_dataloader = accelerator.prepare(
            qwen3tts.model, mtp_module, optimizer, train_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(qwen3tts.model, optimizer, train_dataloader)

    num_epochs = args.num_epochs
    model.train()
    if mtp_module is not None:
        mtp_module.train()

    global_step = 0

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                # Compute combined loss
                loss = outputs.loss + 0.3 * sub_talker_loss

                # Add MTP loss if enabled
                mtp_loss = None
                if mtp_module is not None and talker_hidden_states.shape[0] > 0:
                    # Get cb0 embeddings for MTP
                    cb0_embeds = model.talker.model.codec_embedding(talker_codec_ids[:, 0])

                    mtp_loss = compute_mtp_loss(
                        mtp_module=mtp_module,
                        talker_hidden_states=talker_hidden_states,
                        cb0_embeds=cb0_embeds,
                        codec_ids=talker_codec_ids,
                        num_mtp_heads=args.num_mtp_heads,
                    )

                    # Apply curriculum warmup
                    alpha = compute_mtp_loss_weight(global_step, args.mtp_warmup_steps)
                    mtp_weight = alpha * args.mtp_loss_weight
                    loss = loss + mtp_weight * mtp_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    if mtp_module is not None:
                        accelerator.clip_grad_norm_(mtp_module.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

            if step % 10 == 0:
                loss_str = f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                if mtp_loss is not None:
                    loss_str += f" | MTP Loss: {mtp_loss.item():.4f}"
                accelerator.print(loss_str)

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, encoding="utf-8") as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {args.speaker_name: 3000}
            talker_config["spk_is_dialect"] = {args.speaker_name: False}

            # Save MTP config if enabled
            if args.use_mtp:
                talker_config["use_mtp"] = True
                talker_config["num_mtp_heads"] = args.num_mtp_heads
                talker_config["mtp_trunk_layers"] = 2

            config_dict["talker_config"] = talker_config

            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            # Add MTP module weights if enabled
            if mtp_module is not None:
                unwrapped_mtp = accelerator.unwrap_model(mtp_module)
                for k, v in unwrapped_mtp.state_dict().items():
                    state_dict[f"talker.mtp_module.{k}"] = v.detach().to("cpu")

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict["talker.model.codec_embedding.weight"]
            state_dict["talker.model.codec_embedding.weight"][3000] = (
                target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            )
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)


if __name__ == "__main__":
    train()
