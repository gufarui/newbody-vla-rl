"""Look inside ONE real SmolVLA training step, using the official lerobot-train pipeline.

We do not re-implement anything: we wrap a few SmolVLA functions with "spies" that print the
tensors flowing through them, then run the official trainer for one step on our dataset.

    python scripts/inspect_smolvla_batch.py            # runs on MPS (Mac) or CUDA if available
Writes results/inspect_smolvla_batch.json (used by docs/tutorial/ch02).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

import lerobot.policies.smolvla.modeling_smolvla as ms
from lerobot.scripts import lerobot_train

ROOT = Path(__file__).resolve().parents[1]
REPORT: dict = {}
_done = defaultdict(bool)


def shape(x):
    return list(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__


def once(key):
    if _done[key]:
        return False
    _done[key] = True
    return True


# 1. Policy-level forward: the batch after the preprocessor (renamed, normalized, tokenized).
_orig_policy_forward = ms.SmolVLAPolicy.forward


def policy_forward(self, batch, *a, **kw):
    if once("batch"):
        REPORT["batch_after_preprocessor"] = {k: shape(v) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        s, act = batch["observation.state"], batch["action"]
        REPORT["normalized_state_example"] = s[0].tolist()
        REPORT["normalized_action_first_step_example"] = act[0, 0].tolist()
        REPORT["action_is_pad_count"] = int(batch["action_is_pad"].sum()) if "action_is_pad" in batch else None
        tok = batch["observation.language.tokens"][0]
        mask = batch["observation.language.attention_mask"][0].bool()
        REPORT["language_tokens_used"] = int(mask.sum())
        REPORT["task"] = batch.get("task", [None])[0]
        trainable = defaultdict(int)
        frozen = defaultdict(int)
        for name, p in self.named_parameters():
            group = ".".join(name.split(".")[:3])
            (trainable if p.requires_grad else frozen)[group] += p.numel()
        REPORT["trainable_params_by_module"] = dict(sorted(trainable.items(), key=lambda kv: -kv[1]))
        REPORT["frozen_params_by_module"] = dict(sorted(frozen.items(), key=lambda kv: -kv[1]))
    out = _orig_policy_forward(self, batch, *a, **kw)
    if once("loss"):
        REPORT["loss"] = float(out[0].detach())
        REPORT["loss_dict"] = out[1]
    return out


# 2. Images after SmolVLA's own preprocessing (resize-with-pad to 512, scale to [-1, 1]).
_orig_prepare_images = ms.SmolVLAPolicy.prepare_images


def prepare_images(self, batch):
    images, masks = _orig_prepare_images(self, batch)
    if once("images"):
        REPORT["images_into_model"] = [{"shape": shape(i), "min": float(i.min()), "max": float(i.max())} for i in images]
    return images, masks


# 3. Prefix = what the VLM sees (image tokens + language tokens + state token).
_orig_embed_prefix = ms.VLAFlowMatching.embed_prefix


def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
    embs, pad, att = _orig_embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=state)
    if once("prefix"):
        REPORT["prefix_tokens"] = {"embeddings": shape(embs), "n_tokens": embs.shape[1],
                                   "n_language_tokens_padded": lang_tokens.shape[1]}
    return embs, pad, att


# 4. Flow matching: noisy actions x_t and velocity target u_t.
_orig_fm = ms.make_flow_matching_inputs


def make_fm(actions, noise, time, **kw):
    x_t, u_t, t = _orig_fm(actions, noise, time, **kw)
    if once("fm"):
        REPORT["flow_matching"] = {
            "clean_action_chunk": shape(actions), "noise": shape(noise), "time_per_sample": time.tolist(),
            "x_t": shape(x_t), "target_velocity_u_t": shape(u_t),
            "check_x_t_equals_t*noise+(1-t)*actions": bool(torch.allclose(
                x_t, time[:, None, None] * noise + (1 - time[:, None, None]) * actions, atol=1e-5)),
            "check_u_t_equals_noise-actions": bool(torch.allclose(u_t, noise - actions, atol=1e-5)),
        }
    return x_t, u_t, t


# 5. Suffix = the action expert's input: 50 noisy-action tokens (+ time).
_orig_embed_suffix = ms.VLAFlowMatching.embed_suffix


def embed_suffix(self, noisy_actions, timestep):
    out = _orig_embed_suffix(self, noisy_actions, timestep)
    if once("suffix"):
        REPORT["suffix_tokens"] = {"noisy_actions_in": shape(noisy_actions), "embeddings": shape(out[0])}
    return out


ms.SmolVLAPolicy.forward = policy_forward
ms.SmolVLAPolicy.prepare_images = prepare_images
ms.VLAFlowMatching.embed_prefix = embed_prefix
ms.make_flow_matching_inputs = make_fm
ms.VLAFlowMatching.embed_suffix = embed_suffix


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    sys.argv = [
        "lerobot-train", "--policy.path=lerobot/smolvla_base", f"--policy.device={device}",
        "--policy.push_to_hub=false", "--dataset.repo_id=local/kinova3_pickplace_can_v1_train",
        f"--dataset.root={ROOT}/data/kinova3_pickplace_can_v1/train",
        '--rename_map={"observation.images.agentview": "observation.images.camera1", '
        '"observation.images.wrist": "observation.images.camera2"}',
        "--batch_size=2", "--steps=1", "--log_freq=1", "--num_workers=0", "--save_checkpoint=false",
        f"--output_dir={ROOT}/outputs/inspect", "--job_name=inspect", "--seed=0",
    ]
    import shutil
    shutil.rmtree(ROOT / "outputs/inspect", ignore_errors=True)
    lerobot_train.main()
    out = ROOT / "results/inspect_smolvla_batch.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(REPORT, open(out, "w"), indent=2, default=str)
    print(json.dumps(REPORT, indent=2, default=str))


if __name__ == "__main__":
    main()
