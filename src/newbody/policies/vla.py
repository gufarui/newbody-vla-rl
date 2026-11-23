"""Load a fine-tuned SmolVLA checkpoint and turn simulator observations into env actions.

The VLA sees ONLY what a real robot would have: two camera images, the 5-D proprioceptive
state (same `schema.state_vector` as in the dataset) and the language instruction.

Pipeline per step (same as lerobot's own eval loop):
    raw obs -> preprocessor (rename cameras, tokenize task, normalize state, to device)
            -> policy.select_action (pops from an action queue; re-predicts a 50-step chunk
               every `n_action_steps` steps)
            -> postprocessor (un-normalize) -> env action
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from newbody.data import schema


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VLAController:
    def __init__(self, checkpoint: str | Path, task: str, n_action_steps: int = 50, device: str = "auto"):
        self.device = pick_device(device)
        self.task = task
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint))
        self.policy.config.device = self.device
        # How many actions of each predicted 50-step chunk are executed before re-planning.
        self.policy.config.n_action_steps = n_action_steps
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.chunk_times: list[float] = []  # seconds per new chunk prediction

    def reset(self):
        self.policy.reset()  # clears the action queue
        self.pre.reset()
        self.post.reset()

    def _observation(self, obs: dict) -> dict:
        out = {}
        for cam, key in schema.CAMERAS.items():
            img = torch.from_numpy(schema.get_image(obs, cam)).permute(2, 0, 1).float() / 255.0
            out[key] = img.unsqueeze(0)
        out["observation.state"] = torch.from_numpy(schema.state_vector(obs)).unsqueeze(0)
        out["task"] = [self.task]
        return out

    # ---------------------------------------------------------------- features for residual RL
    def enable_feature_capture(self):
        """Capture pooled VLM prefix features from the SAME forward pass that generates actions.

        SmolVLA runs the VLM once over the prefix (64+64 image tokens, 48 language tokens, 1 state
        token), caches it, then denoises with the action expert. The VLM's prefix hidden states are
        normally discarded; we keep the mean of each camera's 64 image tokens, concatenated (1920-D).
        No extra forward pass, no weight change.
        """
        model = self.policy.model
        vlm = model.vlm_with_expert
        orig_embed_prefix = type(model).embed_prefix
        orig_forward = type(vlm).forward
        self._pad_mask = None
        self.last_features = None
        self.feature_calls = 0

        def embed_prefix(*args, **kwargs):
            embs, pad, att = orig_embed_prefix(model, *args, **kwargs)
            self._pad_mask = pad
            return embs, pad, att

        def forward(*args, inputs_embeds=None, **kwargs):
            out, cache = orig_forward(vlm, *args, inputs_embeds=inputs_embeds, **kwargs)
            if inputs_embeds is not None and inputs_embeds[1] is None:  # prefix-only pass
                h = out[0].float()  # (B, L=177, 960): [64 front-cam | 64 wrist-cam | 48 language | 1 state]
                m = self._pad_mask[:, : h.shape[1]].unsqueeze(-1).float()
                # Per-camera mean pooling (measured: front-cam tokens predict the can-gripper xy offset with
                # R2 0.81/0.96 vs 0.67/0.87 for a mean over all tokens; results/probe_vla_features_rollout.json).
                self.last_features = torch.cat([h[:, :64].mean(1), h[:, 64:128].mean(1)], dim=-1).detach()  # (B, 1920)
                self.last_hidden, self.last_mask = h.detach(), m.detach()  # for analysis only
                self.feature_calls += 1
            return out, cache

        model.embed_prefix = embed_prefix
        vlm.forward = forward

    @torch.inference_mode()
    def predict_chunk(self, obs: dict, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """One VLA call: returns (reference chunk (n_steps, 4) in env units, features (1920,)).
        Gripper column is binarized by sign, xyz clipped to [-1, 1]."""
        batch = self.pre(self._observation(obs))
        t0 = time.perf_counter()
        chunk = self.policy.predict_action_chunk(batch)[:, :n_steps]  # (1, n, 4) normalized
        chunk = self.post(chunk).squeeze(0).float().cpu().numpy()
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.chunk_times.append(time.perf_counter() - t0)
        ref = np.clip(chunk, -1.0, 1.0).astype(np.float32)
        ref[:, 3] = np.where(chunk[:, 3] > 0, 1.0, -1.0)
        return ref, self.last_features.squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def act(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Returns (env_action, raw_vla_action). Gripper is binarized by sign (+1 close, -1 open)."""
        new_chunk = len(self.policy._queues["action"]) == 0
        t0 = time.perf_counter()
        batch = self.pre(self._observation(obs))
        action = self.post(self.policy.select_action(batch))
        if new_chunk:
            if self.device == "cuda":
                torch.cuda.synchronize()
            self.chunk_times.append(time.perf_counter() - t0)
        raw = action.squeeze(0).float().cpu().numpy()
        env_action = np.clip(raw, -1.0, 1.0).astype(np.float32)
        env_action[3] = 1.0 if raw[3] > 0 else -1.0
        return env_action, raw
