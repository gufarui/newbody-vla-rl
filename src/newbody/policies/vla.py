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
