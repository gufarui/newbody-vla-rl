"""Collect scripted-expert demos into LeRobot datasets.

    python scripts/collect_demos.py --name kinova3_lift_v1               # train + val per config
    python scripts/collect_demos.py --name smoke --splits train --episodes 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import yaml

from newbody.data.record import record

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/lift_kinova3.yaml"))
    p.add_argument("--name", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--episodes", type=int, default=None, help="override episodes per split")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["env"]["cube_xy_range"] = cfg["data"]["cube_xy_range"]
    out = ROOT / "data" / args.name
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    yaml.safe_dump(cfg, open(out / "collection_config.yaml", "w"))
    for split in args.splits:
        s = cfg["data"]["splits"][split]
        t0 = time.time()
        summary = record(cfg, out, split, args.episodes or s["episodes"], s["seed_start"])
        summary["wall_time_s"] = round(time.time() - t0, 1)
        json.dump(summary, open(out / f"{split}_summary.json", "w"), indent=2)
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
