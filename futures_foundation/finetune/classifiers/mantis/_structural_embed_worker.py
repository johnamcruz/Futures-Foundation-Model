"""Isolated torch worker for frozen Structural NextLeg features and ONNX export."""
import json
import sys
from pathlib import Path

import numpy as np


def main(directory):
    directory = Path(directory)
    cfg = json.loads((directory / "cfg.json").read_text())
    from futures_foundation.finetune.pretext._torch.structural_inference import (
        export_structural_encoder_onnx, structural_features)
    if cfg.get("_export_encoder"):
        export_structural_encoder_onnx(
            cfg.pop("_export_encoder"), encoder_ckpt=cfg.pop("ckpt"),
            trainer_ckpt=cfg.pop("trainer_ckpt"), C=int(cfg.pop("C", 5)),
            seq=int(cfg.pop("seq", 128)), **cfg)
        return
    windows = np.load(cfg.pop("_windows"), mmap_mode="r")
    features = structural_features(
        windows, encoder_ckpt=cfg.pop("ckpt"), trainer_ckpt=cfg.pop("trainer_ckpt"), **cfg)
    np.save(directory / "emb.npy", features)


if __name__ == "__main__":
    main(sys.argv[1])
