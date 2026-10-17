"""Convert the LAION aesthetic predictor v1 linear head (.pth) into the bundled .npz.

Usage: python convert_aesthetic_head.py sa_0_4_vit_b_16_linear.pth photocure/models/<out>.npz
Source: https://github.com/LAION-AI/aesthetic-predictor (MIT). Then update MANIFEST.json.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
sd = torch.load(src, map_location="cpu", weights_only=True)
w = sd["weight"].detach().numpy().reshape(-1).astype(np.float32)
b = np.float32(sd["bias"].detach().numpy().reshape(-1)[0])
if w.shape != (512,):
    sys.exit(f"expected a 512-dim head, got {w.shape}")
np.savez(dst, w=w, b=b, kind="linear10", name="laion-sa-v1-vit-b-16",
         source_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
print(dst, hashlib.sha256(dst.read_bytes()).hexdigest())
