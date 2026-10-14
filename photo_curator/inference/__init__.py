"""GPU inference layer: face detection + semantic (CLIP) classification.

Backends are pluggable so the pipeline runs identically in production
(CUDA + InsightFace + OpenCLIP) and in offline test/CI environments
(deterministic CPU stubs). Backend selection is automatic but overridable.
"""
