"""LoRA recovery on top of a frozen quaternion base (Stage 6 / Phase 6).

Planned architecture:

    Y = QuaternionLinear(X) + scale * ((X @ A.T) @ B.T)

with ``A: [rank, in_features]``, ``B: [out_features, rank]`` and
``scale = alpha / rank``.  Quaternion parameters stay frozen; only ``A`` and
``B`` are trainable.

See plan.md sections 18-20 and 31 (PHASE 6).
"""

from __future__ import annotations

__all__: list[str] = []
