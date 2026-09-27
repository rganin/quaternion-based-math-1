"""Fit all eligible quaternion projections in a Transformer block (Stage 3).

Optimize attention (q/k/v/o), then MLP (gate/up/down), then jointly, against
the original block output using a combined MSE + cosine objective.  See
plan.md section 6 and 31 (PHASE 3).
"""
