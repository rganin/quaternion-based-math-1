"""Full-model conversion driver (Stage 4 / Phases 4-5).

Iterate over Transformer blocks, capture reference outputs, fit quaternion
parameters, evaluate reconstruction and feed the reconstructed state forward
(sequential calibration).  Checkpoint after every block so conversion can be
resumed.  See plan.md sections 7, 23-24 and 31.
"""
