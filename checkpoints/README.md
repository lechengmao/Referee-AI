# Checkpoints

No trained foul-recognition checkpoint is included. Training writes `best.pt`
and `last.pt` beneath the explicit run output directory. Load with the matching
resolved configuration and this repository's evaluation script. The upstream
VARS checkpoint format is not directly compatible.
