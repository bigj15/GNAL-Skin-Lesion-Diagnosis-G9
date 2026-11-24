# Experiment log

## Original baseline (from project handoff)
- Full GNAL, ResNet-50 without pretrained weights, pixel values in [0,1], LR 0.001.
- 20 warmup + 100 main epochs, batch size 12, image size 224, seed 42, fixed 20%-label fold.
- Held-out test: accuracy 60.53%, macro F1 41.06%, macro sensitivity 48.56%, macro specificity 93.01%, macro AUC 74.27%, Brier 0.602129, ECE 0.176855.
- Original checkpoint run `aee4/gnal-training-checkpoints/paper-fold0-20pct` was intentionally left untouched.

## Pretrained v1
- Kept the pinned HF dataset revision, splits, 20% label budget, fold 0, seed 42, batch size 12, image size 224, NCFR/ReGA modules and loss weights.
- Switched ResNet-50 to Torchvision pretrained `IMAGENET1K_V2` weights and enabled ImageNet normalization.
- Changed learning rate 0.001 → 0.0001; duration 20+100 → 5+30 epochs.
- Introduced `pretrained_v1` profile, separate local output and private HF checkpoint run `bigj15/gnal-training-checkpoints/pretrained-v1-fold0-20pct`.
- Kaggle session interrupted; training restored from completed epoch 23 and completed epoch 35. Results and artifacts confirmed backed up to HF.
- Best validation macro F1 61.31% at epoch 14; final test from selected checkpoint: accuracy 77.1599%, macro F1 56.4580%, sensitivity 56.6715%, specificity 94.4200%, AUC 78.4770%, Brier 0.438354, ECE 0.210256.
- All per-class F1 scores improved vs original rounded baseline. DF still weak (F1 0.1429, recall 2/22), and MEL often confused with NV (78/222 MEL → NV).

## Next analyses
- Compare pretrained CNN-only to full GNAL under matched data and training conditions.
- Isolate NCFR and ReGA with ablations, and assess class imbalance and calibration using validation data.
- Avoid repeated test-set tuning; use the held-out split only for planned final comparisons.
