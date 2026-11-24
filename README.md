# GNAL — Semi-supervised Skin Lesion Classification

An **independent implementation** of Graph-Enhanced Neighborhood Association Learning (GNAL) for the HAM10000/ISIC 2018 skin-lesion task, with a separate pretrained-ResNet-50 experiment. This is not the paper authors’ official code or a clinically validated system.

## Reproducibility

- Paper: Chenglin Yu (2026), *Graph-enhanced neighborhood association learning model for semi-supervised skin lesion diagnosis*, Biomedical Signal Processing and Control 119, 109886. DOI: https://doi.org/10.1016/j.bspc.2026.109886
- Pinned dataset: `aee4/ham10000-gnal-fold0-20pct` at commit `5b3b97348122b168f200c20352e32ac8b082c988`
- Splits: 7,010 train / 991 validation / 2,014 test; 20% training-label budget; lesion-level grouping, fold 0
- Training: Kaggle GPU, `notebooks/Train_Model_02.ipynb`
- Code extracted **unchanged** from the notebook-generated modules: `src/gnal_hf.py`, `src/hf_data.py`, and `src/cloud_backup.py`. The notebook itself writes these files into `/kaggle/working` during execution.

## Experiments

| Held-out test metric | Original baseline | Pretrained v1 |
|---|---:|---:|
| Accuracy | 60.53% | 77.16% |
| Macro F1 | 41.06% | 56.46% |
| Macro sensitivity | 48.56% | 56.67% |
| Macro specificity | 93.01% | 94.42% |
| Macro AUC | 74.27% | 78.48% |
| Brier (lower better) | 0.6021 | 0.4384 |
| ECE (lower better) | 0.1769 | 0.2103 |

The original baseline used random ResNet-50 initialization, learning rate 0.001 and 20+100 epochs. Pretrained v1 used Torchvision ImageNet-1K V2 pretrained ResNet-50, ImageNet normalization, learning rate 0.0001 and 5+30 epochs. The rest of the GNAL architecture and fixed data split were preserved. **Since multiple variables changed, this comparison does not isolate the contribution of pretraining.** Best validation macro-F1 checkpoint: epoch 14. The experimental model is not for clinical use.

See [experiment log](docs/experiment_log.md) and `experiments/pretrained_v1/` for config, final metrics, classification report, history and plots.

## Model storage

Training checkpoints (about 289 MB each) remain on Hugging Face, not GitHub. Private repo: `bigj15/gnal-training-checkpoints`, run folder `pretrained-v1-fold0-20pct`. Access is required to retrieve those checkpoints. Do not commit access tokens; use Kaggle secret `HF_TOKEN` or the notebook's hidden prompt.

## How to run

1. Open `notebooks/Train_Model_02.ipynb` on Kaggle with GPU and Internet enabled.
2. Install the dependencies using the notebook's existing installation cell.
3. Run the setup and dataset preparation cells in order. Preserve the pinned revision and split.
4. Review the selected PROFILE and CHECKPOINT_REPO/RUN_ID before running; the provided notebook defaults to completed `pretrained_v1` and will try to resume existing checkpoints.
5. Run training only if intended. For evaluation, use its Results cell with authorized Hugging Face read access.

The notebook contains saved output for reproducibility but also includes session-specific paths and logs. No plaintext HF token was detected in the exported notebook; nevertheless review it before making the repository public.

## Attribution

Dataset preparation and initial notebook setup involved collaborator Eyram. Verify permissions and appropriate credit before public publication. Baseline metrics are retained from the project handoff; original baseline source/checkpoints are not included in this package.
