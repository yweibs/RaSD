# RaSD

**RaSD** (Randomized Synthesis and Disentanglement) is a scalable framework for pre-training MIFMs entirely on synthetic data.
---

## Project Structure (summary)
- `gendata/`: Tools for generating synthetic images and labels.
- `pretrain/`: 3D pretraining scripts and utilities (`train_gen/`, `train_RaSD_online/`, `train_RaSD_offline/`).
- `pretrain2d/`: 2D training script (`train_2D.py`).
- `Downstream/`: Downstream task code.

---

## Quick Start
```bash
# Clone the repo
git clone <your-repo-url>
cd RaSD
```

### Environment
This repository includes an `environment.yml` with core dependencies (MONAI, SimpleITK, etc.). Create the conda environment and activate it:
```bash
conda env create -f environment.yml
conda activate rasd
```

Note: PyTorch is not pinned in `environment.yml` because the correct wheel depends on your CUDA runtime. After activating the environment, install PyTorch following the official guide.

## Pretrained models
Pretrained checkpoints are available at:

https://drive.google.com/drive/folders/1aGwI2NihPzqsWH65t7uQe9Th8hOfMtD2?usp=drive_link

Download the checkpoint(s) you need and place them in a local folder.

## Data Generation (Synthetic)
- Script: `gendata/gen_seg.py`. Core utilities are in `gendata/utils/gen_img.py` and `gendata/utils/gen_lab.py`.
- Usage (generate NIfTI files to disk):
```bash
# generate 1000 samples into /path/to/data, images in Image/, labels in Label/
python gendata/gen_seg.py --data_amount 1000 --root /path/to/data --img_dir Image --lab_dir Label
```
- Useful options:
  - `--data_amount`: number of synthetic samples (default 1000)
  - `--shape`: spatial size used by generator (default 128)
  - `--num_label`: optional override for number of labels (default shape//16)
  - `--dry_run`: generate but do not write files (quick test)


## Pretraining
Below are the common ways to run pretraining.

### 3D - Online (generate on-the-fly during training)
- Script: `pretrain/train_RaSD/train_RaSD.py` (uses `DatasetGEN3D` for on-the-fly generation)
- Example command (change parameters as needed):
```bash
# online training, small test run
python pretrain/train_RaSD/train_RaSD.py --n_channels 1 --lr 1e-5 --epoches 1 --iters 100 --batch_size 1 --checkpoint_dir weights --result_dir results --model_name RaSD --shape 128
```
- Available params (examples): `--n_channels`, `--lr`, `--epoches`, `--iters`, `--batch_size`, `--checkpoint_dir`, `--result_dir`, `--model_name`, `--shape`, `--k` (start epoch offset), `--load` (load checkpoint).

### 3D - Offline (generate dataset first, then train from disk)
1. Generate synthetic dataset using `gendata/gen_seg.py` (see above).
2. Train with cached dataset loader using `pretrain/train_RaSD_offline/train_RaSD_offline.py`:
```bash
# offline training from generated files
python pretrain/train_RaSD_offline/train_RaSD_offline.py --train_dir /path/to/data --n_channels 1 --lr 1e-5 --epoches 1 --iters 100 --batch_size 1 --checkpoint_dir weights --model_name RaSD
```
- Important: `--train_dir` should contain `Image/` and `Label/` subdirectories with matching NIfTI files.

### 2D training
- Script: `pretrain2d/train_2D.py`
- Example:
```bash
python pretrain2d/train_2D.py --n_channels 1 --lr 1e-5 --epoches 10 --iters 200 --batch_size 16 --checkpoint_dir ckpt_2D --model_name RaSD_2D
```

Notes:
- All training scripts accept `--cuda_devices` (e.g. `0` or `0,1`) to set `CUDA_VISIBLE_DEVICES` and `--load` to load an existing checkpoint before training.

---

## Downstream datasets
We list common public datasets used in our paper and experiments in `docs/DATASETS.md`. We do not host the datasets in this repository; please follow the links and any registration/licensing steps for each dataset.

## Downstream tasks
For downstream experiments, check the `Downstream/` directory. Each dataset has its own subfolder named after the dataset (for example, `Downstream/ACDC`), which contains dataset-specific code (data utilities, models, trainers). 
