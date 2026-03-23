# IDL S26 - Guided Project: Diffusion

## Starter Code Usage (local machine)

**Training**

```
python train.py --config configs/ddpm.yaml
```

**Inference and Evaluating**

```
python inference.py --config configs/ddpm.yaml --ckpt /path/to/checkpoint
```

Set up a local Python environment with `setup_env_modern.sh` (Python 3.9+) or `setup_env_368.sh` (Python 3.6.8 where available). See `requirements.txt` / `requirements-py36.txt`.

### PSC / Colab adapters (this repo)

| Piece | Purpose |
|--------|---------|
| `ddpm_runtime.py` | Resolves **runtime profile** (`colab` / `psc` / `local`): Colab forces `num_workers=0`; optional env overrides below. |
| `train.py --runtime auto\|local\|colab\|psc` | **`auto`**: detects Colab via `google.colab`; else local. Env **`DDPM_RUNTIME`** overrides the flag. |
| **`DDPM_DATA_DIR`**, **`DDPM_OUTPUT_DIR`** | Optional: override `data_dir` and `output_dir` (e.g. Ocean or Drive paths). |
| `configs/ddpm_colab.yaml` | Smaller batch / fewer epochs / `num_workers: 0` for notebooks. |
| `configs/ddpm_psc.yaml` | Defaults suited to batch GPU jobs; combine with env paths as needed. |
| `requirements-psc.txt` | Pip deps **only** (use **after** `module load` + `source activate $AI_ENV` — do not reinstall torch). |
| `scripts/train_bridges2_gpu_shared.sbatch` | Example **Slurm** script for **GPU-shared** (edit module, paths, `#SBATCH -A`). |
| `scripts/psc_interactive_gpu.sh` | Prints **interact** + setup commands for copy-paste. |
| `colab/DDPM_CIFAR10.ipynb` | Ready-made Colab notebook (set `REPO_URL`, enable GPU runtime). |

---

## Running on Pittsburgh Supercomputing Center (PSC)

This project is written for **PyTorch on Linux**. On PSC, **Bridges-2** is the primary system where PyTorch is provided through PSC-maintained **AI** modules. Official references:

- [Bridges-2 User Guide](https://www.psc.edu/resources/bridges-2/user-guide/) (partitions, `sbatch`, `interact`, GPU options)
- [PSC environments for AI applications](https://www.psc.edu/resources/software/AI/) (`module spider AI`, `module help …`)
- [PyTorch at PSC](https://www.psc.edu/resources/software/pytorch/) (points to Bridges-2 AI modules)

**Requirements**

- A valid **Bridges-2** account and an allocation that includes **Bridges-2 GPU** if you use GPU partitions (`GPU`, `GPU-shared`). Partition access depends on your allocation; see the User Guide section *Which partitions can I use?*
- Run training on **compute nodes** (batch or interactive job), not on login nodes.

**Software (PyTorch module)**

PSC documents that PyTorch is available via **AI/pytorch** modules on Bridges-2. List versions with:

```bash
module spider AI
```

Load a specific module (example names appear in the `module spider` output, e.g. `AI/pytorch_23.02-1.13.1-py3`; **use the version your site lists**), then activate the environment variable **`$AI_ENV`** as described on the [AI software page](https://www.psc.edu/resources/software/AI/):

```bash
module load AI/pytorch_23.02-1.13.1-py3   # replace with an available module from 'module spider AI'
source activate $AI_ENV
conda list    # verify torch and other packages
```

Install pip dependencies **without** replacing the module’s PyTorch:

```bash
pip install -r requirements-psc.txt
```

**Interactive GPU session**

The Bridges-2 User Guide documents **GPU** partitions (`GPU`, `GPU-shared`), **`interact`**, and GPU requests via **`--gres=gpu:type:n`**, where **`type`** is one of **`v100-16`**, **`v100-32`**, **`l40s-48`**, **`h100-80`** (see the User Guide *GPU partitions* / *interact* sections). Example pattern from the guide for **GPU-shared**:

```bash
interact -p GPU-shared --gres=gpu:v100-32:1 -t 2:00:00
```

Then `module load …`, `source activate $AI_ENV`, `cd` to the repo, `pip install -r requirements-psc.txt`, and run:

```bash
export DDPM_RUNTIME=psc
python train.py --config configs/ddpm_psc.yaml --runtime psc
```

See also `scripts/psc_interactive_gpu.sh` for a printable checklist.

**Batch job (`sbatch`)**

Use the checked-in template **`scripts/train_bridges2_gpu_shared.sbatch`**: edit the **`module load`** line to match `module spider AI`, set **`#SBATCH -A`** if required, and point the working directory at your clone (often under **`/ocean/projects/...`** per the User Guide). Submit from that directory:

```bash
sbatch scripts/train_bridges2_gpu_shared.sbatch
```

The script sets **`DDPM_RUNTIME=psc`** and installs **`requirements-psc.txt`** before `train.py`.

**Help**

PSC lists **help@psc.edu** in the Bridges-2 documentation for support questions.

---

## Running on Google Colab

[Google Colab](https://colab.research.google.com/) is a hosted notebook service that, per the [official Colab FAQ](https://research.google.com/colaboratory/faq.html), provides access to computing resources **including GPUs and TPUs**, with usage limits that **vary over time** and are **not published** in full detail.

**Enable GPU (when available)**

The FAQ states you can select **Runtime → Change runtime type** and set **Hardware accelerator** (e.g. to **GPU** when you need acceleration, or **None** if you are not using the GPU—see the FAQ item on GPU utilization). **Available GPU/TPU types vary over time**; the FAQ explicitly says they are not fixed.

**Notebook in this repo**

Upload or open **`colab/DDPM_CIFAR10.ipynb`** in Colab (e.g. upload the file, or open from GitHub if you host it there). Edit **`REPO_URL`**, enable a GPU runtime as above, then run all cells.

**Command-line equivalent**

Colab’s base image often includes PyTorch; install the rest and train with the Colab config ( **`--runtime colab`** or **`DDPM_RUNTIME=colab`** also forces safe DataLoader settings):

```python
!pip install -q ruamel.yaml tqdm
# clone + %cd into repo ...
!python train.py --config configs/ddpm_colab.yaml --runtime colab
```

With **`--runtime auto`**, Colab is detected automatically and **`num_workers`** is set to **0** even if the YAML says otherwise.

**Persistence and limits**

- The FAQ notes that **free-tier** access to expensive resources like GPUs is **heavily restricted**, and that **runtimes can terminate** (idle timeouts, usage limits). Save checkpoints to **Google Drive** (`google.colab.drive.mount`) or download them before the session ends if you need them later.
- The FAQ also warns that **Drive-mounted I/O can be slow**; prefer keeping active training data on the VM’s local disk when possible.

**Examples from Google**

Colab links to example notebooks such as [TensorFlow With GPU](https://colab.research.google.com/notebooks/gpu.ipynb) for accelerator usage patterns; PyTorch users typically check `torch.cuda.is_available()` after selecting a GPU runtime.

---

## 1. Download the data

Please first download the data from here: https://drive.google.com/drive/u/0/folders/1Hr8LU7HHPEad8ALmMo5cvisazsm6zE8Z

After download please unzip the data with

```
tar -xvf imagenet100_128x128.tar.gz
```

For the **default** `configs/ddpm.yaml` in this repository, **CIFAR-10** is used instead: it is downloaded automatically by `torchvision` under `./data/cifar10/` when `dataset: cifar10` and `cifar_download: true` (see [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)). Use `dataset: imagefolder` if you use the ImageNet-style folder layout above.

## 2.Implementing DDPM from Scratch

This homework will start from implementing DDPM from scratch.

We provide the basic code structure for you and you will be implementing the following modules (by filling all TODOs)):

```
1. pipelines/ddpm.py
2. schedulers/scheduling_ddpm.py
3. train.py
4. configs/ddpm.yaml
```

A very basic U-Net architecture is provided to you, and you will need to improve the architecture for better performance.

## 3. Implementing DDIM

Implement the DDIM from scratch:

```
1. schedulers/scheduling_ddpm.py
2. create a config with ddim by setting use_ddim to True
```

**NOTE: you need to set use_ddim to TRUE**

## 4. Implementing Latent DDPM

Implement the Latent DDPM.

The pre-trained weights of VAE and basic modules are provided. 

Download the pretrained weight here: and put it under a folder named 'pretrained' (create one if it doesn't exist)

You need to implement:

```
1. models/vae.py
2. train.py with vae related stuff
3. pipeline/ddpm.py with vae related stuff
```

**NOTE: you need to set use_vae to TRUE**

## 5. Implementing CFG

Implement CFG

```
1. models/class_embedder.py
2. train.py with cfg related stuff
3. pipeline/ddpm.py with cfg related stuff
```

**NOTE: you need to set use_cfg to TRUE**

## 6. Evaluation

```
inference.py
```

## 7. Kaggle Submission

After generating 5,000 images, create your Kaggle submission CSV:

**From saved images on disk:**
```
python generate_submission.py \
    --image_dir /path/to/generated_images \
    --output submission.csv
```

**From your inference script (Python API):**
```python
from generate_submission import generate_submission_from_tensors

# all_images: tensor (5000, 3, H, W) in [-1, 1] or [0, 1]
all_images = torch.cat(all_images, dim=0)
generate_submission_from_tensors(all_images, output_csv="submission.csv")
```

This extracts Inception-v3 pool3 features, computes mean and covariance, and writes the CSV. Upload the CSV to the Kaggle InClass competition page.
