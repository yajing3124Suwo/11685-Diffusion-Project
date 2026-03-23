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

This project targets **PyTorch on Linux**. On PSC, **Bridges-2** is the main system where PyTorch is provided through PSC **AI** modules. Keep these official pages open while you work:

- [Bridges-2 User Guide](https://www.psc.edu/resources/bridges-2/user-guide/) — **Connecting**, **Running jobs**, **GPU partitions**, `sbatch` / `interact`
- [Getting started with HPC (Bridges-2)](https://www.psc.edu/resources/bridges-2/getting-started-with-hpc/) — login vs compute nodes, SSH vs OnDemand
- [PSC AI software environments](https://www.psc.edu/resources/software/AI/) — `module spider AI`, `$AI_ENV`
- [PyTorch at PSC](https://www.psc.edu/resources/software/pytorch/)

**Important:** Production jobs must run on **compute nodes** (interactive allocation or batch), **not** on login nodes. See the User Guide and [Getting started](https://www.psc.edu/resources/bridges-2/getting-started-with-hpc/).

### Step 1 — Account and allocation

- You need a **PSC / ACCESS (or course)** account and a **Bridges-2** allocation that includes **Bridges-2 GPU** if you plan to use GPU partitions (`GPU`, `GPU-shared`). Which **partitions** you may use depends on that allocation (User Guide: *Which partitions can I use?*).
- Support: **help@psc.edu** (as documented by PSC).

### Step 2 — Log in to Bridges-2 (login node)

PSC documents two ways to reach Bridges-2: **SSH** and **OnDemand** (browser). Full detail is in the User Guide section [**Connecting to Bridges-2**](https://www.psc.edu/resources/bridges-2/user-guide/#connecting).

**SSH (typical workflow from your laptop)**

1. Install or use an **SSH client** (macOS and Linux usually have `ssh` in Terminal; Windows: PuTTY, Windows Terminal, etc.). PSC discusses SSH and keys in [About using SSH](https://www.psc.edu/about-using-ssh/).
2. Per the Bridges-2 User Guide, connect to hostname **`bridges2.psc.edu`**, port **22**, with your **PSC username**:

   ```bash
   ssh YOUR_PSC_USERNAME@bridges2.psc.edu
   ```

3. Complete any extra authentication your institution requires (e.g. Duo), if prompted.

After login, your prompt is on a **login node** (hostname often looks like `bridges2-login…`). Use this node only for **editing files, `git`, submitting jobs** — not for training.

**OnDemand**

If you prefer a web UI, use **OnDemand** as described in the [OnDemand section](https://www.psc.edu/resources/bridges-2/user-guide/#ondemand) of the User Guide. You still use Slurm (`sbatch`, `interact`, etc.) according to the same cluster rules.

### Step 3 — Where to put the project (home vs Ocean)

- **Home directory** (`~`): fine for the code repo and small files; quotas are limited.
- **Ocean** (`/ocean/projects/…`): shared project space for allocations; the User Guide explains Ocean paths for persistent, larger storage. For long runs, many users keep **checkpoints and datasets** under `/ocean/projects/<allocation>/<username>/…` and set **`DDPM_OUTPUT_DIR`** / **`DDPM_DATA_DIR`** (see `ddpm_runtime.py`).

**Large file transfers** to/from Bridges-2 should use PSC’s **Data Transfer Nodes** (**`data.bridges2.psc.edu`**), as described in the User Guide — not sustained heavy transfer through the login node.

### Step 4 — Clone this repository from GitHub (on the login node)

`git` is commonly available on Bridges-2. From your home or Ocean project directory:

```bash
cd ~
# or: cd /ocean/projects/YOUR_ALLOC/YOUR_USER

git clone https://github.com/YOUR_GITHUB_USER/11685-Diffusion-Project.git
cd 11685-Diffusion-Project
```

- **Private repository:** use a [GitHub personal access token](https://docs.github.com/en/authentication) with HTTPS, or configure **SSH keys** on Bridges-2 and clone with `git@github.com:USER/REPO.git`. Do not commit secrets into the repo.
- **Updates:** run `git pull` inside the clone after you push changes from your laptop.

### Step 5 — Load PyTorch (PSC AI module) and pip dependencies

PSC provides PyTorch in **AI/pytorch** modules. List available names:

```bash
module spider AI
```

Load one module (replace with a name **you** see from `module spider`), then activate **`$AI_ENV`** as in the [AI software page](https://www.psc.edu/resources/software/AI/):

```bash
module load AI/pytorch_23.02-1.13.1-py3
source activate $AI_ENV
conda list | head   # optional: confirm torch is present
```

Install **only** extra pip packages (do **not** `pip install torch` in a way that replaces the module stack):

```bash
cd ~/11685-Diffusion-Project   # or your path
pip install -r requirements-psc.txt
```

### Step 6 — Request a GPU and train

**Option A — Interactive GPU (good for debugging)**

From the **login** node, request an interactive session on a **GPU** partition. The User Guide documents **`interact`**, **`--gres=gpu:type:n`**, and GPU types **`v100-16`**, **`v100-32`**, **`l40s-48`**, **`h100-80`**. Example for **GPU-shared** with one **v100-32** GPU:

```bash
interact -p GPU-shared --gres=gpu:v100-32:1 -t 2:00:00
```

Wait until the shell is on a **compute** node. Then:

```bash
module load AI/pytorch_23.02-1.13.1-py3
source activate $AI_ENV
cd ~/11685-Diffusion-Project
pip install -r requirements-psc.txt   # if not already done in this env
export DDPM_RUNTIME=psc
# Optional: large outputs on Ocean
# export DDPM_OUTPUT_DIR=/ocean/projects/ALLOC/USER/ddpm_runs
# export DDPM_DATA_DIR=/ocean/projects/ALLOC/USER/ddpm_data
python train.py --config configs/ddpm_psc.yaml --runtime psc
```

**Option B — Batch job (good for long training)**

1. Edit **`scripts/train_bridges2_gpu_shared.sbatch`**: set **`module load`** to your module, uncomment and set **`#SBATCH -A`** if your allocation requires it.
2. From the **login** node, `cd` to the **repository root** (so `SLURM_SUBMIT_DIR` is the project), then:

   ```bash
   sbatch scripts/train_bridges2_gpu_shared.sbatch
   ```

3. Monitor with **`squeue -u YOUR_USERNAME`** and read Slurm output files in the submit directory (User Guide: **`sbatch`**, **`squeue`**).

**Checklist file:** `scripts/psc_interactive_gpu.sh` prints the same interact/setup steps for copy-paste.

---

## Running on Google Colab

[Google Colab](https://colab.research.google.com/) is a **hosted Jupyter** service: you run **notebook cells**; lines starting with **`!`** run shell commands on the VM. The [Colab FAQ](https://research.google.com/colaboratory/faq.html) states that Colab offers resources **including GPUs and TPUs**, that **limits and hardware types change over time**, and that **free GPU access is heavily restricted**.

### Do you need Google Drive?

| Goal | Need Google Drive? |
|------|---------------------|
| Put **source code** on the Colab VM | **No**, if you **`git clone`** or **upload a zip** (see below). |
| **CIFAR-10** (default config) in one session | **No**: `torchvision` downloads into **`./data`** on the VM. |
| **Keep checkpoints** after disconnect | **Strongly recommended:** VMs are **temporary**; the FAQ describes disconnects and limits. Use **Drive**, **browser download**, or another external store. |
| Multi-day training | Expect **multiple sessions**; persist **`experiments/`** (and optionally **`data/`**) to Drive or download between runs. |

The FAQ notes **Drive-mounted I/O can be slow**. For training, keep **`data/`** and hot checkpoints on the VM disk under **`/content/...`**, and **copy finished checkpoints to Drive** (or download) when an epoch completes.

### How to get this project into Colab (pick one)

#### Method 1 — `git clone` in a notebook cell (no Drive needed for code)

Works for a **public** repo. For a **private** repo, use a GitHub **token** in the HTTPS URL — **do not** share notebooks containing a raw token; a **public fork** is safer for coursework.

1. **Runtime → Change runtime type → Hardware accelerator → GPU** (recommended for DDPM).
2. In a cell:

```python
REPO_URL = "https://github.com/YOUR_USER/11685-Diffusion-Project.git"
!git clone {REPO_URL}
%cd 11685-Diffusion-Project
```

#### Method 2 — Open `colab/DDPM_CIFAR10.ipynb`

1. Ensure the repo is on **GitHub** (or download **`colab/DDPM_CIFAR10.ipynb`** only from the GitHub web UI).
2. **Upload** that notebook: Colab **File → Upload notebook**, **or** open directly from GitHub (URL pattern):

   `https://colab.research.google.com/github/YOUR_USER/11685-Diffusion-Project/blob/main/colab/DDPM_CIFAR10.ipynb`

3. Set **`REPO_URL`** in the first code cell, enable **GPU** runtime, **Run all**.

#### Method 3 — Upload a zip of the project

1. On your computer, zip the project folder (omit `.venv`, huge `data/` if you will re-download CIFAR).
2. In Colab, open the **Files** pane → **Upload** the zip.
3. In a cell:

```python
!unzip -q -o 11685-Diffusion-Project.zip -d /content/
%cd /content/11685-Diffusion-Project
```

(Change the zip filename to match your upload.)

#### Method 4 — Copy from Google Drive (optional)

Use if your copy of the repo lives on Drive (e.g. synced with **Google Drive for desktop**).

```python
from google.colab import drive
drive.mount("/content/drive")
```

Copy to **local** VM disk before training (reduces slow reads from Drive during imports/training):

```python
!cp -r "/content/drive/MyDrive/path/to/11685-Diffusion-Project" /content/ddpm
%cd /content/ddpm
```

### Install dependencies and run training

Colab often ships with **PyTorch**; install the rest, then train:

```python
!pip install -q ruamel.yaml tqdm
%cd /content/11685-Diffusion-Project   # adjust path if needed
import os
os.environ["DDPM_RUNTIME"] = "colab"
!python train.py --config configs/ddpm_colab.yaml --runtime colab
```

With **`--runtime auto`**, Colab is detected and **`num_workers`** is forced to **0** for stable DataLoader behavior.

### Saving checkpoints after training

- Use the **Files** sidebar to **download** `experiments/`, or  
- **Copy to Drive** (better for a few large files than training directly off Drive):

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p "/content/drive/MyDrive/ddpm_backup"
!cp -r experiments "/content/drive/MyDrive/ddpm_backup/"
```

### References

- [Colab FAQ](https://research.google.com/colaboratory/faq.html) — GPUs, limits, Drive, disconnects  
- FAQ-linked example: [TensorFlow with GPU](https://colab.research.google.com/notebooks/gpu.ipynb); for PyTorch use `torch.cuda.is_available()` after selecting a GPU runtime.

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
