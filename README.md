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

Install any missing pip dependencies for this repo (e.g. `ruamel.yaml`) after activation:

```bash
pip install -r requirements.txt
```

**Interactive GPU session**

The Bridges-2 User Guide documents **GPU** partitions (`GPU`, `GPU-shared`), **`interact`**, and GPU requests via **`--gres=gpu:type:n`**, where **`type`** is one of **`v100-16`**, **`v100-32`**, **`l40s-48`**, **`h100-80`** (see the User Guide *GPU partitions* / *interact* sections). Example pattern from the guide for **GPU-shared**:

```bash
interact -p GPU-shared --gres=gpu:v100-32:1 -t 2:00:00
```

Then `module load …`, `source activate $AI_ENV`, clone or `cd` to the project, and run `python train.py --config configs/ddpm.yaml`.

**Batch job (`sbatch`)**

The User Guide includes **sample batch scripts** for the **GPU-shared** partition (e.g. `#SBATCH -p GPU-shared`, `#SBATCH --gpus=v100-32:…`, walltime). Adapt that template: set `#SBATCH -A` if you must charge a non-default allocation (documented under `sbatch` options). Example skeleton:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 8:00:00
#SBATCH -J ddpm-cifar

set -euo pipefail
cd /path/to/11685-Diffusion-Project

module load AI/pytorch_23.02-1.13.1-py3   # match 'module spider AI' on Bridges-2
source activate $AI_ENV
pip install -q -r requirements.txt      # if needed

python train.py --config configs/ddpm.yaml
```

Use your PSC **project path** (e.g. under Ocean) for `cd` and outputs; the User Guide’s batch examples use `/ocean/projects/...` paths.

**Help**

PSC lists **help@psc.edu** in the Bridges-2 documentation for support questions.

---

## Running on Google Colab

[Google Colab](https://colab.research.google.com/) is a hosted notebook service that, per the [official Colab FAQ](https://research.google.com/colaboratory/faq.html), provides access to computing resources **including GPUs and TPUs**, with usage limits that **vary over time** and are **not published** in full detail.

**Enable GPU (when available)**

The FAQ states you can select **Runtime → Change runtime type** and set **Hardware accelerator** (e.g. to **GPU** when you need acceleration, or **None** if you are not using the GPU—see the FAQ item on GPU utilization). **Available GPU/TPU types vary over time**; the FAQ explicitly says they are not fixed.

**Suggested notebook workflow**

1. Open a new notebook in Colab.
2. Set the runtime type as above if you want a GPU runtime.
3. Clone the repository and install dependencies (Colab’s base image includes PyTorch in many runtimes; you still need this project’s other packages):

```python
# Replace the URL with your fork or the course repo.
!git clone https://github.com/<org-or-user>/11685-Diffusion-Project.git
%cd 11685-Diffusion-Project
!pip install -q ruamel.yaml tqdm
```

4. Train (CIFAR-10 will download automatically if `cifar_download` is true in config):

```python
!python train.py --config configs/ddpm.yaml --num_workers 0
```

Use `--num_workers 0` in Colab to avoid multiprocessing issues with the DataLoader in some notebook setups.

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
