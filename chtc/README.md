# Training OpenPI on CHTC

Train pi0/pi0.5 models on [CHTC GPU Lab](https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs) servers, then pull checkpoints back to your local lab GPU for inference.

## Prerequisites

- CHTC account ([request one](https://chtc.cs.wisc.edu/uw-research-computing/form.html))
- `/staging` directory ([request quota](https://chtc.cs.wisc.edu/uw-research-computing/quota-request))
- Docker Hub account (for pushing the training image)
- Local lab GPU with >= 8 GB VRAM (for inference only)

## Workflow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  LOCAL MACHINE   │     │      CHTC        │     │  LOCAL MACHINE  │
│                  │     │                  │     │                 │
│ 1. Convert data  │────>│ 3. Train on GPU  │────>│ 5. Serve policy │
│ 2. Stage to CHTC │     │    (A100/H100)   │     │    (local GPU)  │
│                  │     │ 4. Save ckpts    │     │ 6. Run robot    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Step-by-step

### 1. Convert your dataset locally

```bash
uv run examples/collab/convert_collab_data_to_lerobot.py \
    --db_dir /path/to/episodes \
    --repo_name local/collab \
    --fps 10 --downsample_factor 2
```

### 2. Build and push the Docker image

```bash
docker build -t openpi_train -f chtc/train.Dockerfile .
docker tag openpi_train <dockerhub_user>/openpi_train:latest
docker push <dockerhub_user>/openpi_train:latest
```

### 3. Stage data to CHTC

```bash
./chtc/stage_data.sh <netid>
```

### 4. Submit the training job

Edit `chtc/train.sub`:
- Replace `<dockerhub_user>` with your Docker Hub username
- Set your `netid`

#### Secure W&B setup (matches your `aiml_ws` style)

OpenPI now supports `wandb_entity`, `wandb_group`, and `wandb_tags` via `TrainConfig` (and the provided `pi*_collab` configs already set these to match your `aiml_ws` defaults).

**Do not** hardcode `WANDB_API_KEY` into `train.sub` (submit files are plain text). Instead, set it in your submit shell environment:

```bash
# On the CHTC submit node (e.g. submit1), in your shell:
export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Because `train.sub` sets `getenv = True`, HTCondor will pass `WANDB_API_KEY` into the job environment.

If you want this to persist across logins, add it to a private file like `~/.bashrc` (chmod 600) on the submit node.

Then from the CHTC submit node:

```bash
ssh <netid>@submit1.chtc.wisc.edu
condor_submit chtc/train.sub config_name=pi05_collab exp_name=my_run
```

Monitor with:
```bash
condor_q          # job status
condor_tail <id>  # stream stdout
```

### 5. Pull checkpoints and run inference locally

```bash
./chtc/pull_checkpoints.sh <netid> <cluster_id>

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_collab \
    --policy.dir=checkpoints/pi05_collab/my_run/30000
```

## GPU Lab Limits

| Job Type | Max Runtime | Per-User GPU Limit |
|----------|-------------|-------------------|
| Short    | 12 hours    | 2/3 of GPU Lab    |
| Medium   | 24 hours    | 1/3 of GPU Lab    |
| Long     | 7 days      | 4 GPUs            |

The submit file defaults to `long` (7 days max). For LoRA fine-tuning that finishes faster, switch to `short` for more scheduling priority.

## Available GPUs

Key options on CHTC GPU Lab (as of 2025):

| GPU | VRAM | Capability | Count |
|-----|------|------------|-------|
| A100-40GB | 40 GB | 8.0 | 8 |
| A100-80GB | 80 GB | 8.0 | 32 |
| L40/L40S | 45 GB | 8.9 | 46 |
| H100 | 80 GB | 9.0 | 8 |
| H200 | 141 GB | 9.0 | 16 |

The submit file requests `gpus_minimum_capability = 8.0` and `gpus_minimum_memory = 40960` (40 GB), so jobs will land on A100-40GB or better.

## Adjusting for LoRA vs Full Fine-Tuning

**LoRA** (fits in ~24 GB VRAM):
```
# In train.sub, relax GPU requirements:
gpus_minimum_memory = 24576
+GPUJobLength = "short"
```

**Full fine-tuning** (needs ~70+ GB VRAM):
```
# In train.sub (default settings work):
gpus_minimum_memory = 40960
+GPUJobLength = "long"
```
