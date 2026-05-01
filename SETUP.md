# NER Translation — vast.ai Training Setup Guide

Step-by-step instructions for setting up a new machine to rent a GPU on vast.ai and run the SeqDiffuSeq training notebook.

---

## 1. Prerequisites

Install the vastai CLI on your local machine:

```bash
brew install pipx && pipx install vastai
vastai set api-key <YOUR_API_KEY>   # from vast.ai → Account → API Keys
```

Your API key is on the [vast.ai account page](https://vast.ai/console/account/).

---

## 2. Generate an SSH key (first time only)

```bash
ssh-keygen -t ed25519 -C "your@email.com" -f ~/.ssh/vastai_key
```

Add the **public key** to vast.ai:
1. Go to [vast.ai → Account → SSH Keys](https://vast.ai/console/account/)
2. Click **Add SSH Key**
3. Paste the contents of `~/.ssh/vastai_key.pub`
   ```bash
   cat ~/.ssh/vastai_key.pub
   ```

---

## 3. Find and rent a GPU instance

### Search for a cheap RTX 4090 or A100:

```bash
vastai search offers 'reliability>0.95 num_gpus=1 gpu_name=RTX_4090 inet_up>200' --order dph_total
```

Pick an instance ID (e.g. `12345678`) from the output, then rent it:

```bash
vastai create instance 12345678 \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel \
  --disk 50 \
  --ssh \
  --direct
```

### Get the SSH connection details:

```bash
vastai show instance 12345678
```

Look for `ssh_host` and `ssh_port` in the output. It will look like:
```
ssh_host: 123.45.67.89
ssh_port: 12345
```

---

## 4. Configure SSH on your local machine

Add this to `~/.ssh/config` (replace IP and port with your instance's values):

```
Host vastai
  HostName 123.45.67.89
  Port 12345
  User root
  IdentityFile ~/.ssh/vastai_key
  StrictHostKeyChecking no
```

Test the connection:

```bash
ssh vastai "echo connected"
```

---

## 5. Upload all files to the instance

Run these three commands from your local machine. They upload the SeqDiffuSeq repo, the dataset, and the notebook:

```bash
# Upload SeqDiffuSeq source code (~5MB)
rsync -avz --progress \
  "/path/to/NER_translation/SeqDiffuSeq/" \
  vastai:~/SeqDiffuSeq/

# Upload training dataset (~80MB)
rsync -avz --progress \
  "/path/to/NER_translation/train_dataset/" \
  vastai:~/train_dataset/

# Upload the notebook
scp -i ~/.ssh/vastai_key \
  "/path/to/NER_translation/zh_vi_translation.ipynb" \
  vastai:~/
```

> **Replace `/path/to/NER_translation/`** with your actual local path, e.g.:
> `/Users/holly.nguyen/Documents/My Research/NER_translation/`

---

## 6. Install Python dependencies on the instance

SSH into the instance and install the NLP packages (PyTorch is pre-installed in the Docker image):

```bash
ssh vastai "pip install -q \
  bert-score blobfile datasets 'huggingface-hub==0.4.0' \
  mpi4py nltk pandas protobuf rouge-score sacrebleu \
  sacremoses scikit-learn scipy spacy tokenizers \
  torchmetrics tqdm 'transformers==4.18.0' \
  jupyterlab ipykernel"
```

---

## 7. Start JupyterLab with an SSH tunnel

On your local machine, open a tunnel so you can reach Jupyter in your browser:

```bash
ssh -N -L 8888:localhost:8888 vastai &
```

On the instance, start JupyterLab:

```bash
ssh vastai "jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password='' &"
```

Open [http://localhost:8888](http://localhost:8888) in your browser. Open `zh_vi_translation.ipynb`.

---

## 8. Run the notebook

Run cells top to bottom through the phases:

| Phase | What it does |
|-------|-------------|
| Phase 1 | Clones/verifies the SeqDiffuSeq repo (skipped — already uploaded) |
| Phase 2 | Copies data from `~/train_dataset/` into `~/SeqDiffuSeq/data/en_zh/` |
| Phase 3 | Trains a 32k BPE tokenizer on the corpus |
| Phase 4 | Trains the diffusion model for 100k steps (use tmux — see below) |
| Phase 5 | Runs inference and evaluates BLEU |

### Important: run training inside tmux so it survives disconnects

Before running Phase 4, open a tmux session on the instance:

```bash
ssh vastai "tmux new -s train"
```

Inside tmux, run the training cell manually or run the whole notebook headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  ~/zh_vi_translation.ipynb 2>&1 | tee ~/train.log
```

Detach from tmux with `Ctrl+B` then `D`. Reconnect later with:

```bash
ssh vastai "tmux attach -t train"
```

---

## 9. Monitor training

Watch the training log:

```bash
ssh vastai "tail -f ~/train.log"
```

Check progress CSV (created by the training script):

```bash
ssh vastai "tail -5 ~/SeqDiffuSeq/ckpts/en_zh/progress.csv"
```

**Expected milestones:**

| Step | Expected loss | What to check |
|------|--------------|---------------|
| 2,500 | < 8.0 | If still > 8, check data path |
| 25,000 | < 1.0 | Run inference — outputs should NOT be "马斯马斯..." |
| 100,000 | ~0.07 | Run full inference, target BLEU (13a) > 15 |

---

## 10. Download results

After training/inference, download the output CSV and summary:

```bash
scp -r vastai:~/SeqDiffuSeq/ckpts/en_zh/inference_out/ \
  "/path/to/NER_translation/out/en_zh_v2/"
```

---

## 11. Destroy the instance when done

**Important: destroy the instance when you stop working to avoid idle charges.**

```bash
vastai destroy instance 12345678
```

Or from the [vast.ai web console](https://vast.ai/console/instances/).

---

## Key config values (already baked into the notebook)

| Parameter | Value | Why |
|-----------|-------|-----|
| BPE vocab size | 32,000 | Prevents clamping collapse (was 10k) |
| vocab_size (model) | 32,005 | 32k + 5 special tokens |
| init_pretrained | True | Load BART-base weights → better geometry |
| lr_anneal_steps | 100,000 | Full LR schedule (was 30k → premature decay) |
| diffusion_steps (inference) | 200 | DDIM — 5x faster, better quality |
| use_ddim | True | Deterministic denoising, avoids collapse |
| num_samples (eval) | 200 | More reliable BLEU estimate (was 50) |
