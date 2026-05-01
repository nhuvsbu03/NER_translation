# NER Translation — vast.ai Training Setup Guide (Windows)

Step-by-step instructions for Windows users. All commands run in **PowerShell** unless noted.

---

## 0. One-time prerequisites

### Install Git for Windows
Download and install from [git-scm.com](https://git-scm.com/download/win).  
This gives you **Git Bash**, which you'll use for file uploads (scp).

### Install Python (if not already installed)
Download from [python.org](https://www.python.org/downloads/). During install, check **"Add Python to PATH"**.

### Clone the repo
Open PowerShell and run:
```powershell
git clone https://github.com/nhuvsbu03/NER_translation.git
cd NER_translation
```

---

## 1. Install the vastai CLI

Open PowerShell:
```powershell
pip install vastai
vastai set api-key <YOUR_API_KEY>
```

Your API key is on the [vast.ai account page](https://vast.ai/console/account/) under **Account → API Keys**.

---

## 2. Generate an SSH key (first time only)

In PowerShell:
```powershell
ssh-keygen -t ed25519 -C "your@email.com" -f "$env:USERPROFILE\.ssh\vastai_key"
```

Press Enter twice to skip the passphrase.

Print the public key:
```powershell
Get-Content "$env:USERPROFILE\.ssh\vastai_key.pub"
```

Add the public key to vast.ai:
1. Go to [vast.ai → Account → SSH Keys](https://vast.ai/console/account/)
2. Click **Add SSH Key**
3. Paste the output from the command above

---

## 3. Find and rent a GPU instance

Search for available GPUs in PowerShell:
```powershell
vastai search offers 'reliability>0.90 num_gpus=1 gpu_ram>=24 inet_up>100 disk_space>30' --order dph_total
```

Pick an instance ID from the output (e.g. `12345678`), then rent it:
```powershell
vastai create instance 12345678 `
  --image vastai/pytorch `
  --disk 50 `
  --ssh `
  --direct
```

> **Note:** PowerShell uses a backtick `` ` `` for line continuation, not `\`.

Get the SSH connection details:
```powershell
vastai show instance 12345678
```

Look for `ssh_host` and `ssh_port` in the output.

---

## 4. Configure SSH

Create or edit the SSH config file. In PowerShell:
```powershell
notepad "$env:USERPROFILE\.ssh\config"
```

Add this block (replace IP and port with your instance's values):
```
Host vastai
  HostName 123.45.67.89
  Port 12345
  User root
  IdentityFile C:\Users\YourUsername\.ssh\vastai_key
  StrictHostKeyChecking no
```

> Replace `YourUsername` with your actual Windows username (run `echo $env:USERNAME` if unsure).

Test the connection:
```powershell
ssh vastai "echo connected"
```

---

## 5. Upload all files to the instance

Open **Git Bash** (not PowerShell — Git Bash handles Unix-style paths better for scp).

Replace `C:/path/to/NER_translation` with your actual repo path (e.g. `C:/Users/YourUsername/NER_translation`):

```bash
# Upload SeqDiffuSeq source code (~5MB)
scp -i ~/.ssh/vastai_key -P <PORT> -r \
  "C:/path/to/NER_translation/SeqDiffuSeq" \
  root@<IP>:~/SeqDiffuSeq

# Upload training dataset (~80MB)
scp -i ~/.ssh/vastai_key -P <PORT> -r \
  "C:/path/to/NER_translation/train_dataset" \
  root@<IP>:~/train_dataset

# Upload the notebook
scp -i ~/.ssh/vastai_key -P <PORT> \
  "C:/path/to/NER_translation/zh_vi_translation.ipynb" \
  root@<IP>:~/
```

> Replace `<PORT>` and `<IP>` with the values from step 3.  
> If you set up the `~/.ssh/config` in step 4, you can simplify to `vastai:~/` instead of `root@<IP>:~/`.

---

## 6. Install Python dependencies

The notebook's Phase 1 cell handles this automatically when you run it. It installs all required NLP packages (~2 minutes).

---

## 7. Start JupyterLab

**Open two PowerShell windows.**

**Window 1** — SSH tunnel (keep this open the whole time):
```powershell
ssh -N -L 8888:localhost:8888 vastai
```

**Window 2** — Start JupyterLab on the instance:
```powershell
ssh vastai "jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''"
```

Open [http://localhost:8888](http://localhost:8888) in your browser. Open `zh_vi_translation.ipynb`.

---

## 8. Run the notebook

Run cells top to bottom through the phases:

| Phase | What it does |
|-------|-------------|
| Phase 2 | Copies data from `~/train_dataset/` into `~/SeqDiffuSeq/data/en_zh/` |
| Phase 3 | Trains a 32k BPE tokenizer on the corpus |
| Phase 4 | Trains the diffusion model for 100k steps |
| Phase 5 | Runs inference on the full test set |
| Phase 6 | Computes BLEU score |

### Important: run training inside tmux so it survives disconnects

In PowerShell, before running Phase 4:
```powershell
# Open a tmux session on the instance
ssh vastai "tmux new-session -d -s train"

# Attach to it
ssh vastai "tmux attach -t train"
```

Inside tmux, run the notebook headlessly:
```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  ~/zh_vi_translation.ipynb 2>&1 | tee ~/train.log
```

Detach from tmux: press `Ctrl+B` then `D`. The training keeps running after you close the window.

Reconnect later:
```powershell
ssh vastai "tmux attach -t train"
```

---

## 9. Monitor training

Watch the log in PowerShell:
```powershell
ssh vastai "tail -f ~/train.log"
```

Check the progress CSV:
```powershell
ssh vastai "tail -5 ~/SeqDiffuSeq/ckpts/en_zh/progress.csv"
```

**Expected milestones:**

| Step | Expected loss | What to check |
|------|--------------|---------------|
| 2,500 | < 8.0 | If still > 8, data path or tokenizer is broken |
| 25,000 | < 1.0 | Run inference — outputs should NOT repeat a single token |
| 100,000 | ~0.07 | Run full inference, target BLEU (13a) > 15 |

---

## 10. Download results

In **Git Bash**:
```bash
scp -i ~/.ssh/vastai_key -P <PORT> -r \
  root@<IP>:~/SeqDiffuSeq/ckpts/en_zh/inference_out/ \
  "C:/path/to/NER_translation/out/en_zh_v2/"
```

---

## 11. Destroy the instance when done

**Important: always destroy the instance when you stop working to avoid idle charges.**

```powershell
vastai destroy instance 12345678
```

Or from the [vast.ai web console](https://vast.ai/console/instances/).

---

## Troubleshooting

**`ssh-keygen` not found**  
OpenSSH ships with Windows 10/11. If it's missing: Settings → Apps → Optional Features → Add "OpenSSH Client".

**`scp` upload is slow**  
Normal for large files over a home connection. The dataset is ~80MB — expect 2–5 minutes on a typical upload speed.

**Permission denied (publickey)**  
Make sure you're using `-i ~/.ssh/vastai_key` and the public key (`vastai_key.pub`) was added to the vast.ai web UI, not the private key.

**Port in SSH config vs scp command**  
SSH config uses `Port 12345` (no flag needed for `ssh vastai`).  
`scp` requires `-P 12345` (capital P) when not using the SSH config alias.

**JupyterLab shows "connection refused"**  
The tunnel in Window 1 must be running. If you closed it, rerun:
```powershell
ssh -N -L 8888:localhost:8888 vastai
```
