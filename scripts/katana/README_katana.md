# Running on UNSW Katana (HPC)

This directory contains everything you need to reproduce the entire experiment end-to-end on UNSW Katana with one `qsub` command per stage — or one shell command for the whole pipeline.

Katana uses **OpenPBS** (`qsub`), not SLURM. All job scripts here are `.pbs` files.

---

## TL;DR — One-time setup, then one command

```bash
# (1) Copy code + data to Katana
scp -r battery-de-ddpm zXXXXXXX@katana.unsw.edu.au:~/

# (2) SSH in
ssh zXXXXXXX@katana.unsw.edu.au
cd ~/battery-de-ddpm

# (3) One-time env setup (installs Miniconda if needed, creates `battery-soh` env)
bash scripts/katana/setup_env.sh

# (4) Submit the whole pipeline (1 prepare + 5 trains + 1 evaluate, with dependencies)
bash scripts/katana/submit_all.sh

# (5) Monitor
watch -n 60 'qstat -u $USER'

# (6) When done, compare with your original Block 3 numbers
python scripts/katana/compare_with_paper.py \
    --new outputs/eval_downstream/metrics.csv \
    --ref outputs/reference_metrics.csv      # ← upload this from your local NewJournal_Final/
```

That's the whole story. Below are the details.

---

## What's in this folder

| File | Role |
|---|---|
| `setup_env.sh` | Installs Miniconda (if absent) and creates conda env `battery-soh` with the right PyTorch+CUDA build. Run once. |
| `activate_env.sh` | Sourced at the top of every PBS job to activate the env. |
| `00_prepare.pbs` | Block 1: prepare_data.py. CPU only, ~1 min. |
| `01_train_proposed.pbs` | Block 2: Proposed (Auto-DE + UpBlock2D, 500 epochs). |
| `01_train_resampled_de.pbs` | Block 2: Resampled-DE (DE-DDPM, 400 epochs). |
| `01_train_1d_ddpm.pbs` | Block 2: 1D conditional DDPM (500 epochs). |
| `01_train_vanilla_gan.pbs` | Block 2: GAN baseline (400 epochs). |
| `01_train_vanilla_vae.pbs` | Block 2: VAE baseline (400 epochs). |
| `02_evaluate.pbs` | Block 3: downstream SOH prediction, 3×9×8×5 = 1080 runs. |
| `submit_all.sh` | Submits all 7 jobs with `-W depend=afterok:...` so they chain correctly. |
| `compare_with_paper.py` | Diffs the new `metrics.csv` against any reference CSV; flags real divergences and tracks method-ranking preservation. |

---

## Job graph

```
00_prepare ─┬─► 01_train_proposed     ─┐
            ├─► 01_train_resampled_de  │
            ├─► 01_train_1d_ddpm       ├─► 02_evaluate
            ├─► 01_train_vanilla_gan   │
            └─► 01_train_vanilla_vae  ─┘
```

The 5 training jobs run **in parallel** after prepare succeeds. Evaluate waits until **all 5 finish successfully** (`afterok:`).

---

## Resource requests (per job)

| Job | Resources | Walltime |
|---|---|---|
| 00_prepare | `select=1:ncpus=2:mem=8gb` (CPU queue) | 0:30 |
| 01_train_proposed | `select=1:ncpus=8:ngpus=1:mem=46gb` | 6:00 |
| 01_train_resampled_de | same | 5:00 |
| 01_train_1d_ddpm | same | 2:00 |
| 01_train_vanilla_gan | same | 1:00 |
| 01_train_vanilla_vae | same | 1:00 |
| 02_evaluate | same | 6:00 |

These follow Katana's recommended pattern: `ncpus = ngpus × 8` and `mem = ngpus × 46 GB`. All walltimes are well under the 12-hour cutoff that gets faster queue start times.

Expected wall-clock (5 trains in parallel, evaluate serial after): **~7 hours** on V100, **~5 hours** on A100. No control over which GPU you get; both work.

---

## Where outputs land

```
outputs/
├── prepared_data.pkl
├── ckpt/
│   ├── proposed/final.pt
│   ├── resampled_de/final.pt
│   ├── 1d_ddpm/final.pt
│   ├── vanilla_gan/vanilla_gan.pt
│   └── vanilla_vae/vanilla_vae.pt
├── curves/                      # generated synthetic curves
│   ├── Proposed_curves.pkl
│   ├── DE-DDPM_curves.pkl
│   └── ...
└── eval_downstream/
    ├── metrics.csv              # ← the deliverable
    └── results.pkl
```

PBS stdout / stderr go to `logs/` (each job writes its own `.log`).

---

## Monitoring

```bash
qstat -u $USER                          # all your jobs
qstat -f <JOBID>                        # detailed status of one job
tail -f logs/01_train_proposed.log      # live training output
ls -la logs/                            # all logs
```

If a job fails (state = `F` and exit status ≠ 0), check the matching `.log` first. Common failures:

- `outputs/prepared_data.pkl` not found — prepare didn't finish, or wasn't in the dependency chain
- `torch.cuda.is_available() == False` inside a GPU job — you accidentally landed on a CPU node; check `qstat -f` for the actual node assignment
- OOM — increase `mem` in the `.pbs` file, e.g. `ngpus=2` gives you 92 GB (and 2 GPUs)

---

## How to validate the results

When `02_evaluate` finishes, `outputs/eval_downstream/metrics.csv` is the final deliverable. To verify it matches your paper's numbers:

1. **Upload your original Block 3 CSV** (e.g. from `NewJournal_Final/final_downstream_soh_results.csv`) to Katana as `outputs/reference_metrics.csv`.
2. Run the comparator:

```bash
python scripts/katana/compare_with_paper.py
```

The script:

- Compares each (Task, Method, Model) row's `MAE_mean` against the reference
- Flags rows with relative diff > 2% (⚠) and > 5% (✗)
- Re-derives method rankings per (Task, Model) and reports any swaps
- Prints a final verdict

**Expected outcome**: > 95% rows show < 2% relative diff, no ranking swaps. CUDA non-determinism + retraining noise typically lives in the 0.5–2% range for this kind of regression-style task.

---

## Re-running just one stage

You don't have to re-submit everything if something fails partway. Each `.pbs` script is standalone:

```bash
# Just re-run one training (assumes prepare succeeded earlier):
qsub scripts/katana/01_train_proposed.pbs

# Just re-run evaluation (assumes all 5 ckpts exist):
qsub scripts/katana/02_evaluate.pbs
```

---

## Cleaning up

```bash
# nuke all caches and ckpts (start fresh)
rm -rf outputs/ logs/

# keep ckpts but redo evaluation
rm -rf outputs/eval_downstream outputs/curves
qsub scripts/katana/02_evaluate.pbs
```

---

## Reference

- Katana docs: https://docs.restech.unsw.edu.au/
- PBS jobs:  https://docs.restech.unsw.edu.au/using_katana/running_jobs/
- Python:    https://docs.restech.unsw.edu.au/software/python/
