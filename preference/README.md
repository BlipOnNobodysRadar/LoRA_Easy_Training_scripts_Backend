# SDXL preference collection and Diffusion-DPO

Experimental, local, single-GPU preference training. Ordinary supervised training
keeps its existing configuration and code path. This subsystem trains a **separate
UNet LoRA**, leaving the input checkpoint and any starting adapters untouched.

## Two starting points

* **Checkpoint only:** leave `model.base_loras` empty. The frozen reference is
  the checkpoint; the output is a new preference LoRA for that checkpoint.
* **Checkpoint plus existing LoRA:** provide `base_loras` as an ordered list of
  `{ "path": "...safetensors", "weight": 1.0 }`. Those adapters remain frozen
  and active in both policy and reference predictions. The output is a new
  preference LoRA to use **alongside the same checkpoint and original adapters
  at the recorded weights**.

This is an additive adapter, not a merged replacement for the original LoRA.
Only the preference adapter is disabled for reference inference; a second SDXL
copy is not resident on the GPU. Text encoders and VAE are frozen. The UNet uses
bf16, native SDPA and gradient checkpointing; trainable adapter weights and AdamW
states use fp32. The VAE retains checkpoint precision and runs in fp32. Text
encoders and the VAE move to the GPU only when needed; embeddings and sampled
latents are cached in CPU memory for the run.

Training defaults to `deterministic: true`, selecting deterministic CUDA
operations and disabling cuDNN attention backward. This supports exact resume
checks on the same hardware/software stack. Set it to false only if accepting
run-to-run numerical variation; it is part of the resume signature. GPU kernels
and library versions can affect reproducibility across machines or upgrades;
see [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Quick start from the frontend checkout

1. Copy `preference.example.json` to `preference.local.json` and set the checkpoint,
   optional original adapter, prompts, dataset and output directories. Paths in
   a configuration are relative to that JSON file. Local JSON files are ignored
   by Git. Use `.safetensors` inputs.
2. Run `run_preferences.bat` on Windows, or select **Utils → Preferences / DPO**
   in the main trainer. The standalone UI needs the frontend environment; GPU
   jobs use `backend/sd_scripts/venv`.
3. In Configuration, select **Generate pairs**. Each pair shares prompt, negative
   prompt, model and sampling settings; its two images use different seeds. A/B
   assignment is reproducible. Later batches advance the seed by twice the
   existing comparison count, so clicking Generate again collects fresh pairs.
4. In Rate, choose A/B slightly or strongly preferred, a tie, or no preference.
   Quality A and B are independent: excellent, good, acceptable, bad, especially
   bad, or unrated. Add optional reasons. **Save** or **Save & Next** commits a
   revision; simply navigating does not save. **Skip** is explicit and durable.
   **Undo last feedback** restores the preceding rating without erasing history.
   Double-click an image to inspect it at full resolution.
5. Save actual A/B preferences for enough varied prompts, then select **Train**.
   A job with no eligible training pairs fails before loading SDXL. Ties, skips
   and quality-only ratings remain available but supply no fabricated winner.
   Two bad images can still have a relative preference if the user chooses one;
   otherwise use No preference or Tie and record their absolute quality.

The form edits the first original adapter; additional adapters in a JSON stack
are preserved when that first adapter remains present. Clearing the original
adapter field intentionally selects checkpoint-only mode and removes the stack.
Advanced options, including holdout fraction, token length and strength weights,
can be set in JSON and survive a GUI save. The default original/preference
adapter weight is 1. Training a loaded preference adapter requires weight 1;
other preference weights are for inference only.

Equivalent CLI commands (PowerShell, from the frontend checkout):

```powershell
$python = '.\backend\sd_scripts\venv\Scripts\python.exe'
& $python -X utf8 -u -m backend.preference.cli generate --config preference.local.json
& $python -X utf8 -u -m backend.preference.cli train --config preference.local.json
& $python -m backend.preference.cli status --dataset 'D:\preferences'
& $python -m backend.preference.cli export --dataset 'D:\preferences' --output 'D:\backup\ratings.jsonl'
```

The CLI accepts `--stop-file PATH` for generation and training. Creating that
file requests an orderly stop. Generation checks between sampling steps;
training finishes the current optimizer step and saves its state. Loading and
encoding can take time before the next cancellation check. The UI provides
Request stop and waits for the worker when closing. Supported launchers in this
checkout share a process lock with ordinary training. Separately launched CLI
trainers, other checkouts and other GPU applications are outside that lock.

## The objective

For a winning image `w` and rejected image `l`, encode both to VAE latents and
sample **the same diffusion timestep and Gaussian noise** for both. Compute
mean squared denoising errors per image under the current policy and frozen
reference, using the positive prompt's SDXL conditioning:

```
margin = (policy_mse_w - policy_mse_l) - (reference_mse_w - reference_mse_l)
logit  = -0.5 * beta * margin
loss   = -strength_weight * log_sigmoid(logit)
```

This follows the beta convention in the
[Diffusion-DPO paper](https://arxiv.org/abs/2311.12908), its
[reference implementation](https://github.com/SalesforceAIResearch/DiffusionDPO/blob/main/train.py)
and the [Diffusers SDXL example](https://github.com/huggingface/diffusers/blob/main/examples/research_projects/diffusion_dpo/train_diffusion_dpo_sdxl.py).
It optimizes the paired relative error against a frozen reference. It does not
train only the winners. Inference negative prompts and CFG are recorded for
reproducible collection; training uses ordinary conditional denoising without CFG.

Default strength weights are slight 0.5, normal 1, strong 2. Losses are averaged
over pairs, not divided by the sum of weights, so strengths still matter with
one pair per microbatch. Each optimizer step averages the configured number of
pair microbatches. Absolute quality is stored for future objectives and analysis;
it does not silently modify this DPO loss. No SNR clipping, EDM2 weighting,
multiresolution noise or supervised-loss mixing is added to this objective.

Defaults (rank/alpha 16, AdamW LR 1e-5, beta 5000, accumulation 4) are starting
parameters, not validated quality recommendations. The original supervised
LoCon's rank, optimizer, batch size and learning rate are not copied to DPO.

## Replay, validation and resume

Comparison PNGs are immutable and content-hashed. SQLite uses transactions, WAL
and append-only feedback revisions. Metadata includes prompts, negative prompts,
seeds, sampling settings, resolution, scheduler configuration, model/adapter
SHA256 hashes and weights, software/precision details, timestamps and session.
Relative image paths make a dataset directory portable. Back up the entire
dataset directory (images and database), preferably while jobs and rating edits
are stopped. JSONL export contains current ratings and rich comparison metadata;
the database retains undo history.

All eligible ratings in the dataset are replayed with uniform pair sampling
across sessions. A normalized exact prompt determines its group; a seeded hash
assigns groups to train or validation. Existing group assignments are preserved
across generation rounds. Different wording of the same concept is not detected
as a near duplicate: use genuinely distinct validation prompts. Training rejects
image tampering, incompatible references, malformed group identities and group
overlap. Synthetic fixtures are rejected by default and cannot mix with human
data. `allow_synthetic` is strictly an execution-testing option.

Each new run has a unique directory with config, frozen dataset snapshot,
metrics JSONL, status, and numbered checkpoints containing:

* `preference_lora.safetensors`: UNet adapter plus reference metadata.
* `training_state.pt`: optimizer and RNG state, loaded with `weights_only=True`.
* `state.json`: step, signature and metrics; `latest.json` in the parent points
  to the latest complete checkpoint.

Set the resume checkpoint directory in the UI, or pass
`train --config ... --resume ...\checkpoint-000025`. Resume requires the same
rated dataset and training settings. You may increase `max_steps` or change the
checkpoint interval. Only the latest checkpoint may continue in the same run.
To add fresh human ratings between rounds, start a **new run** with the previous
preference adapter as `model.preference_lora`, clear Resume, and keep the original
checkpoint/adapters fixed. The reference remains fixed across rounds.

Validation reports fixed-noise held-out DPO loss and preference accuracy. A
falling training loss or a synthetic smoke-test pass does not demonstrate better
images. Judge improvement with fresh human comparisons on unseen prompts.

## Scope and implementation boundaries

Supported: standard epsilon-prediction, four-channel SDXL base checkpoints;
native sd-scripts LoRA and LyCORIS LoRA/LoCon, including text encoder weights
and normalization deltas in starting adapters. Reconstruction accounts for every
saved tensor and fails on unsupported adapter algorithms instead of dropping
weights. A narrow compatibility loader handles the installed LyCORIS factory's
optional PiSSA-argument mismatch; actual PiSSA tensors remain unsupported.

Not implemented: full-weight DPO, non-SDXL families, inpainting, v-prediction,
distilled/turbo objectives, distributed training, arbitrary LyCORIS algorithms,
ranking groups larger than two, online reward models or a public labeling server.
More-than-two-image records can be retained by storage; this trainer only accepts
two images with IDs `a` and `b`.

`store.py` and `config.py` are torch-free; `loss.py` owns the mathematical
objective; `sdxl.py` owns model/adapter conditioning and inference; `generation.py`
and `training.py` orchestrate collection and learning; `cli.py` connects them to
the UI. Another model family should provide its own model implementation without
changing the rating schema or pairing semantics.

Run backend checks from the frontend checkout with
`backend\sd_scripts\venv\Scripts\python.exe -m unittest discover -s backend/tests`.
Qt checks use `venv\Scripts\python.exe -m unittest discover -s tests -p test_preference_ui.py`.
