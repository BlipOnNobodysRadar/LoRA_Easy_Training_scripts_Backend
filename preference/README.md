# SDXL preferences, aligned edits, and concept training

Experimental, local, single-GPU preference training. Ordinary supervised training
keeps its existing configuration and code path. This subsystem trains a **separate
UNet LoRA**, leaving the input checkpoint and any starting adapters untouched.

## Selectable objectives

Choose **Adaptation objective** in Configuration. All methods train the separate
UNet adapter, support checkpoint/resume, and export the original adapter stack
plus the learned adjustment as one inference LoRA. DPO remains the default;
existing DPO configs and resume signatures retain their previous meaning.

| Method | Data and practical purpose |
| --- | --- |
| Diffusion-DPO | Rated winner/loser pairs; moves relative denoising errors toward the preferred image compared with the frozen reference. |
| ADDifT (aligned image edits) | Explicitly imported, aligned before/after images with a human winner; learns the illustrated visual change. |
| LECO (text concept) | Prompt-defined concept direction; no images or ratings required. |

Quality labels and reason tags are review metadata, **not additional losses**.
DPO and ADDifT use preference strength (slight 0.5, normal 1, strong 2 by default).
LECO does not use pair feedback. No arbitrary mixture of these losses is enabled:
their scales and sampling distributions differ. To combine methods, start a new
run using a previous adjustment as the optional trainable adapter (same frozen
reference), or use the exported combined LoRA as a new frozen starting adapter.
Clear Resume when changing method. New image datasets must match the reference.

### Aligned image import and ADDifT

Choose a separate Dataset directory for aligned edits. **Import aligned edit...**
accepts an original image, its edited counterpart, the shared scene prompt, and
an optional negative prompt. The files must have matching dimensions (multiples
of 64, 256–1536); the importer does not resize or realign them. Copies are hashed
and stored immutably. A is the original, B is the edit. Review each pair in Rate;
an ADetailer/inpainting result is not automatically the winner. Ties and skips do
not train. Alignment is your declaration, not something this importer proves.

Bulk import accepts a JSON list, with paths relative to the manifest:

```json
[
  {"source": "before/001.png", "target": "after/001.png", "aligned": true,
   "prompt": "a ceramic mug on a table", "negative_prompt": "blur", "seed": 123}
]
```

```powershell
backend\sd_scripts\venv\Scripts\python.exe -m backend.preference.cli import-pairs --config preference.local.json --manifest edits.json
```

Import is idempotent for the same reference/files/prompt, leaves source files
untouched, and preserves existing prompt-group holdouts. Omit seed when unknown;
original sampler settings for an external image are not invented. The current
reference is recorded as the model being adapted, not proof of external-image
generation provenance. Imported pairs can also train with DPO. Do not overwrite
images already in a collected dataset.

This **ADDifT-style SDXL variant** uses the VAE posterior mean, shared diffusion
noise and timesteps, default interval [400, 900), and MSE between:

```text
student: adapter +1, noisy preferred-image latent
teacher: adapter  0, noisy rejected-image latent
```

Both predictions use the same positive scene prompt. Optional alternate inverse
direction swaps the two latents and trains at adapter -1 on alternating optimizer
steps. Backward retains that sign during gradient-checkpoint recomputation. The
saved adapter is calibrated at strength 1. This follows the paired prediction
matching idea in [TrainTrain](https://github.com/hako-mikan/sd-webui-traintrain),
with explicit unit-strength training, deterministic VAE means, uniform timesteps,
and our existing frozen-reference/resume machinery. It is not a numerical port
of TrainTrain's quarter-strength and timestep schedule. An analytic Gaussian
denoiser test verifies the positive edit direction for both alternating signs.

Whole-image loss is currently used: no edit-mask loss, crop loss, automatic
quality label conversion, or guaranteed feature disentanglement. Small facial
edits may provide a weak signal relative to the rest of the image. Preference
for edits is viable, but evaluate fresh raw generations without ADetailer.

### LECO prompt settings

This mode reuses the bundled sd-scripts `PromptSettings.build_target` formula:

```text
enhance: desired prediction = neutral + strength * (concept - contrast)
erase:   desired prediction = neutral - strength * (concept - contrast)
```

All three predictions come from the frozen reference at the same latent/timestep.
The trainable adapter learns that prediction under **Prompt to change**. Latents
come from a randomly truncated reference DDIM trajectory using the configured
resolution, Partial sampling steps (default 20) and CFG (default 3). This uses
the [sd-scripts LECO objective](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_leco.md)
with reference-only DDIM trajectory sampling, rather than that trainer's policy
DDPM sampling. It supports the existing frozen adapter stack and stop/resume.
This is a deliberately identified sampling variant, not a reproduction claim.

For a simple red-to-blue mug experiment: set Prompt to change, Neutral, and
Contrast to `an illustration of a red ceramic mug`; set Concept to
`an illustration of a blue ceramic mug`; choose Enhance, strength 1. This makes
the target prediction exactly the reference's blue-mug prediction. Keep
generation width/height at 512 for an inexpensive execution experiment, then
evaluate at the resolution you actually use. Ordinary generation prompts,
negative prompts, sampling steps and sampler do not configure LECO's concept
loss. DPO beta is disabled in the UI for both edit methods.

LECO training loss measures matching its text-defined teacher. It supplies no
held-out image preference accuracy and no automatic quality assessment. Test
unseen seeds, scenes, and objects at adapter strengths -1, 0 and +1.

## Two starting points

* **Checkpoint only:** leave `model.base_loras` empty. The frozen reference is
  the checkpoint; the output is a new preference LoRA for that checkpoint.
* **Checkpoint plus existing LoRA:** provide `base_loras` as an ordered list of
  `{ "path": "...safetensors", "weight": 1.0 }`. Those adapters remain frozen
  and active in both policy and reference predictions. The output is a new
  preference LoRA to use **alongside the same checkpoint and original adapters
  at the recorded weights**.

Internally training uses an additive preference adapter. Completed runs that
start with LoRAs also export `combined_METHOD_stepNNNNNN.safetensors` beside the run's
checkpoints: a standalone inference LoRA containing the originals plus the DPO
adjustment at their recorded weights. Load the combined file **instead of** the
original and delta LoRAs, on the same base checkpoint, at weight 1. The original
files stay untouched. Checkpoint-only training already produces a usable standalone
preference LoRA. Only the preference adapter is disabled for reference inference; a second SDXL
copy is not resident on the GPU. Text encoders and VAE are frozen. The UNet uses
bf16, native SDPA and gradient checkpointing; trainable adapter weights use fp32.
AdamW states use fp32; SimplifiedAdEMAMixExM uses its native configurable state
storage (CPU bfloat16 by default). The VAE retains checkpoint precision and runs in fp32. Text
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
   Use **Add prompt** to add a positive/negative textbox pair and its own number
   of comparisons, or **Remove prompt** to remove it before generation. For
   example, 20 pairs for one prompt and 5 for another generates 25 comparisons
   (50 images). A multiline positive textbox is one prompt. Blank positive rows
   must be filled or removed; an empty negative is allowed.
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

### Exporting a combined LoRA

**Export combined LoRA...** lets you choose an existing `preference_lora.safetensors`
checkpoint and a new output filename. It uses the original stack and weights from
the current config; the preference checkpoint's reference identity must match.
The preference adjustment is included at weight 1. Export runs on CPU and uses
factor concatenation: it retains every supported text, convolution and normalization
delta, without SVD rank reduction. The output uses fp32 and can be significantly
larger than the original. It represents the same additive policy mathematically;
runtime rounding and differences between inference engines can change pixels.

Supported exports are plain LoRA/LoCon (linear or 1x1-up convolution factors) and
additive norm deltas. DoRA, LoCon mid factors and other algorithms are rejected,
never partially loaded. Output files are never overwritten. If an automatic
export is unsupported or cannot be written, training checkpoints remain usable
and the error is recorded in `status.json`. Stop requests during export are checked
between modules and before publishing the completed file.

For further DPO resume, use the separate checkpoint and its original reference
stack; a combined export is an inference artifact, not the resumable delta. You
can use it as the starting original LoRA for a new round with a new dataset.

```powershell
& .\backend\sd_scripts\venv\Scripts\python.exe -m backend.preference.cli combine `
    --config preference.local.json --preference path/to/preference_lora.safetensors `
    --output path/to/character_DPO.safetensors
```

Per-prompt configuration uses `generation.prompts` entries of the form
`{"prompt": "...", "negative_prompt": "...", "pairs": 20}`. Legacy string entries
still inherit `generation.negative_prompt` and `generation.pairs_per_prompt`.
Explicit empty negatives override that legacy default. GUI saves write explicit
objects. Prompt grouping still uses positive text, so using another negative for
the same positive cannot move that prompt into a different validation split.
Negative prompts guide generation and are recorded in pair/PNG metadata; current
Diffusion-DPO training conditions on the positive text only.

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

## Practical evaluation and data collection

The pair count needed for useful personal preference tuning has not been
established for this implementation. The original Diffusion-DPO experiment used
851,293 non-tied pairs and 58,960 prompts, a different scale and training setup;
its results do not validate a small personal LoRA dataset. See
[the paper's experimental setting](https://arxiv.org/html/2311.12908v1#S5.SS1).

As an engineering starting point, use a handful of pairs to check execution,
50-100 eligible pairs to seek an early narrow signal, and roughly 250-500 eligible
pairs across 30-50 meaningfully different prompts for a first serious character
experiment. These are trial budgets, not evidence-based minimums or guarantees.
Collect more generations than the desired eligible count because ties/skips are
excluded. For broader style/quality preferences, expect substantially more data
and broader subject/style coverage; expand only after a held-out benefit appears.

For a character, vary poses, camera distance, expressions, outfits, lighting,
backgrounds and requested styles. Repeating one prompt hundreds of times does
not test generalization. Choose winners consistently: if closeups always win
over full-body compositions, a closeup bias may be learned alongside any improved
character details. Negative prompts should reflect intended use. If both images
are bad, a relative winner is valid, but it does not teach an absolute quality
threshold; tie/skip when there is no useful preference.

Hold out whole prompt groups (the default grouping does this), inspect actual
train/validation counts, and also compare fresh generations on prompts/seeds
not used to select checkpoints. Include unrelated subjects to detect unwanted
effects: an active preference LoRA is not automatically gated to a trigger word.
The frozen reference anchors learning but does not guarantee that the adapted
model preserves diversity or improves on unseen prompts.

Compare baseline versus adapted inference with identical checkpoint, prompt,
negative, seed, dimensions, sampler, sampling steps, CFG and inference engine.
For the delta file, baseline is the original stack and treatment is that same
stack plus `preference_lora:1`; alternatively replace the original stack with
the combined export at weight 1. Review several prompts/seeds, preferably with
blinded A/B identities. An image-quality win is the acceptance criterion; a lower
training loss or perfect denoising preference accuracy on a tiny holdout is not
sufficient. Current validation ranks recorded pairs at fixed noise/timestep
draws; it does not generate new validation images automatically.

An optimizer step samples `gradient_accumulation` pairs with replacement. The
expected presentations per training pair are approximately
`max_steps * gradient_accumulation / train_pair_count`. With 50 steps,
accumulation 4 and two training pairs, that is about 100 presentations per pair,
with newly sampled noise/timesteps, rather than 200 independent examples. Keep
early runs short and compare intermediate checkpoints before increasing steps.

## Optimizer and learning-rate configuration

Existing configurations keep AdamW (weight decay 0), constant learning rate and
unit gradient clipping. The training JSON also supports the installed native
`SimplifiedAdEMAMixExM` optimizer and a single-cycle RAWR learning-rate schedule.
These advanced fields are preserved when a config is loaded/saved in the GUI;
dedicated optimizer widgets are not yet present. For example, within `training`:

```json
{
  "learning_rate": 0.0004,
  "max_steps": 100,
  "optimizer": {
    "type": "SimplifiedAdEMAMixExM",
    "args": {
      "betas": [0.99, 0.997], "beta1_warmup": "total_steps", "min_beta1": 0.95,
      "alpha": 1.0, "amsgrad_min_decay_rate": 0.96, "amsgrad_max_decay_rate": 0.96,
      "use_adabelief": true, "torch_compile": false, "update_strategy": "cautious"
    }
  },
  "lr_schedule": {"type": "rawr", "warmup_ratio": 0.05, "min_lr": 0.000001, "gamma": 0.9, "d": 0.9},
  "max_grad_norm": 0.0
}
```

This is an experimental recipe, not a quality recommendation. The simplified
optimizer names its run-length momentum setting `beta1_warmup`, not `beta3`.
`"total_steps"` resolves to `max_steps`; a positive fixed integer or null is also
accepted. RAWR spans `max_steps`, with warmup rounded up to complete optimizer
steps (five here). The scheduler advances after each accumulated optimizer step.
`max_grad_norm: 0` disables global norm clipping while still checking finiteness;
the optimizer's own update bounds remain active. Other native defaults remain
in effect, except custom `torch_compile` defaults to false. Unsupported options
are rejected rather than ignored; see `optimization.py` for the allowed fields.

Runs save `optimization.json` and log the learning rate actually used, the next
rate, and effective momentum beta1 where applicable. Checkpoints include scheduler
state and preserve offloaded optimizer state precision/location on resume. The
schedule horizon is fixed for RAWR and automatic momentum warmup: increasing
`max_steps` then requires a new run. Legacy constant-AdamW resumes can still
increase it. The inference sampler, diffusion noise schedule and learning-rate
schedule are separate settings.

Uniform diffusion timestep sampling, squared denoising error and the sigmoid DPO
objective remain unchanged. Quality labels and reason text are recorded only;
explicit A/B choices and strength weights supply the loss signal.

Alternative L1/Huber denoising losses, SNR weighting and different preference
objectives require their own validation. Replacing the squared Gaussian
denoising error changes the standard Diffusion-DPO surrogate. They should be
explicit experimental modes rather than ordinary-training controls that appear
to work but are ignored. Beta is also not a simple strength slider: it affects
reference regularization and gradient scaling, so tune it jointly with learning
rate using held-out results. The paper discusses this interaction in
[its hyperparameters and beta ablation](https://arxiv.org/html/2311.12908v1#S5.SS1).

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
* `training_state.pt`: optimizer, optional learning-rate scheduler and RNG state,
  loaded with `weights_only=True`.
* `state.json`: step, signature and metrics; `latest.json` in the parent points
  to the latest complete checkpoint.

Set the resume checkpoint directory in the UI, or pass
`train --config ... --resume ...\checkpoint-000025`. Resume requires the same
rated dataset and training settings. You may change the checkpoint interval, or
increase `max_steps` if neither RAWR nor automatic momentum warmup fixes the
schedule horizon. Only the latest checkpoint may continue in the same run.
To add fresh human ratings between rounds, start a **new run** with the previous
preference adapter as `model.preference_lora`, clear Resume, and keep the original
checkpoint/adapters fixed. The reference remains fixed across rounds.

Validation reports fixed-noise held-out DPO loss and preference accuracy. A
falling training loss or a synthetic smoke-test pass does not demonstrate better
images. Judge improvement with fresh human comparisons on unseen prompts.

## Other research and scope

[LoFA](https://github.com/GAP-LAB-CUHK-SZ/LoFA) now publishes code, but its released
preview checkpoint targets MotionX action video with explicitly limited
generalization. Its identity-personalized image checkpoint is still listed as
unreleased (checked 2026-09-14). Fast prediction comes after training a suitable
hypernetwork. This subsystem does not implement LoFA or train an SDXL predictor.

[LoRA.rar](https://github.com/donaldssh/LoRA.rar) does provide a pretrained SDXL
hypernetwork. It predicts merging coefficients for existing subject/style
adapters, rather than predicting a new identity adapter from examples. Its
released implementation predicts coefficients for attention Q/output projections
and adds K/V normally; it is not an arbitrary feature filter for a full
LoCon/normalization adapter. Its repository license is CC BY-NC-SA 4.0. Neither
its code nor its weights are bundled here. A separate compatibility experiment
would need to account for every original tensor and evaluate preservation and
style transfer. There is no LoFA/LoRA.rar checkbox that silently does nothing.

Learning a gloss slider and suppressing it in a style adapter may help, but a
direction is not guaranteed orthogonal to identity, anatomy or other style
features. Preventing those features from being learned needs matched examples
and preservation evaluation; predicting weights faster does not solve this.

## Implementation boundaries

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
