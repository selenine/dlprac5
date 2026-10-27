# R3GAN on CIFAR-10 horses: experiment log

All the GAN experiments run in this repo, what each one changed, and what came out. The final
configuration is the default in `src/workspace/gan.py`.

## Task and constraints

- Generate CIFAR-10 horses (class 7). 4,000 training images; 1,000 held-out validation images.
- At most **1M parameters** (generator + discriminator + buffers), **10k gradient steps each for
  G and D** (one D and one G update per iteration), **batch size 64**, **no gradient accumulation,
  no distillation, no pretrained models**.
- Metric: **KID × 1000** (lower is better), computed by the provided evaluation code from 1,000
  generated images against the validation split, using Inception-v3 features (evaluation only;
  no pretrained network is used in training).
- Hardware: one RTX 5090. A full run (10k steps + evaluation) takes about 11–15 minutes.

## Final result

| | Seed 3801 | Seed 3802 | Mean |
|---|---|---|---|
| Starting point: R3GAN as specified (lr 2e-4 → 5e-5) | 47.8 | 52.5 | 50.2 |
| **Final config** (flat lr 7.5e-4, G-heavy widths) | **2.50** | **3.91** | **3.21** |

About a **15× lower KID**. Almost all of it comes from the learning rate; the architecture change on
top is within noise (see below).

**Final config** (defaults in `gan.py`):

- R3GAN: relativistic pairing loss `softplus(D(G(z)) − D(x))` with zero-centred R1 + R2 gradient
  penalties, γ cosine-decayed 0.05 → 0.005.
- Normalization-free ConvNeXt-style residual blocks (1×1 expand ×2, grouped 3×3 with group width
  16, 1×1 project), leaky ReLU 0.2, fix-up initialization (last layer of each branch zero, others
  scaled by L^(−1/4)), bilinear up / average-pool down.
- Generator: 64-d z modulating a learned 4×4 basis, stages at 4/8/16/32 px, widths
  112·112·96·64, 2 blocks per stage. Discriminator: widths 64·80·80·96, 2 blocks per stage,
  depthwise 4×4 basis + linear logit. **968,308 parameters** (558k G / 411k D), no buffers.
- Adam, betas (0, 0.99), **flat learning rate 7.5e-4** for both networks.
- Real images x-flipped with probability 0.5; no other augmentation.
- Generator EMA with a 10k-image half-life (ramped up over the first 5% of images), used for
  evaluation.

## How runs were organised

- **Seeds and noise.** Unless noted, runs use training seed 3801. Re-running the same config with
  seed 3802 moved KID by **4.7** at vanilla's level (47.8 vs 52.5), **3.4** at ~30, and **0.9–1.4**
  at ~3. Single-run differences smaller than that are not meaningful, so every candidate for the
  final config was confirmed on a second seed.
- **Infrastructure.** Runs execute under `supervisor` so they survive session drops:
  `gan_sweep.sh` runs each line of `sweeps/sweep.txt` (`<out_dir> <seed> <JSON CONFIG
  overrides>`) in order, writing to `results/` and skipping runs whose zip exists. Two planner jobs
  (`sweeps/plan.py`, `sweeps/plan_arch.py`) read finished results and queued follow-up runs
  automatically (learning-rate ladder, architecture/no-flip/EMA selection, seed checks); their
  decisions are in `logs/plan*.log`. The supervisor scripts and configs live under `/opt` and
  `/etc` on the training machine; copies are in `sweeps/supervisor/`.
- **Logging.** Losses, R1/R2, logits and schedules go to the terminal and to W&B (project
  `r3gan-cifar-horses`), with sample grids every 1000 steps.
- Every run folder `results/lab-05-hhtt85*/` holds the exact source it ran (`lab-05-code.py`), its config,
  metrics and loss history (`lab-05-results.json`), and sample grids.

## Phase 1: architecture and method changes (all worse)

All at the original learning rate (2e-4 → 5e-5), seed 3801. Vanilla R3GAN: **47.8** (52.5 on seed
3802).

| Run | Change | KID×1000 |
|---|---|---|
| `-dw` | ExtraDW: depthwise 5×5 at the start of each block (MobileNet-V4 UIB / ConvNeXt) | 53.5 |
| `-fourier` | Generator input from StyleGAN3-style Fourier features modulated by z | 59.8 |
| `-af` | Adversarial Flow Models (arXiv 2511.22475), one-step: 3×32×32 noise, U-Net generator, OT loss ‖G(z) − z‖² (0.2 → 0.005), gradient normalization, logit centering | 60.1 |
| `-mp` | EDM2 magnitude-preserving layers: forced weight normalization, MP residual sums, no biases, learned output gains, lr ×10 | 90.5 |
| `-sg3` | StyleGAN3 stack: mapping network, Fourier input, modulated convs, per-layer filtered activations, magnitude EMA, style mixing, D input blur, ADA | 207 |

Notes:

- ExtraDW is within noise of vanilla but 2.6× slower (depthwise double-backward is slow in cuDNN).
- AF's samples were noticeably softer. Its OT term needs image-shaped z, which forced a U-Net
  generator into the same budget.
- The magnitude-preserving run's 10× learning rate was a guess, and D ran ahead of G throughout.
  It wasn't retuned.
- The StyleGAN3 run changed many things at once, so it says nothing about individual components.

## Phase 2: optimisation hyperparameters

Same vanilla architecture, seed 3801 unless noted.

### γ (R1/R2 weight) and other R3GAN settings, at the original learning rate

| Run | Change | KID×1000 |
|---|---|---|
| `-gamma-const` | γ held at 0.05 | 67.0 |
| `-gamma-0.02` | γ decays to 0.02 | 56.9 |
| (vanilla) | γ decays to 0.005 | **47.8** / 52.5 |
| `-gamma-0.002` | γ decays to 0.002 | 57.6 |
| `-aug-0.55` | R3GAN's CIFAR augmentation (ADA pipeline: geometric + colour + cutout; p cosine 0 → 0.55) | 55.2 |
| `-beta2-ramp` | R3GAN's Adam β2 ramp 0.9 → 0.99 | 57.7 |

γ = 0.005 at the end is best on both sides. The rising R1/R2 seen late in training is not a problem
to fix. Augmentation and the β2 ramp didn't help.

### Learning rate (the big one)

| Schedule | Seed 3801 | Seed 3802 |
|---|---|---|
| 2e-4 → 5e-5 (vanilla) | 47.8 | 52.5 |
| flat 1.5e-4 | 39.6 | |
| 2e-4 → 1e-4 | 29.9 | 33.3 |
| flat 2e-4 | 28.9 | |
| flat 3e-4 | 12.3 | |
| flat 4e-4 | 8.2 | |
| flat 5e-4 | 5.2 | |
| **flat 7.5e-4** | **2.80** | **3.73** |
| flat 1e-3 | 3.1 | |

KID falls steadily as the learning rate rises, up to a plateau around 7.5e-4 to 1e-3. R3GAN's
schedule was tuned for ~20M training images. With 640k images (10k × 64), it spends most of the
run at learning rates far too low to make progress.

Also tried at 2e-4 → 1e-4:

- γ decaying to 0.002: 42.3 (vs 29.9)
- augmentation p → 0.3: 41.4 (vs 29.9)

Both are worse. Augmentation stays off and γ stays at 0.005.

## Phase 3: architecture, flips and EMA at flat 7.5e-4

Each variant was sized to fit under 1M parameters. Results are on seed 3801.

| Run | Shape | Params | KID×1000 |
|---|---|---|---|
| `-arch-g-heavy` | Budget shifted to G: G 112·112·96·64, D 64·80·80·96 | 968k | **2.50** |
| `-arch-deeper` | 3 blocks per stage, narrower (G 80·80·64·48) | 962k | 2.54 |
| vanilla | G 96·96·96·64, D 64·96·96·96, 2 blocks per stage | 941k | 2.80 |
| `-arch-deep-gheavy` | Both of the above combined (D shrinks to 401k to fit) | 983k | 3.81 |
| `-arch-lowres-heavy` | Channels concentrated at low resolution | 961k | 4.02 |
| `-arch-expand3` | Expansion ratio 3, narrower | 962k | 4.14 |
| `-arch-shallow-wide` | 1 block per stage, wider | 922k | 9.88 |

On the seed-3801 winner (G-heavy):

| Change | KID×1000 |
|---|---|
| No x-flips | 8.32 |
| EMA half-life 2.5k / 5k / **10k** / 20k / 40k images | 5.07 / 3.69 / **2.50** / 3.20 / 2.95 |

- Flips matter a lot with 4,000 images.
- The default 10k-image EMA half-life is best.
- Depth beats width: one block per stage is much worse.

The G-heavy seed-3802 verification gave 3.91, a two-seed mean of **3.21**, against **3.26** for
the vanilla architecture. That's the agreed selection rule, and the margin is tiny. **Treat the
two architectures as tied.**

## Takeaways

1. **The learning-rate schedule was the bottleneck.** The settings were borrowed from a much
   longer training run. Under a fixed 10k-step budget, a flat, roughly 4× higher learning rate cut
   KID by 15×. Everything else tried moved KID by at most the seed noise once the learning rate
   was right, except changes that made things clearly worse.
2. **Measure seed noise early.** Several early "results" (ExtraDW, Fourier input, AF) were within
   or near one seed's spread.
3. **Heavier machinery didn't pay off at this scale.** Within 1M parameters and 640k images, the
   StyleGAN3 components, ADA-style augmentation, EDM2 layers and the AF objective were all worse
   than plain R3GAN with a good learning rate.

## Reproducing

```bash
uv run workspace            # trains the final config, evaluates KID, writes lab-05-hhtt85.zip
```

To run a variant, override `CONFIG` before `run()`, as the sweep does:

```python
from workspace import gan
gan.CONFIG.update({"learning_rate": 3e-4, "final_learning_rate": 3e-4})
gan.run(out="my-run", seed=3801, show=False)
```

## Files

| Path | Contents |
|---|---|
| `src/workspace/gan.py` | Model, losses, training and evaluation; final config as defaults. Supports `aug_p`, `initial_beta2` and `architecture` overrides used in the sweeps. |
| `src/workspace/__init__.py` | `main()` entry point (`uv run workspace`) |
| `results/lab-05-hhtt85*/` | One folder per run: exact source, config, metrics, loss history, samples |
| `sweeps/sweep.txt` | The sweep queue |
| `sweeps/plan.py`, `sweeps/plan_arch.py`, `sweeps/arch-template.txt` | Planners that queued follow-up runs |
| `sweeps/supervisor/` | Copies of the supervisor scripts and configs that ran the sweeps |
| `logs/sweep.log` | Full training log of every sweep run |
| `logs/plan.log`, `logs/plan_arch.log` | The planners' decisions |
| `logs/train*.log` | Logs of the early single runs (Fourier, StyleGAN3, AF, MP, ExtraDW) |

The submission zips (`results/lab-05-hhtt85*.zip`), the dataset (`data/`, re-downloaded from the Hugging
Face mirror on demand) and local W&B files (`wandb/`, synced online) are git-ignored.
