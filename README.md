# AI-Based Restoration of Degraded Images — KLA PS01

Restores degraded semiconductor inspection images: removes speckle and Gaussian
noise and upscales 2× (128×128 → 256×256, or 256×256 → 512×512).

**Team ForgeX** — Abishek SR · Anmol BA · Hardik — VIT Vellore

---

## Quick start

```bash
pip install -r requirements.txt

python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py ./test_images ./restored
```

That is the whole thing. `run.py` reads every `.npy` file in the input
directory, restores each at 2× resolution, and writes one `.npy` file of the
same name into the output directory (created automatically if it does not
exist). It auto-detects CUDA and falls back to CPU.

**Input:** `.npy`, grayscale, shape `(H, W)` or `(H, W, 1)`, float32. Values may
fall outside `[0, 1]` — that is expected for this degradation and is handled.

**Output:** `.npy`, float32, shape `(2H, 2W)` — or `(2H, 2W, 1)` if the input
carried a trailing channel axis. Guaranteed finite (no `NaN`/`Inf`) and clipped
to `[0, 1]`.

**Requirements:** `torch` and `numpy` only. No internet access, no API keys, no
model downloads, no user interaction, no manual configuration. Weights ship in
`models/model.pt`.

Optional flags (none required):

| Flag | Effect |
|---|---|
| `--weights PATH` | use a different checkpoint (default `models/model.pt`) |
| `--device cpu` | force CPU |
| `--no-fp16` | disable half precision on CUDA |
| `--tta` | 8× self-ensemble: +0.04 dB, ~7× slower. **Off by default.** |
| `--batch N` | images per forward pass (default: 16 on GPU, 1 on CPU) |

The script prints total wall-clock time and milliseconds per image.

---

## Results

Measured on our held-out validation split of 200 real KLA pairs
(`make_split(seed=0)` — never seen during training).

Held-out split of 200 real pairs, never seen during training.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ms/image |
|---|---|---|---|---|
| Bicubic ×2 (no denoising) | 23.23 | 0.548 | — | — |
| **Ours — 1.37 M params** | **28.43** | **0.764** | **0.309** | **9.7** |
| Ours — 3.74 M params (rejected) | 28.45 | 0.765 | 0.315 | 21.0 |

**+5.19 dB over bicubic.** The larger model was rejected: 2.7× the parameters
bought +0.026 dB and cost 1.7× the inference time.

End-to-end throughput including disk I/O: **31.1 ms/image** over 400 images.

Reproduce with `python src/validate.py --data data/train --ckpt models/model.pt --baseline`.

---

## Approach

**The degradation, reverse-engineered.** We fitted KLA's own generator from the
supplied pairs. It is a 2×2 box-average downsample plus signal-dependent noise:

```
var(residual)  =  σ_add²  +  σ_mul² · I²
σ_mul ~ U(0.13, 0.21)     speckle — multiplicative, grows with brightness
σ_add ~ U(0.00, 0.07)     additive Gaussian
```

Fitting a Gaussian blur kernel before decimation selects σ = 0 on every pair, so
the downsample is an exact box average with no extra blur. The noise is
spatially white and its level is re-drawn per image. `tools/calibrate.py`
reproduces this fit and checks our synthetic degradation against the real one.

**Why that matters.** Knowing the generator lets us synthesise unlimited
training pairs from the 3200 ground-truth images, with fresh noise every epoch
and the three operations applied in a random order (KLA's brief: *"do not read
into the order of it"*). The model cannot memorise a fixed noise pattern, so it
is pushed to learn the noise physics — which is what transfers to the
out-of-distribution test set. We train on ranges deliberately **wider** than we
measured, and keep 30% real pairs in the mix as insurance.

**Architecture** (`src/model.py`): a residual CNN that does all its work at low
resolution and upsamples only in the final block via PixelShuffle — 4× cheaper
than upsample-first designs, which matters because inference time is scored.
A global bicubic skip means the network predicts only the *correction* to a
cheap baseline, so it starts at the 23.4 dB bicubic score and converges fast.
A parameter-free variance-stabilising stem feeds the first convolution raw,
`√x` and `log(1+x)` views of the input, so it can pick the representation in
which speckle noise is closest to uniform. No BatchNorm.

**Loss** (`src/losses.py`): Charbonnier + SSIM + gradient. No GAN and no
perceptual loss — both work by synthesising plausible texture, and invented
texture on an inspection image is a fabricated defect. The gradient term
sharpens edges only where the ground truth has edges.

---

## Repository layout

```
run.py                  ← THE SUBMISSION SCRIPT: python run.py <in-dir> <out-dir>
models/model.pt         ← trained weights, 1.37 M params, 5.5 MB
src/
  degrade.py            degradation model (the "damage machine")
  dataset.py            real + synthetic pair loading, augmentation, split
  model.py              the network
  train.py              training loop
  losses.py             Charbonnier / SSIM / gradient
  metrics.py            PSNR, SSIM, LPIPS (LPIPS optional)
  validate.py           score a checkpoint, append a row to results.csv
tools/
  check_data.py         verify dataset layout before training
  calibrate.py          does our synthetic damage match KLA's real damage?
  preview.py            before/after figure for the deck
outputs/                restored test-set images
results.csv             one row per training run
```

---

## Reproducing training

Expected data layout:

```
data/train/GT/000000.npy        256×256 float32 in [0,1]
data/train/NoisyLR/000000.npy   128×128 float32, may fall outside [0,1]
```

```bash
python tools/check_data.py --data data/train      # verify layout
python tools/calibrate.py  --data data/train      # verify degradation model
python src/train.py --data data/train --smoke     # ~5 s, CPU, checks plumbing

python src/train.py --data data/train --out runs/v1 --amp \
                    --epochs 60 --iters 500 --batch 32 --ch 64 --nb 16

python src/validate.py --data data/train --ckpt runs/v1/best.pt --baseline
cp runs/v1/best.pt weights/model.pt
```

Interrupted run? `--resume runs/v1/last.pt` picks up exactly where it stopped,
optimiser and schedule included.

The control experiment for the loss ablation:

```bash
python src/train.py --data data/train --out runs/charb --loss charbonnier --amp
```

**Hardware used:** NVIDIA RTX 4050 Laptop, 6 GB (75 W), Windows 11 · **Training
time:** 4.3 h (120 epochs, 60,000 steps) · **Peak VRAM:** 0.95 GB · **Model
size:** 1.37 M parameters, 5.5 MB · **Inference:** 9.7 ms/image on RTX 4050

---

## Notes for reviewers

* `run.py` is fully self-contained — the network definition is inlined, so it
  imports only `torch` and `numpy` and depends on no other file in this repo
  except the weights. It downloads nothing at runtime.
* It resolves `models/model.pt` relative to its own location, so it can be run
  from any working directory.
* Outputs are explicitly passed through `nan_to_num` and clamped to `[0,1]`, so
  the finite-value and range guarantees hold regardless of input.
* Nothing in the model hardcodes an input resolution; 256×256 → 512×512 works
  without changes.
* `requirements-full.txt` is the complete `pip freeze` of the training
  environment; `requirements.txt` is the minimal set needed to run inference.

## References

1. B. Lim et al., *Enhanced Deep Residual Networks for Single Image
   Super-Resolution*, CVPRW 2017. (residual SR backbone, no BatchNorm)
2. W. Shi et al., *Real-Time Single Image and Video Super-Resolution Using an
   Efficient Sub-Pixel Convolutional Neural Network*, CVPR 2016. (PixelShuffle)
3. W.-S. Lai et al., *Deep Laplacian Pyramid Networks for Fast and Accurate
   Super-Resolution*, CVPR 2017. (Charbonnier loss)
4. H. Zhao et al., *Loss Functions for Image Restoration with Neural Networks*,
   IEEE Trans. Computational Imaging, 2017. (L1 + MS-SSIM loss design)
5. R. Zhang et al., *The Unreasonable Effectiveness of Deep Features as a
   Perceptual Metric*, CVPR 2018. (LPIPS)
6. F. J. Anscombe, *The transformation of Poisson, binomial and
   negative-binomial data*, Biometrika, 1948. (variance stabilisation)
