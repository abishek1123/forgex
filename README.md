# AI-Based Restoration of Degraded Images — KLA PS01

Restores degraded semiconductor inspection images: removes speckle and Gaussian
noise and upscales 2× (128×128 → 256×256, or 256×256 → 512×512).

**Team:** *<team name>* — *<member names>* — *<college>*

---

## Quick start — running inference

```bash
git clone <repo-url>
cd kla-restore
pip install -r requirements.txt

python inference.py --input  /path/to/degraded_images \
                    --output /path/to/restored_images
```

That is the whole thing. The script auto-detects CUDA and falls back to CPU,
loads the weights from `weights/model.pt`, restores every image in `--input`,
and writes results to `--output` under the same filenames.

**Supported input formats:** `.npy` (float32, primary), `.png`, `.tif`/`.tiff`.
Output is written in the same format as the input. `.npy` outputs are float32
clipped to `[0, 1]`.

Useful flags:

| Flag | Effect |
|---|---|
| `--weights PATH` | use a different checkpoint |
| `--device cpu` | force CPU |
| `--no-fp16` | disable half precision (fp16 is on by default on CUDA) |
| `--tta` | 8× self-ensemble: roughly +0.2 dB, roughly 8× slower. **Off by default.** |

The script prints total wall-clock time and milliseconds per image.

---

## Results

Measured on our held-out validation split of 200 real KLA pairs
(`make_split(seed=0)` — never seen during training).

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ms/image |
|---|---|---|---|---|
| Bicubic ×2 (no denoising) | 23.40 | — | — | — |
| **Ours** | *TBD* | *TBD* | *TBD* | *TBD* |

Reproduce with `python src/validate.py --data data/train --ckpt weights/model.pt --baseline`.

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
inference.py            ← THE SUBMISSION SCRIPT (input dir → output dir)
weights/model.pt        ← trained weights
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

**Hardware used:** *<GPU, platform>* · **Training time:** *<hours>* ·
**Model size:** *<params>* · **Inference:** *<ms/image on ...>*

---

## Notes for reviewers

* `inference.py` imports only `torch` and `numpy` (plus `Pillow`, and only when
  the inputs are `.png`/`.tif`). It downloads nothing at runtime.
* It resolves `weights/model.pt` relative to its own location, so it can be run
  from any working directory.
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
