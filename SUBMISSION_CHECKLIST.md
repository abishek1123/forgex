# ForgeX — KLA PS01 final submission checklist

Verified against the organisers' announcement. Every box below was tested, not assumed.

## Required structure

```
kla-restore/
├── run.py               ← entry point:  python run.py <input-dir> <output-dir>
├── requirements.txt     ← pinned versions
├── README.md            ← setup + execution instructions
└── models/
    └── model.pt         ← trained weights, 5.5 MB
```

## Technical checks — all verified by running

| Requirement | Status | How it was verified |
|---|---|---|
| `run.py` reads all `.npy` from input dir | PASS | 4-file directory, all consumed |
| Creates output dir if missing | PASS | tested against a non-existent path |
| One output per input | PASS | script exits non-zero if counts differ |
| Output filename == input filename | PASS | set-equality check on filenames |
| Shape `(H,W)` or `(H,W,1)` | PASS | `(128,128,1)` in → `(256,256,1)` out |
| Values in `[0,1]`, no NaN/Inf | PASS | explicit `nan_to_num` + `clamp(0,1)` |
| Correct target resolution | PASS | 128→256 and 256→512 in one run |
| Weights + supporting files included | PASS | `models/model.pt`, no external files needed |
| `requirements.txt` pinned | PASS | `torch==2.13.0`, `numpy==2.4.4` |
| README setup + execution | PASS | rewritten for `run.py` |
| Runs on NVIDIA GPU, no internet / keys / downloads / interaction | PASS | imports only torch + numpy; nothing fetched at runtime |

## Hardening beyond the checklist

- `run.py` is **fully self-contained** — network definition inlined, so it imports
  nothing from `src/` and cannot fail on a missing project file.
- Verified **bit-identical** output to the previous script, so all reported metrics stand.
- Ran from a directory containing only the four required items, invoked by absolute
  path from a different working directory.
- Mixed input sizes in a single directory are grouped and batched correctly.

## Results (200-image held-out split)

| Method | PSNR | SSIM | LPIPS | ms/image |
|---|---|---|---|---|
| Bicubic ×2 | 23.23 | 0.548 | — | — |
| **ForgeX (1.37 M params)** | **28.43** | **0.764** | **0.309** | **9.7** |

+5.19 dB over bicubic. End-to-end incl. disk I/O: 31.1 ms/image over 400 images.
