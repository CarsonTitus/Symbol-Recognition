# Data sources and licenses

Every dataset we use, where it came from, and its license (CLAUDE.md §5a).
**Rule:** check a dataset's license before using it, and record it here. Mark "verified" only after reading the license yourself.

| Dataset | Link | License | Verified? | Used for |
|---|---|---|---|---|
| Hand-Drawn Shapes (HDS) | https://github.com/frobertpixto/hand-drawn-shapes-dataset | CC BY 4.0 (Creative Commons Attribution: free to use with credit) | Yes: dataset README, 2026-11-18 | `triangle` ← triangle; `unknown_symbol` ← rectangle, ellipse, other |
| Quick, Draw! (Google) | https://github.com/googlecreativelab/quickdraw-dataset | CC BY 4.0 | Yes: dataset README, 2026-11-18 | `triangle` ← triangle; `unknown_symbol` ← circle, square, hexagon, octagon, star, zigzag, line, lightning, diamond, smiley face, squiggle |
| Geometric Shapes (text background) | https://huggingface.co/datasets/0-ma/geometric-shapes | MIT | Yes: dataset card, 2026-11-18 | `no_symbol` ← label "1" (text, no shape) |
| Geometric Shapes Mathematics | https://www.kaggle.com/datasets/reevald/geometric-shapes-mathematics | **TODO** (see the "License" box on the Kaggle page) | No | `triangle` ← triangle; `unknown_symbol` ← others (mapping set after inspecting samples) |
| Four Shapes | https://www.kaggle.com/datasets/smeschke/four-shapes | **TODO** (Kaggle page) | No | `triangle` ← triangle; `unknown_symbol` ← square, circle, star |
| Our own photos | taken by the team on a Pixel 9 Pro XL | Ours; every photographed person/drawer agreed | n/a | all three classes; the **only** source for `test/` |

## Credit lines (for the presentation and README)

- Hand-drawn Shapes (HDS) Dataset © 2022 by Francois Robert, licensed under CC BY 4.0.
- Quick, Draw! data made available by Google, Inc. under the Creative Commons Attribution 4.0 International license.
- Geometric Shapes Dataset (`0-ma/geometric-shapes`) on Hugging Face, MIT license.
