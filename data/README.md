# data/

Everything in this folder is **ignored by git** except this file and `SOURCES.md`.
Images are too big for GitHub, and the photos may show people's handwriting.
Move data between machines with a USB drive or a shared drive, not git.

## Layout

```
data/
  raw_public/          downloaded public datasets (scripts/download_public.py, coming next)
  raw_own/             our Pixel photos
    originals/         uncropped photos, kept for future "Step 1" detector work
    no_symbol/         cropped photos, one folder per class
    unknown_symbol/
    triangle/
    quadrilateral/
    ellipse/
    novel_shapes/      shapes that appear NOWHERE in training (metadata class: novel_shape)
    metadata.csv       one row per photo (columns below)
  processed/           224x224 RGB images, ready for training (made by scripts/prepare_data.py)
    manifest.csv       one row per image: split, label, file, source, drawer, original location
    train/<class>/     one folder per class: no_symbol, unknown_symbol, triangle, quadrilateral, ellipse
    val/<class>/
    test/<class>/      our own photos ONLY
    test_novel_shapes/unknown_symbol/              shapes never used in training
```

## `raw_own/metadata.csv` columns

Start from the template: copy `data/metadata_template.csv` to `data/raw_own/metadata.csv`, delete the three EXAMPLE rows, and add one row per photo. It opens in Excel, Google Sheets or LibreOffice Calc; save it as CSV.

```bash
mkdir -p data/raw_own/no_symbol data/raw_own/unknown_symbol data/raw_own/triangle data/raw_own/quadrilateral data/raw_own/ellipse data/raw_own/novel_shapes data/raw_own/originals
```

```bash
cp data/metadata_template.csv data/raw_own/metadata.csv
```

```
filename,class,drawer_id,pen,surface,distance_m,angle_deg,lighting,notes
IMG_0001.jpg,triangle,D01,sharpie_thin,white_notecard,1,30,room,slightly open corner
```

- `drawer_id`: who drew the symbol (D01, D02, ...). Use IDs, not names. Photos are split into train/val/test **by drawer**, so one person's drawings never land in two splits.
- `class`: `no_symbol`, `unknown_symbol`, `triangle`, `quadrilateral`, `ellipse`, or `novel_shape` (photo goes in `novel_shapes/` and is used only in the special novel-shapes test).
- The photo file must be in the folder named after its class, e.g. `raw_own/triangle/IMG_0001.jpg`. `prepare_data.py` lists any row whose file is missing or whose class is misspelled.
- **What label does a photo get?** The outline of the **main thing** in the cropped photo, whether it is drawn, painted, printed or a real object:
  - `triangle`: drawn triangles, yield/warning signs, triangular stickers
  - `quadrilateral`: drawn squares/rectangles/diamonds/kites, cards, windows, books, picture frames, a kite (the toy)
  - `ellipse`: drawn circles/ovals, balls, plates, clocks, coins, round signs
  - `unknown_symbol`: a clear shape that is none of those: stars, arrows, X's, hexagons, zigzags, squiggles
  - `no_symbol`: nothing clear: blank cards, walls, floors, carpet, clutter, plants, irregular objects
- **`drawer_id` for real objects:** use the person who took the photo plus the object, e.g. `D01_ball`, so the same ball never lands in two splits.
- **Choosing novel shapes:** they must not appear in ANY training data. Held back since 2026-11-20: **hearts, the letter "A", lightning bolts**. Everything in `novel_shapes/` should be one of those (the letter "A" is the most interesting: it has a triangle inside it).
- **Don't use** octagons or smiley faces anywhere: their outline is basically an ellipse, so the label is unclear.
- `distance_m`: 0.3, 1, 2 or 3. `angle_deg`: 0, 30 or 60.
- `lighting`: room, daylight, dim, flash or shadow.
