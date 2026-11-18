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
    novel_shapes/      shapes that appear NOWHERE in training (metadata class: novel_shape)
    metadata.csv       one row per photo (columns below)
  processed/           224x224 RGB images, ready for training (made by scripts/prepare_data.py)
    manifest.csv       one row per image: split, label, file, source, drawer, original location
    train/{no_symbol,unknown_symbol,triangle}/
    val/{no_symbol,unknown_symbol,triangle}/
    test/{no_symbol,unknown_symbol,triangle}/      our own photos ONLY
    test_novel_shapes/unknown_symbol/              shapes never used in training
```

## `raw_own/metadata.csv` columns

Start from the template: copy `data/metadata_template.csv` to `data/raw_own/metadata.csv`, delete the three EXAMPLE rows, and add one row per photo. It opens in Excel, Google Sheets or LibreOffice Calc; save it as CSV.

```bash
mkdir -p data/raw_own/no_symbol data/raw_own/unknown_symbol data/raw_own/triangle data/raw_own/novel_shapes data/raw_own/originals
```

```bash
cp data/metadata_template.csv data/raw_own/metadata.csv
```

```
filename,class,drawer_id,pen,surface,distance_m,angle_deg,lighting,notes
IMG_0001.jpg,triangle,D01,sharpie_thin,white_notecard,1,30,room,slightly open corner
```

- `drawer_id`: who drew the symbol (D01, D02, ...). Use IDs, not names. Photos are split into train/val/test **by drawer**, so one person's drawings never land in two splits.
- `class`: `no_symbol`, `unknown_symbol`, `triangle`, or `novel_shape` (photo goes in `novel_shapes/` and is used only in the special novel-shapes test).
- The photo file must be in the folder named after its class, e.g. `raw_own/triangle/IMG_0001.jpg`. `prepare_data.py` lists any row whose file is missing or whose class is misspelled.
- **Choosing novel shapes:** they must not appear in ANY training data. The public data already contains hearts, crescents, letters, pentagons, cylinders (Hand-Drawn Shapes "other") and lightning bolts (Quick, Draw!), so don't use those. Shapes the training data doesn't contain include a spiral, an infinity sign (∞), a peace sign, a plus sign (+) and the letter "A".
- `distance_m`: 0.3, 1, 2 or 3. `angle_deg`: 0, 30 or 60.
- `lighting`: room, daylight, dim, flash or shadow.
