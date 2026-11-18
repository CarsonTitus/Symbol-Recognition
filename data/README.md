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
    <class>/           cropped photos, one folder per class
    metadata.csv       one row per photo (columns below)
  processed/           224x224 RGB images, ready for training (scripts/prepare_data.py, coming next)
    train/{no_symbol,unknown_symbol,triangle}/
    val/{no_symbol,unknown_symbol,triangle}/
    test/{no_symbol,unknown_symbol,triangle}/      our own photos ONLY
    test_novel_shapes/unknown_symbol/              shapes never used in training
```

## `raw_own/metadata.csv` columns

```
filename,class,drawer_id,pen,surface,distance_m,angle_deg,lighting,notes
IMG_0001.jpg,triangle,D01,sharpie_thin,white_notecard,1,30,room,slightly open corner
```

- `drawer_id`: who drew the symbol (D01, D02, ...). Use IDs, not names. Photos are split into train/val/test **by drawer**, so one person's drawings never land in two splits.
- `class`: `no_symbol`, `unknown_symbol` or `triangle`.
- `distance_m`: 0.3, 1, 2 or 3. `angle_deg`: 0, 30 or 60.
- `lighting`: room, daylight, dim, flash or shadow.
