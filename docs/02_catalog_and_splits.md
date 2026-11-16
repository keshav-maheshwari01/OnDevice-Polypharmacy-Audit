# Notebook 02: Catalog and Splits

## Purpose

Decide **which pills our model supports** (the catalog) and **which photo is used for training, validation and testing** (the split). This is the most important notebook for fairness: a bad split gives results that look good but mean nothing. We made it separate so the split is created once and every later notebook reads the same files.

## What it does, in order

1. Looks at one pill's consumer photos to check that they are genuinely different shots.
2. Counts how many pills have 4 or more consumer photos (648) and exactly 5 (425).
3. Investigates the official fold files and works out how they split the data.
4. Builds the catalog: 40 pills chosen at random with a fixed seed (42) from the 648.
5. Splits each pill's photos, runs leakage checks, and saves the CSV files.

## Key findings

**The official folds split by pill type, not by photo.** Each of the five folds has 192 different pills, with all of their consumer photos, and no pill appears in two folds. So testing on a fold means testing on pills whose consumer photos were never seen in training. Our app supports a fixed list of pills the model has seen, so we built our own split. (The `_all` file is a copy of the full label file, and folds 0 to 4 hold only consumer images.)

**The catalog:** 40 pills, 188 consumer photos (28 pills with 5, 12 with 4), 80 reference photos (2 per pill).

**The split rule:** for each pill, shuffle its consumer photos (seed 42) and number them inside the pill. Number 0 goes to test, number 1 to validation, and the rest to train. All reference photos go to train. The number is only a counter, not a quality score.

| Split | Images | Consumer | Reference |
|---|---|---|---|
| Train | 188 | 108 | 80 |
| Validation | 40 | 40 | 0 |
| Test | 40 | 40 | 0 |

The count works out as 28 pills x 3 + 12 pills x 2 = 108 consumer training photos, plus 80 reference photos.

**Leakage checks (both gave 0):** no image appears in two splits, and no pill is missing a validation or test photo.

## Why these choices

- **Why a catalog?** The synopsis asks for a limited, fixed list of supported pills.
- **Why random with a fixed seed?** So nobody can say we picked easy pills, and anyone can reproduce the list.
- **Why 4 or more photos?** One for validation, one for test, at least two for training.
- **Why validation and test are real photos only:** the app will see real photos.
- **Why a locked test set?** So the final number is honest.

## Inputs and outputs

- Input: `all_labels.csv` and the fold files.
- Output: `catalog.csv`, `train.csv`, `val.csv`, `test.csv` in `data/processed/`. The CSV files are lists that point to the images. The images themselves are not copied.

## Possible viva questions

- *Why not use the official folds?* See above: they test unseen pills, not our fixed catalog.
- *How do you know there is no leakage?* Code checks for duplicate images across splits and for missing validation or test photos. We also looked at several pills' photos and they look different, though only a few were checked.
- *Could a pill's photos be near-duplicates?* Possibly for some pills. We only inspected a few.
- *Why are the CSVs not images?* They are lists that keep the split fixed and make checks easy, without copying files.

## Known issues

- The split files were first saved with relative paths. Make sure exactly one copy of each CSV exists, in `data/processed/`.
- One cell uses `cat` before the cell that creates it, so the notebook only works when cells run in a particular order. Reorder it.
- The catalog and split are for 40 pills. We plan 100 and 648 pills in a later notebook.
