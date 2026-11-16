# Notebook 01: Explore the Data

## Purpose

Before building anything, we must know what the data actually contains. This notebook only **looks**. It does not train or save anything. A wrong assumption here (for example, that each pill has many real photos) would ruin every later step, and we did correct one such assumption here.

## What it does, in order

**Part A: ePillID**
1. Loads `all_labels.csv` into a table, one row per image.
2. Prints the shape and counts: reference vs consumer images, number of pill types, number of pill types that have consumer images.
3. Counts consumer images per pill type and draws a histogram.
4. Shows a reference photo and a consumer photo of the same pill side by side.

**Part B: MEDISEG (for the detector)**
5. Finds the `annotations.json` files.
6. Reads the annotations (COCO format) and prints the number of images, boxes and classes.
7. Counts boxes (pills) per image.
8. Draws the boxes on five images to check by eye that they sit on the pills.

## Key findings

- 13,532 labelled images: 9,804 reference and 3,728 consumer.
- 4,902 pill types. Only 960 have consumer photos.
- **Every pill type has at most 5 consumer photos** (average about 3.9).
- Reference photos are clean and controlled. Consumer photos vary in lighting, background and angle. That gap is why training only on reference photos would not be enough.
- MEDISEG: **[fill in]** your own counts of images, boxes per image and classes, and whether the drawn boxes sat tightly on the pills.

## Why this matters

The "at most 5 consumer photos per pill" finding changed our plan. We first intended a normal classifier on 20 to 50 pills with many photos each. That is not possible, so the problem is low-shot, and we use reference photos and a baseline based on matching.

## Inputs and outputs

- Input: `data/raw/epillid/ePillID_data/all_labels.csv` and the images under `classification_data/`; the MEDISEG extract.
- Output: none. Only printed numbers and plots.

## Possible viva questions

- *Why did you not start training immediately?* We inspect data first to avoid building on wrong assumptions, as required in our plan.
- *What is the difference between a reference and a consumer image?* Reference: controlled lab photo. Consumer: real-world photo.
- *Why did you check the MEDISEG boxes by drawing them?* A model trained on shifted or wrong boxes learns the wrong thing. Looking at five images is a quick sanity check, not proof that all boxes are correct.

## Known issues

- Part B depends on the folder layout of the MEDISEG extract, which we confirmed by running it.
- The extra image files in the ePillID zip (about 33,000 files vs 13,532 rows) are unexplained, so we only use images listed in `image_path`.
