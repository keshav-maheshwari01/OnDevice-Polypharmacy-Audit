# Notebook 03: Baseline (No Training)

## Purpose

Measure how well a pretrained model does on our task **without any training**. This is the score that any trained model has to beat. Without a baseline, we could not say whether training helped.

## The idea

A pretrained network (MobileNetV3-Small, trained on ImageNet) has already learned what shapes, colours and textures look like. If we remove its last layer, it gives a **fingerprint** for any image: a list of 576 numbers. Images that look alike get similar fingerprints. For each real photo, we find which pill's clean photos look most similar.

## What it does, in order

1. Loads the training and validation lists from Notebook 02.
2. Loads the pretrained MobileNetV3-Small and replaces its classifier with `Identity`, so it outputs the 576-number fingerprint.
3. Sets the preprocessing: resize to 224 x 224, convert to numbers, and normalise with the ImageNet averages.
4. Makes fingerprints for the 80 reference photos (the **gallery**) and the 40 validation photos (the **questions**).
5. Rescales every fingerprint to length 1, so the dot product of two fingerprints is their cosine similarity.
6. Computes a similarity score for every question photo against every gallery photo (a 40 x 80 table).
7. Each pill has two gallery photos (front and back). For each pill, keeps the **higher** of the two scores. The maximum is used because a real photo normally shows only one side.
8. Ranks the 40 pills for each photo and checks whether the true pill is first (top-1) or among the first five (top-5).
9. Lists the wrong photos and shows some of them.

## Results (from our runs)

- Gallery fingerprints: shape [80, 576]. Query fingerprints: shape [40, 576].
- **Top-1: 0.500 (20 of 40 right). Top-5: 0.825 (33 of 40).**
- Chance level for 40 pills: 2.5% top-1 and 12.5% top-5, so the baseline is far above chance.
- Looking at the mistakes, the wrong guesses had the **same shape but a different colour and texture**.
- The baseline has no randomness, so these numbers are the same every time.

## Why these choices

- **Why use reference photos as the gallery?** They are clean photos of exactly the same pills, and in this baseline they are the training-group images.
- **Why validation photos as questions?** They are real photos that were not used to make any decision. The test photos stay locked.
- **Why the maximum, not the average?** Reasoned choice, not tested.
- **Why normalise?** The pretrained model was trained on inputs scaled this way, so we prepare ours the same way.

## Inputs and outputs

- Input: `train.csv`, `val.csv`, the images, and downloaded pretrained weights.
- Output: numbers only (top-1, top-5, the list of wrong photos).

## Possible viva questions

- *What does a baseline tell you?* It sets the number to beat, so we can tell whether training helped.
- *What is an embedding?* A list of numbers describing how an image looks, taken from a network with its last layer removed.
- *What is cosine similarity?* A score of how much two fingerprints point the same way. Higher means more alike.
- *Where is the training data?* The reference photos serve as the gallery. Nothing is learned in this notebook.
- *Why does the model confuse pills?* The wrong guesses we looked at had the same shape but different colour and texture. Our reading is that the pretrained model has not learned to separate such pills, which fine-tuning may fix. We did not test that explanation separately.

## Known issues

- `device` is used without being defined in this notebook. Add `device = "cuda" if torch.cuda.is_available() else "cpu"` near the top.
- A cell uses a hard-coded `wrong[19]`, which breaks if the number of wrong photos changes.
- With 40 validation photos, each photo is worth 2.5 percentage points, so the score is rough.
