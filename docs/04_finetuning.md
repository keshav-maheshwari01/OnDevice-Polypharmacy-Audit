# Notebook 04: Fine-Tuning the Classifier

## Purpose

Train MobileNetV3-Small on our 40 pills, check that it beats the baseline, measure how much results vary from run to run, and look at what it gets wrong.

## What it does, in order

1. Defines a `PillDataset` that reads a CSV list and returns one image (as numbers) plus its label (the pill's number, 0 to 39).
2. Defines the training augmentation: random crop (70 to 100% of the image), rotation up to 30 degrees, small brightness and contrast changes. **No hue changes**, because colour helps tell these pills apart. Validation uses plain preprocessing with no randomness.
3. Builds data loaders that hand the model 16 images at a time (shuffled for training, not for validation).
4. Loads the pretrained MobileNetV3-Small and replaces the last layer with a new one of **40 outputs** (1,024 inputs). We checked its shape: weights [40, 1024] and biases [40].
5. Trains for 20 epochs with the AdamW optimiser (learning rate 3e-4) and cross-entropy loss. After every epoch, evaluates top-1 and top-5 on the validation photos.
6. Saves the best model to disk (see "Saving" below).
7. Repeats training with five different seeds to measure the run-to-run spread.
8. Reloads the saved model, collects its guesses, and shows the wrong photos.

## How the model learns (viva explanation)

- The model receives one photo and outputs **40 scores**, one per pill. The highest score is its guess.
- Each pill's score has its own set of **1,024 weights plus one bias**. These weights are the same for every photo. The number of outputs equals the number of pills.
- During training, the 40 scores are compared with the **label** (the correct pill number), not with other photos. The loss measures how wrong the model was, and the optimiser adjusts the weights.
- **All weights are trained.** The early layers start from the pretrained weights and are nudged slightly. The new last layer starts from random numbers and is learned from scratch.
- The reference photos are only training examples now. When the model answers, it does not compare with any stored photo.
- Device used: **cuda** (the RTX 3050), about 1 second per epoch for 188 training images.

## Results (from our runs)

Validation, 40 photos.

| Method | Top-1 | Top-5 |
|---|---|---|
| Baseline (Notebook 03) | 0.500 | 0.825 |
| One run, epoch 13 (best top-1 of that run) | 0.750 | 0.975 |
| One run, last epoch | 0.725 | 0.950 |
| **Five seeds, last epoch, mean +/- std** | **0.735 +/- 0.037** | **0.945 +/- 0.029** |

Per-seed last-epoch top-1: 0.725, 0.800, 0.700, 0.750, 0.700.

**Which number to report:** the five-seed last-epoch average. Picking the best epoch uses the validation photos, so those numbers flatter the model.

The gain over the baseline (about 0.24 top-1 and 0.12 top-5) is much larger than the run-to-run spread, so we believe the improvement is real. Five runs and 40 validation photos give only a rough estimate.

## What the seed is

The seed fixes the random numbers used in training: the new last layer's starting weights, the augmentation of each image, and the photo order. Different seeds give different runs of the same experiment, which lets us measure the noise. The seed does not change the pretrained weights, the data or the split.

## Saving the model

The training cell saves the model only when the validation top-1 beats the best score already stored on disk (kept in `best_score.json`), and it first copies the current model to `backup_before_run.pt`. Without this guard, every run overwrote the file and the earlier 5-wrong model was lost. The saved model is chosen using validation, so its validation score is optimistic. The honest final number will come from the locked test photos, once.

## Mistake analysis

**[fill in]** What the wrong photos looked like for the trained model (for example: same shape, different colour? Hard-to-read photos?). Baseline mistakes had the same shape but a different colour and texture.

## Inputs and outputs

- Input: `train.csv`, `val.csv`, the images, pretrained weights.
- Output: `../models/best_classifier.pt`, `best_score.json`, and printed scores. The test set is not used.

## Possible viva questions

- *What is overfitting, and how did you watch for it?* The model memorises its training images. Training loss kept falling while validation score flattened after about epoch 11, which is what we expect with 188 training photos. We watch the validation score.
- *Why 1,024 weights per pill?* The model design fixes it: the layer before the last one outputs 1,024 numbers, so each pill's weight set has 1,024 entries.
- *Why a learning rate of 3e-4?* A common starting value. We did not tune it.
- *Why five seeds?* To measure the noise and report a mean and spread instead of one lucky run.
- *Why last epoch, not best?* Choosing the best epoch uses the validation photos, which makes the score optimistic.
- *Did you use the test set?* No. It is locked for the end.
- *Could freezing the early layers be better?* Possibly. We have not tested it.

## Known issues

- The last cell (`for i in wrong:`) has no body and gives a syntax error.
- The seed-saving lines (saving each seed's model under its own file name) are not in `run_training` yet.
- The saved five-seed output showed only seed 0. Re-run the full loop and use only a complete output with all five seeds.
- The first training run has no seed, so it cannot be repeated exactly.
- Re-running only the training cell continues training the same model. Always run the model-building cell together with the training cell.
