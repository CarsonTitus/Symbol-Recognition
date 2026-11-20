# Lesson 36 Checkpoint: Progress Summary

**Team:** Carson Titus, Sabrina McGarvey
**Repository:** https://github.com/CarsonTitus/Symbol-Recognition (tag `lesson36-checkpoint`)

## Completed functionality

We built a CNN (convolutional neural network) that looks at a close-up image and names the shape of the main thing in it: **triangle**, **quadrilateral**, **ellipse**, or **unknown** (anything else, including no shape at all). It fine-tunes a MobileNetV3 pretrained on ImageNet using about 58,000 public hand-drawn and synthetic shape images. After every epoch, the model is scored with our reward rubric: +1 for naming the right shape, +0.5 for "unknown," and 0 for anything wrong.

On validation data, the model reaches **99.4% accuracy (99.6% of the maximum reward)**. The learning-curve experiment trained on 10%, 25%, 50% and 100% of the data and scored 95.8%, 98.6%, 99.3% and 99.3% accuracy, above the ~90% target at every size. The repository has scripts to download and prepare data, train, evaluate (confusion matrix, reward convergence graph, error gallery), run the learning curve, and classify a photo from the terminal (`predict.py`). There are 53 automated tests, including hand-checked reward math.

## Remaining work

- Collect our own Pixel photos (≥300 per class, 5+ drawers, including real objects such as balls and signs). These photos become the only test set.
- Stage B fine-tuning on those photos, and a before/after comparison.
- A never-seen-shapes test (hearts, the letter "A", lightning bolts).
- Choosing the confidence threshold from precision-recall curves.
- Final plots and a write-up of the tactical implications of failures.

## Challenges

- The first version mislabeled some public data: Quick, Draw! "diamonds" are often triangle-shaped gems, and some Kaggle images are near-duplicate video frames. We found these through error galleries and fixed the class mappings.
- We revised the scope from one named shape to three, and to include real objects, because ordinary object detectors (like YOLO) recognize objects but cannot see drawn shapes.
- Validation scores come from public drawings. The honest test, on our own photos, is still ahead.

## Results in this repository

`results/lesson36/` holds the confusion matrix, reward convergence graph, training curves and learning curve (CSV and PNG).
