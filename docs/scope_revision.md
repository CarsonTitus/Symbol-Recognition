# Revised scope (draft for the CS471 scoping worksheet)

Revised 2026-11-20. Only **States**, **Actions** and **Reward** change; everything else is as submitted.

## Why we revised it

Our first scope recognized one shape (a hand-drawn triangle) and called everything else "unknown." While building it we saw two problems:

1. **One named shape is too narrow a proof of concept.** Real symbology has many distinct shapes. Naming three basic shapes (triangle, quadrilateral, ellipse) shows that the approach generalizes.
2. **The capstone's real gap is objects vs. drawings.** The app's YOLO detector recognizes objects but not drawn shapes. Our classifier names the **outline of the main thing in a close-up**, whether it is drawn, painted, printed or a physical object (a ball is an ellipse, a window is a quadrilateral, a yield sign is a triangle).

3. **"No symbol" and "unknown" don't need to be separate.** For an operator, both mean "nothing I can name here," and keeping them apart added labeling work the assignment doesn't need. So anything that is not one of the three shapes, including a blank card, is "unknown / cannot identify."

We chose "quadrilateral" and "ellipse" instead of "square" and "circle" because nobody hand-draws a perfect square or circle, and real objects seen at an angle rarely look like one.

## Revised fields

| Field | Revised answer |
|---|---|
| **States** | Each image (a close-up of a drawn mark or of a real-world object) is a state. |
| **Actions** | The agent labels each image as one of four outputs: **"symbol detected: triangle"**, **"symbol detected: quadrilateral"**, **"symbol detected: ellipse"**, or **"unknown / cannot identify"** (anything else, including images with no symbol at all). |
| **Reward** | Incorrect (including naming the wrong shape, or naming a shape when there is none) → **+0**. Unknown / cannot identify → **+0.5**. Shape detected and correctly identified → **+1**. |

## Unchanged fields

| Field | Answer |
|---|---|
| **Operational problem** | Alert operators to symbology (e.g., gang signs) in their surroundings. Basic shapes are the proof of concept. |
| **Algorithm** | CNN (convolutional neural network) |
| **Lesson 36 checkpoint** | Training output and the trained model recognizing symbols correctly ~90% of the time across different dataset sizes. |
| **Lesson 41 evaluation** | Confusion matrix and reward convergence graph, showing 90% accuracy on test data. |
