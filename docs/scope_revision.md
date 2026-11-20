# Revised scope (draft for the CS471 scoping worksheet)

Revised 2026-11-20. Only **States**, **Actions** and **Reward** change; everything else is as submitted.

## Why we revised it

Our first scope recognized one shape (a hand-drawn triangle) and called everything else "unknown." While building it we saw two problems:

1. **One named shape is too narrow a proof of concept.** Real symbology has many distinct shapes. Naming three basic shapes (triangle, quadrilateral, ellipse) shows that the approach generalizes.
2. **The capstone's real gap is objects vs. drawings.** The app's YOLO detector recognizes objects but not drawn shapes. Our classifier names the **outline of the main thing in a close-up**, whether it is drawn, painted, printed or a physical object (a ball is an ellipse, a window is a quadrilateral, a yield sign is a triangle).

We chose "quadrilateral" and "ellipse" instead of "square" and "circle" because nobody hand-draws a perfect square or circle, and real objects seen at an angle rarely look like one.

## Revised fields

| Field | Revised answer |
|---|---|
| **States** | Each image (a close-up of a drawn mark or of a real-world object) is a state. |
| **Actions** | The agent labels each image as one of five outputs: **"no symbol"**, **"symbol detected but cannot identify"**, **"symbol detected: triangle"**, **"symbol detected: quadrilateral"**, or **"symbol detected: ellipse"**. |
| **Reward** | Incorrect (including naming the wrong shape) → **+0**. Symbol detected but can't identify → **+0.5**. Symbol detected and correctly identified → **+1**. Correctly says no symbol → **+1**. |

## Unchanged fields

| Field | Answer |
|---|---|
| **Operational problem** | Alert operators to symbology (e.g., gang signs) in their surroundings. Basic shapes are the proof of concept. |
| **Algorithm** | CNN (convolutional neural network) |
| **Lesson 36 checkpoint** | Training output and the trained model recognizing symbols correctly ~90% of the time across different dataset sizes. |
| **Lesson 41 evaluation** | Confusion matrix and reward convergence graph, showing 90% accuracy on test data. |
