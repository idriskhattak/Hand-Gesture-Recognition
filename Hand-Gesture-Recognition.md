# Hand Gesture Recognition

Real-time hand-sign recognition from a webcam: MediaPipe finds the hand, a Random
Forest classifies the landmarks, OpenCV draws the result.

📄 Full write-up: [idriskhattak.github.io/Idris_Portfolio/projects/hand-gesture-recognition/](https://idriskhattak.github.io/Idris_Portfolio/projects/hand-gesture-recognition/)

## How it works

Recognising a sign from a live feed needs more than a classifier. The pipeline has to
find the hand, represent it consistently as it moves around the frame, and return a
label fast enough to feel live.

1. **Detect** — MediaPipe Hands returns 21 landmarks for one hand.
2. **Normalise** — x and y coordinates are shifted by subtracting the minimum
   coordinate, giving 42 features that do not depend on where in the frame the hand
   appears. This is the step that makes the classifier work at all; raw pixel
   coordinates would encode hand position rather than hand shape.
3. **Classify** — a scikit-learn Random Forest over those 42 features.
4. **Draw** — OpenCV renders the hand skeleton and the predicted label above it.

Recognises 8 signs, including Hello, Yes, No, OK and Good Luck.

## Running it

```bash
pip install opencv-python mediapipe scikit-learn numpy
cd Sign-Language-Detection
python inference_classifier.py     # live webcam inference
```

To retrain on your own gestures, work through `Main.ipynb` — it covers collection,
landmark extraction, training and evaluation.

## Results

| | |
| --- | --- |
| Input representation | 42 normalised x/y landmark features |
| Classifier | scikit-learn Random Forest |
| Classes | 8 sign labels |
| Test split | 20%, stratified and shuffled |
| Reported accuracy | 100.00% |

**Read that 100% carefully.** It is one random train/test split of a single
collection session. A random split is optimistic when frames of the same gesture,
recorded seconds apart in the same lighting and the same hand position, land in both
the training and test sets — the classifier can be recognising the session rather
than the sign. There is no cross-validation, no separate test recording, no confusion
matrix, and no measured frame rate.

## Known limitations

- Assumes exactly one visible hand.
- Fixed label mapping — every detected hand is forced into one of the eight classes.
  There is no "unknown" or low-confidence state.
- Sensitive to lighting, background, pose and hand orientation, none of which are
  measured.
- No FPS or end-to-end latency figures.

## What I would do differently

Collect separate recordings for training and testing so the split cannot leak.
Report a per-sign confusion matrix — some of these signs are visually close and the
aggregate number hides which ones get confused. Measure FPS and latency on the
target machine. Add a confidence threshold with an explicit unknown state.
