SSD Assignment – README
=======================

1. Overview
-----------

This project implements a simplified Single Shot Detector (SSD) for
detecting three foreground classes:

- class 0: cat
- class 1: dog
- class 2: person
- class 3: background (implicit, not predicted as a foreground box)

The code trains the SSD model on the provided COCO-style dataset,
evaluates mAP on the validation set, saves predictions on the test set,
and runs detection on 10 custom images downloaded from the internet.

How to run the code
----------------------

(1) Training + validation + test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main entry point is `main.py`. It is written so that running

    python main.py

will:

- load the training and validation datasets
- train the SSD model for the configured number of epochs
- evaluate mAP on the validation set and plot the PR curves
- evaluate on the test set and write predictions into `predictions/`
- optionally create visualizations in `viz/`

Details of the training loop (batch size, learning rate, epochs, etc.)
are set inside `main.py` and can be changed there.


(2) Test result for training: 

python main.py -test


(3) Running on custom images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After training, make sure `network.pth` (the saved model checkpoint)
is present in the project root. Then run:

    python run_custom_images.py

This script:

- loads `network.pth`
- reads all images from `data/custom_data/`
- runs the SSD model on each image
- draws bounding boxes, class labels, and confidence scores
- writes results to `viz_custom/custom_XXX.png`

2. Directory structure
----------------------

The expected directory layout is:

project_root/
├─ main.py                 # training / validation / test script
├─ model.py                # SSD network definition
├─ dataset.py              # COCO dataset + default_box_generator
├─ utils.py                # loss, NMS, decode_box, visualization helpers
├─ run_custom_images.py    # runs the trained model on custom images
├─ run.sh                  # convenience script to run the project
├─ README.txt
├─ requirements.txt
└─ data/
   ├─ train/
   │   ├─ images/          # training images
   │   └─ annotations/     # txt annotations: class x y w h
   ├─ val/
   │   ├─ images/
   │   └─ annotations/
   ├─ test/
   │   ├─ images/
   │   └─ annotations/     # (if provided) or dummy files
   └─ custom_data/         # 10 custom images from the internet

Runtime output folders (created automatically if missing):

- predictions/     : txt files with test predictions
- viz/             : 2x2 visualizations of GT and predictions
- viz_custom/      : visualizations for custom_data images




3. Important implementation details
-----------------------------------

- Ground-truth annotations are read from txt files with format:

      <class_id> <x> <y> <w> <h>

  where `(x, y)` is the top-left corner and `(w, h)` is the width and
  height in original pixel coordinates.

- Images are resized to 320×320 before feeding them into the network.
  Ground-truth boxes are scaled to match the resized resolution and
  then normalized to [0,1] before matching with default boxes.

- `default_box_generator()` (in `dataset.py`) builds the anchor
  boxes for each feature map layer given:
  `layers=[10, 5, 3, 1]`, `large_scale=[0.2, 0.4, 0.6, 0.8]`,
  `small_scale=[0.1, 0.3, 0.5, 0.7]`.

- `SSD_loss()` combines classification loss and localization loss, and
  uses per-class weights to handle class imbalance if desired.

- `non_maximum_suppression()` (in `utils.py`) runs NMS per class on
  the decoded XYXY boxes and returns:
  - `suppressed_conf`: confidence matrix with suppressed boxes set to 0
  - `decoded_boxes`: decoded boxes (normalized coordinates)

- `visualize_pred()` draws a 2×2 grid showing:
  - top-left: ground-truth decoded boxes
  - top-right: default boxes that matched GT
  - bottom-left: predicted boxes with labels and scores
  - bottom-right: anchors that fired (default boxes for predictions)


6. Notes
--------
- Predicted bounding boxes and classes are in prediction file.
- The code assumes GPU is available.
- Using cpu to test need to uncomment the code at the bottom of the main.py, and uncomment the cuda lines.
- Paths inside `main.py` and `run_custom_images.py` may need to be
  updated if your dataset folder names are different.
- mAP and PR curves are computed for the three foreground classes
  (cat/dog/person) only; background is ignored for evaluation.
- For the custom images in `data/custom_data/`, a confidence threshold
  of around 0.8 gave visually best results for this trained model.
