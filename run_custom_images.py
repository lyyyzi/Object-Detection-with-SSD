import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils import non_maximum_suppression, decode_box
from model import SSD
from dataset import default_box_generator
import os
import glob

# ------------------------------
# Load model
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 320
class_num = 4                       # cat / dog / person / background
# BGR colors (OpenCV uses BGR!)
colors = {
    0: (255,0, 0),   # cat → green
    1: (0, 255, 0),   # dog → blue
    2: (0, 0, 255),   # person → red

}

# -------- default boxes --------
boxs_default = default_box_generator(
    layers=[10, 5, 3, 1],
    large_scale=[0.2, 0.4, 0.6, 0.8],
    small_scale=[0.1, 0.3, 0.5, 0.7]
)
boxs_default = np.array(boxs_default)

# -------- load network ---------
network = SSD(class_num)
network.load_state_dict(torch.load("network.pth", map_location=device))
network.to(device)
network.eval()

# ------------------------------
# Directory for your 10 images
# ------------------------------
image_paths = sorted(glob.glob("data/custom_data/*"))
os.makedirs("viz_custom", exist_ok=True)

# ------------------------------
# Process each custom image
# ------------------------------
class_names = ["cat", "dog", "person"]  # 0,1,2
num_real_classes = 3

for idx, path in enumerate(image_paths):
    print("Processing:", path)

    # load + convert
    img_bgr = cv2.imread(path)
    orig_h, orig_w = img_bgr.shape[:2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))

    inp = img_resized.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))
    inp = torch.from_numpy(inp).unsqueeze(0).to(device)

    # ------------------------------
    # Forward pass
    # ------------------------------
    with torch.no_grad():
        pred_conf, pred_box = network(inp)

    pred_conf = F.softmax(pred_conf, dim=-1)
    pred_conf_ = pred_conf[0].cpu().numpy()
    pred_box_ = pred_box[0].cpu().numpy()

    # ------------------------------
    # Run your NMS
    # ------------------------------
    pred_conf_nms, decoded_boxes_nms = non_maximum_suppression(
        pred_conf_,
        pred_box_,
        boxs_default,
        overlap=0.5,
        threshold=0.6        # YOU CAN TUNE THIS
    )

    # ------------------------------
    # Draw predictions on the ORIGINAL image
    # ------------------------------
    canvas = img_bgr.copy()

    for c in range(num_real_classes):   # 0,1,2
        scores = pred_conf_nms[:, c]
        best_score = scores.max()

        if best_score < 0.5:
            continue

        best_idx = scores.argmax()

        # NMS-decoded box in normalized coords
        x_min, y_min, x_max, y_max = decoded_boxes_nms[best_idx]

        x1 = int(x_min * orig_w)
        y1 = int(y_min * orig_h)
        x2 = int(x_max * orig_w)
        y2 = int(y_max * orig_h)

        color = colors[c]   # class-specific color

        # draw box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

        # draw label
        cv2.putText(
            canvas,
            f"{class_names[c]} {best_score:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


    # ------------------------------
    # Save result
    # ------------------------------
    out_path = f"viz_custom/custom_{idx:03d}.png"
    cv2.imwrite(out_path, canvas)
    print("Saved:", out_path)
