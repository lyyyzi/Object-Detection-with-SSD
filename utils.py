import numpy as np
import cv2
from dataset import iou
import os

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes
def decode_box(rel_box, default_box, w_img, h_img):
    """
    rel_box:     [tx, ty, tw, th]  (relative to default box)
    default_box: [px, py, pw, ph, x_min, y_min, x_max, y_max]
    w_img, h_img: image width/height in pixels
    returns:     integer pixel coords (x1, y1, x2, y2)
    """
    px, py, pw, ph = default_box[:4]
    tx, ty, tw, th = rel_box

    # SSD-style decoding (same as assignment spec)
    gx = pw * tx + px
    gy = ph * ty + py
    gw = pw * np.exp(tw)
    gh = ph * np.exp(th)

    x1 = int((gx - gw / 2.0) * w_img)
    y1 = int((gy - gh / 2.0) * h_img)
    x2 = int((gx + gw / 2.0) * w_img)
    y2 = int((gy + gh / 2.0) * h_img)

    # clip to image bounds
    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(0, min(w_img - 1, x2))
    y2 = max(0, min(h_img - 1, y2))

    return x1, y1, x2, y2

def visualize_pred(windowname, pred_confidence, pred_box,
                   ann_confidence, ann_box, image_, boxs_default, suffix=None):
    # pred_confidence: [num_boxes, num_classes]
    num_boxes, num_classes = pred_confidence.shape
    class_num = num_classes - 1      # ignore background
    class_names = ["cat", "dog", "person"]  # indices 0,1,2


    # Restore image from [C,H,W] and scale to uint8
    image = np.transpose(image_, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h_img, w_img, _ = image.shape

    image1 = image.copy()   # gt boxes
    image2 = image.copy()   # gt default boxes
    image3 = image.copy()   # predicted boxes
    image4 = image.copy()   # predicted default boxes

    # ---------------- GROUND TRUTH ----------------
    for i in range(num_boxes):
        # which foreground class has highest gt prob (0..2)
        cls = np.argmax(ann_confidence[i, :class_num])
        if ann_confidence[i, cls] <= 0.5:
            continue

        # rel_box: the encoded [tx,ty,tw,th] for this default box
        rel_box = ann_box[i]              # shape (4,)
        default_box = boxs_default[i]     # shape (8,)

        x1, y1, x2, y2 = decode_box(rel_box, default_box, w_img, h_img)
        cv2.rectangle(image1, (x1, y1), (x2, y2), colors[cls], 2)

        # Draw the raw default box on image2
        px, py, pw, ph = default_box[:4]
        dx1 = int((px - pw / 2.0) * w_img)
        dy1 = int((py - ph / 2.0) * h_img)
        dx2 = int((px + pw / 2.0) * w_img)
        dy2 = int((py + ph / 2.0) * h_img)
        cv2.rectangle(image2, (dx1, dy1), (dx2, dy2), colors[cls], 2)

    # ---------------- PREDICTIONS ----------------
    for i in range(num_boxes):
        # predicted best class (ignore background)
        cls = np.argmax(pred_confidence[i, :class_num])
        score = pred_confidence[i, cls]
        if score <= 0.5:
            continue

        rel_box = pred_box[i]            # predicted [tx,ty,tw,th]
        default_box = boxs_default[i]

        x1, y1, x2, y2 = decode_box(rel_box, default_box, w_img, h_img)
        cv2.rectangle(image3, (x1, y1), (x2, y2), colors[cls], 2)

        label_text = f"{class_names[cls]} {score:.2f}"

        cv2.putText(
            image3,
            label_text,                                           # confidence score
            (int(x1),int(max(0, y1 - 5))),             # position (slightly above box)
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colors[cls],                      # same color as box
            2
        )
        # anchor that fired
        px, py, pw, ph = default_box[:4]
        dx1 = int((px - pw / 2.0) * w_img)
        dy1 = int((py - ph / 2.0) * h_img)
        dx2 = int((px + pw / 2.0) * w_img)
        dy2 = int((py + ph / 2.0) * h_img)
        cv2.rectangle(image4, (dx1, dy1), (dx2, dy2), colors[cls], 2)

    # ------------- Combine into 2x2 grid -------------
    h, w, _ = image1.shape
    canvas = np.zeros((h * 2, w * 2, 3), np.uint8)
    canvas[:h, :w] = image1
    canvas[:h, w:] = image2
    canvas[h:, :w] = image3
    canvas[h:, w:] = image4

    os.makedirs("viz", exist_ok=True)
    if suffix is None:
        # no suffix → same name as before
        save_name = f"{windowname}_viz.png"
    else:
        # unique name per call (e.g., train_000_viz.png, train_001_viz.png)
        save_name = f"{windowname}_{suffix:03d}_viz.png"

    save_path = os.path.join("viz", save_name)
    cv2.imwrite(save_path, canvas)
    print("saved ->", save_path)

 

def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    #TODO: non maximum suppression
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    num_boxes, num_classes = confidence_.shape
    num_real_classes = num_classes - 1  # ignore background class

    # ==================================
    # Step 1: Decode relative box to absolute XYXY coordinates
    # ==================================
    decoded_boxes = np.zeros((num_boxes, 4), dtype=np.float32)

    for i in range(num_boxes):
        px, py, pw, ph = boxs_default[i, :4]   # default center + size
        dx, dy, dw, dh = box_[i]               # predicted deltas

        cx = pw * dx + px
        cy = ph * dy + py
        w  = pw * np.exp(dw)
        h  = ph * np.exp(dh)

        decoded_boxes[i, 0] = cx - w / 2   # x_min
        decoded_boxes[i, 1] = cy - h / 2   # y_min
        decoded_boxes[i, 2] = cx + w / 2   # x_max
        decoded_boxes[i, 3] = cy + h / 2   # y_max

    # ==================================
    # Step 2: Apply NMS per-class
    # ==================================
    suppressed_conf = np.zeros_like(confidence_)  # initialize output confidence

    for c in range(num_real_classes):   # run only classes 0,1,2
        class_scores = confidence_[:, c]

        # Select boxes above confidence threshold
        valid_idx = np.where(class_scores > threshold)[0]
        if len(valid_idx) == 0:
            continue

        # Sort by descending score (IMPORTANT)
        order = valid_idx[np.argsort(-class_scores[valid_idx])]

        while len(order) > 0:
            # pick highest-score box
            best_idx = order[0]
            suppressed_conf[best_idx, c] = class_scores[best_idx]

            # if only one candidate left → done
            if len(order) == 1:
                break

            # remaining boxes to compare
            rest_idx = order[1:]

            # IoU computation between best and rest
            bb = decoded_boxes[best_idx]
            cand_boxes = decoded_boxes[rest_idx]

            xx1 = np.maximum(bb[0], cand_boxes[:, 0])
            yy1 = np.maximum(bb[1], cand_boxes[:, 1])
            xx2 = np.minimum(bb[2], cand_boxes[:, 2])
            yy2 = np.minimum(bb[3], cand_boxes[:, 3])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter_area = inter_w * inter_h

            area_best = (bb[2] - bb[0]) * (bb[3] - bb[1])
            area_cand = (cand_boxes[:, 2] - cand_boxes[:, 0]) * \
                        (cand_boxes[:, 3] - cand_boxes[:, 1])
            union = area_best + area_cand - inter_area
            iou = inter_area / np.maximum(union, 1e-8)

            # keep boxes that DO NOT overlap too much
            keep_flags = iou <= overlap

            # update order (this removes best_idx automatically)
            order = rest_idx[keep_flags]

    return suppressed_conf, decoded_boxes





def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    """
    confidence_ : [540, num_classes]   (softmax applied)
    box_        : [540, 4]             (predicted deltas)
    boxs_default: [540, 4]             (default cx, cy, w, h)
    """
    num_boxes, num_classes = confidence_.shape
    num_real_classes = num_classes - 1  # last channel is background

    # ==================================
    # Step 1: Decode predicted box deltas into XYXY (normalized 0–1)
    # ==================================
    decoded_boxes = np.zeros((num_boxes, 4), dtype=np.float32)

    for i in range(num_boxes):
        px, py, pw, ph = boxs_default[i, :4]   # default anchor
        dx, dy, dw, dh = box_[i]               # predicted delta

        cx = px + pw * dx
        cy = py + ph * dy
        w  = pw * np.exp(dw)
        h  = ph * np.exp(dh)

        decoded_boxes[i, 0] = cx - w / 2.0
        decoded_boxes[i, 1] = cy - h / 2.0
        decoded_boxes[i, 2] = cx + w / 2.0
        decoded_boxes[i, 3] = cy + h / 2.0

    # CLIP to valid normalized range (VERY IMPORTANT)
    decoded_boxes = np.clip(decoded_boxes, 0.0, 1.0)

    # ==================================
    # Step 2: Per-class NMS
    # ==================================
    suppressed_conf = np.zeros_like(confidence_)

    for c in range(num_real_classes):  # class 0,1,2
        class_scores = confidence_[:, c]

        # select confident boxes
        valid_idx = np.where(class_scores > threshold)[0]
        if len(valid_idx) == 0:
            continue

        # sort by descending score
        order = valid_idx[np.argsort(-class_scores[valid_idx])]

        while len(order) > 0:
            best_idx = order[0]
            suppressed_conf[best_idx, c] = class_scores[best_idx]

            if len(order) == 1:
                break

            rest = order[1:]
            bb = decoded_boxes[best_idx]
            others = decoded_boxes[rest]

            # IoU calculation
            xx1 = np.maximum(bb[0], others[:, 0])
            yy1 = np.maximum(bb[1], others[:, 1])
            xx2 = np.minimum(bb[2], others[:, 2])
            yy2 = np.minimum(bb[3], others[:, 3])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter_area = inter_w * inter_h

            area_best = (bb[2] - bb[0]) * (bb[3] - bb[1])
            area_other = (others[:, 2] - others[:, 0]) * \
                         (others[:, 3] - others[:, 1])

            union = area_best + area_other - inter_area
            iou = inter_area / np.maximum(union, 1e-6)

            # keep boxes with LOW IoU
            keep = iou <= overlap

            # update list
            order = rest[keep]

    return suppressed_conf, decoded_boxes


import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_mAP(network, dataloader_test, boxs_default, device="cuda",epoch=None):
    """
    Compute mAP on the given dataloader_test using a trained SSD network and
    the default boxes passed from main.py.

    Called from main.py as:
        mAP = generate_mAP(network, dataloader_test, boxs_default, device="cuda")
    """
    DEVICE = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    network.to(DEVICE)
    network.eval()

    NUM_CLASSES = 3         # 0: cat, 1: dog, 2: person (background is class 3)
    IOU_THRESH  = 0.5
    CONF_THRESH = 0.5

    # Stats containers
    total_gt    = {c: 0 for c in range(NUM_CLASSES)}   # ground-truth count per class
    all_scores  = {c: [] for c in range(NUM_CLASSES)}  # prediction scores
    all_matches = {c: [] for c in range(NUM_CLASSES)}  # 1 = TP, 0 = FP

    # --- IoU helper for [x1,y1,x2,y2] boxes ---
    def iou_xyxy(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter   = inter_w * inter_h

        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = area1 + area2 - inter + 1e-8

        return inter / union

    print("\n[generate_mAP] start evaluation...\n")

    with torch.no_grad():
        for images, ann_box, ann_conf in dataloader_test:
            images = images.to(DEVICE)                  # [B, 3, 320, 320]
            ann_box_np  = ann_box.numpy()               # [B, 540, 4] (tx,ty,tw,th w.r.t default)
            ann_conf_np = ann_conf.numpy()              # [B, 540, 4] one-hot

            pred_conf, pred_box = network(images)       # [B, 540, 4], [B, 540, 4]
            pred_conf = torch.softmax(pred_conf, dim=-1).cpu().numpy()
            pred_box  = pred_box.cpu().numpy()

            B = images.shape[0]
            for b in range(B):
                pc  = pred_conf[b]      # [540, 4]
                pb  = pred_box[b]       # [540, 4]
                gt_b = ann_box_np[b]    # [540, 4]
                gt_c = ann_conf_np[b]   # [540, 4]

                # ---------- count GT per class ----------
                # for i in range(gt_c.shape[0]):
                #     cls = np.argmax(gt_c[i])
                #     if gt_c[i, cls] > 0.5 and cls < NUM_CLASSES:   # ignore background=3
                #         total_gt[cls] += 1
                gt_labels = np.argmax(gt_c[:, :NUM_CLASSES], axis=1)

                for cls in gt_labels:
                    total_gt[cls] += 1

                # ---------- decode predicted boxes ----------
                num_boxes = pb.shape[0]
                decoded_boxes = np.zeros((num_boxes, 4), dtype=np.float32)
                for i in range(num_boxes):
                    px, py, pw, ph = boxs_default[i, :4]  # default center+size
                    dx, dy, dw, dh = pb[i]                # relative offsets

                    cx = pw * dx + px
                    cy = ph * dy + py
                    w  = pw * np.exp(dw)
                    h  = ph * np.exp(dh)
                    decoded_boxes[i] = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

                # ---------- per-class prediction matching ----------
                for c in range(NUM_CLASSES):
                    cls_scores = pc[:, c]
                    sel = np.where(cls_scores >= CONF_THRESH)[0]

                    for idx in sel:
                        score   = cls_scores[idx]
                        pred_bb = decoded_boxes[idx]
                        matched = 0

                        # check against all GT boxes of same class
                        for j in range(gt_c.shape[0]):
                            if np.argmax(gt_c[j]) != c:
                                continue

                            # gx, gy, gw, gh = gt_b[j]
                            # gt_box_xyxy = [
                            #     gx - gw / 2.0,
                            #     gy - gh / 2.0,
                            #     gx + gw / 2.0,
                            #     gy + gh / 2.0,
                            # ]
                            
                            # decode GT box like prediction box
                            px, py, pw, ph = boxs_default[j, :4]
                            dx, dy, dw, dh = gt_b[j]

                            # decode center + size
                            cx = pw * dx + px
                            cy = ph * dy + py
                            w  = pw * np.exp(dw)
                            h  = ph * np.exp(dh)

                            # convert to xyxy
                            gt_box_xyxy = [
                                cx - w / 2.0,
                                cy - h / 2.0,
                                cx + w / 2.0,
                                cy + h / 2.0,
                            ]

                            if iou_xyxy(pred_bb, gt_box_xyxy) >= IOU_THRESH:
                                matched = 1
                                break

                        all_scores[c].append(score)
                        all_matches[c].append(matched)

    # ---------- compute AP & mAP ----------
    def compute_AP(recalls, precisions):
        # 11-point interpolated AP
        return np.mean([
            np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
            for t in np.linspace(0, 1, 11)
        ])

    APs = []
    for c in range(NUM_CLASSES):
        scores  = np.array(all_scores[c])
        matches = np.array(all_matches[c])

        if len(scores) == 0:
            APs.append(0.0)
            continue

        order   = np.argsort(-scores)
        matches = matches[order]

        TP = np.cumsum(matches)
        FP = np.cumsum(1 - matches)

        if total_gt[c] > 0:
            recalls = TP / float(total_gt[c])
        else:
            recalls = np.zeros_like(TP)

        precisions = TP / np.maximum(TP + FP, 1e-8)

        AP = compute_AP(recalls, precisions)
        APs.append(AP)

        plt.plot(recalls, precisions, label=f"Class {c} (AP={AP:.3f})")

    mAP = float(np.mean(APs)) if len(APs) > 0 else 0.0

    plt.title(f"Precision–Recall (mAP={mAP:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.legend()
    import os
    os.makedirs("pr_curves", exist_ok=True)
    
    if epoch is None:
      filename = "pr_curve.png"
    else:
        filename = f"pr_curve_epoch_{epoch:03d}.png"

    plt.savefig(os.path.join("pr_curves", filename))
    plt.close()

    print("\n===== Evaluation results =====")
    for c, ap in enumerate(APs):
        print(f"AP for class {c}: {ap:.4f}")
    print(f"Mean AP (mAP): {mAP:.4f}\n")

    return mAP









