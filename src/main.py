import argparse
import os
import numpy as np
import time
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from dataset import COCO, default_box_generator
from model import SSD, SSD_loss
from utils import visualize_pred, non_maximum_suppression, generate_mAP

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 100
batch_size = 32


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True
import torch

# device = torch.device("cpu")   # <-- FORCE CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# network = SSD(class_num)
# network.to(device)


if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)
    # # ===== DEBUG: visualize several GT samples before training =====
    # from utils import decode_box
    # import cv2, os
    # import numpy as np
    # import sys

    # os.makedirs("gt_debug", exist_ok=True)

    # num_debug = 100

    # for idx, (images_, ann_box_, ann_conf_) in enumerate(dataloader):
    #     if idx >= num_debug:
    #         break

    #     image = images_[0].numpy()          # [C, H, W]
    #     ann_box = ann_box_[0].numpy()       # [N, 4]
    #     ann_conf = ann_conf_[0].numpy()     # [N, C]
    #     H, W = image.shape[1], image.shape[2]

    #     # convert to BGR for OpenCV
    #     img = np.transpose(image, (1, 2, 0))
    #     img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #     for i in range(ann_box.shape[0]):
    #         cls = np.argmax(ann_conf[i, :-1])      # best FG class (0,1,2)
    #         if ann_conf[i, cls] < 0.5:             # skip background/low prob
    #             continue

    #         default_box = boxs_default[i]          # [cx,cy,w,h,...]
    #         rel_box = ann_box[i]                   # [tx,ty,tw,th]

    #         x1, y1, x2, y2 = decode_box(rel_box, default_box, W, H)
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(img, f"GT-{cls}", (x1, max(0, y1 - 5)),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #     save_path = f"gt_debug/sample_{idx:02d}.png"
    #     cv2.imwrite(save_path, img)
    #     print("saved ->", save_path)

    # # stop after writing debug images so training doesn’t run yet
    # sys.exit(0)
    # # ===== END DEBUG =====

    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()
    os.makedirs("debug_gt", exist_ok=True)

    for k, data in enumerate(dataloader):
      if k >= 50:      # how many samples you want to visualize
          break

      images_, ann_box_, ann_confidence_ = data
    #   # visualize the first image in this batch
    #   visualize_gt_debug(
    #       images_[0],
    #       ann_box_[0],
    #       ann_confidence_[0],
    #       boxs_default,
    #       name=f"sample_gt_{k}"
    #   )
    # exit()  # STOP PROGRAM HERE JUST FOR DEBUG

    for epoch in range(num_epochs):
        #TRAINING
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        network.train()
        
        train_start = time.time()
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1

        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, suffix=epoch)
        
        
        #VALIDATION
        val_start = time.time()       
        network.eval()
       
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
        
        train_time = time.time() - train_start

        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred("val", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, suffix=epoch)
        
        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)
        if (epoch + 1) % 5 == 0:
            print("\n========== Computing mAP ==========\n")
            mAP = generate_mAP(network, dataloader_test, boxs_default, device="cuda", epoch=epoch+1)
            print(f"[Epoch {epoch+1}] mAP = {mAP:.4f}")
        
        
        val_time = time.time() - val_start
        print(f"[Epoch {epoch+1}] train time: {train_time:.1f}s | val time: {val_time:.1f}s")

        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'network.pth')

else:
    #TEST
    dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()
    
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("viz", exist_ok=True)

    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_ = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)
        pred_confidence = F.softmax(pred_confidence, dim=-1)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        
        pred_confidence_nms, decoded_boxes_nms = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default, overlap=0.5, threshold=0.8)
        
        # keep = np.max(pred_confidence_nms[:, :3], axis=1) > 0
        # filtered_boxes = decoded_boxes_nms[keep]
        # filtered_scores = pred_confidence_nms[keep]
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        img_name = dataset_test.img_names[i]               
        out_name = os.path.splitext(img_name)[0] + ".txt"  
        out_path = os.path.join("predictions", out_name)
        # with open(out_path, "w") as f:
        #             for box, scores in zip(filtered_boxes, filtered_scores):
        #                 for c in range(3):  # real classes: 0=cat,1=dog,2=person
        #                     if scores[c] > 0:
        #                         x_min, y_min, x_max, y_max = box
        #                         f.write(f"{c} {scores[c]:.4f} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}\n")
        with open(out_path, "w") as f:
            num_boxes, num_classes = pred_confidence_nms.shape
            real_classes = num_classes - 1   # ignore background
            # for b in range(decoded_boxes_nms.shape[0]):

            #     cls = np.argmax(pred_confidence_nms[b, :3])   # best class
            #     score = pred_confidence_nms[b, cls]

            #     if score < 0.5:       
            #         continue
                
            #     x_min, y_min, x_max, y_max = decoded_boxes_nms[b]

            #     f.write(f"{cls} {score:.4f} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}\n")

            for b in range(num_boxes):
                for c in range(real_classes):   # 0:cat, 1:dog, 2:person
                    score = pred_confidence_nms[b, c]
                    if score <= 0.0:
                        continue  # suppressed / not confident enough

                    # decode predicted box for this default box b
                    px, py, pw, ph = boxs_default[b, :4]   # default box center + size
                    dx, dy, dw, dh = pred_box_[b]         # predicted offsets

                    cx = pw * dx + px
                    cy = ph * dy + py
                    w_box = pw * np.exp(dw)
                    h_box = ph * np.exp(dh)

                    x_min = cx - w_box / 2.0
                    y_min = cy - h_box / 2.0
                    x_max = cx + w_box / 2.0
                    y_max = cy + h_box / 2.0

                    # save: class_id, score, x_min, y_min, x_max, y_max (all normalized [0,1])
                    f.write(f"{c} {score:.4f} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}\n")
            
        viz_name = f"test_{i:04d}"

        visualize_pred(viz_name, pred_confidence_nms, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        # visualize_pred(
        #         viz_name,
        #         filtered_scores,
        #         filtered_boxes,
        #         ann_confidence_[0].numpy(),
        #         ann_box_[0].numpy(),
        #         images_[0].numpy(),
        #         boxs_default=None
        #     )
        cv2.waitKey(1000)
# else:
# else:
#     #TEST
#     dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train = False, image_size=320)
#     dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
#     network.load_state_dict(torch.load('network.pth'))
#     network.eval()
    
#     os.makedirs("predictions", exist_ok=True)
#     os.makedirs("viz", exist_ok=True)

#     for i, data in enumerate(dataloader_test, 0):
#         images_, ann_box_, ann_confidence_ = data
#         images = images_.cuda()
#         ann_box = ann_box_.cuda()
#         ann_confidence = ann_confidence_.cuda()

#         pred_confidence, pred_box = network(images)
#         pred_confidence = F.softmax(pred_confidence, dim=-1)
#         print("========== SHAPE DEBUG ==========")
#         print("pred_confidence:", pred_confidence.shape)
#         print("pred_box:       ", pred_box.shape)
#         print("ann_confidence: ", ann_confidence.shape)
#         print("ann_box:        ", ann_box.shape)
#         print("num_default:    ", len(boxs_default))
#         print("=================================\n")

#         pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
#         pred_box_ = pred_box[0].detach().cpu().numpy()
        
#         pred_confidence_nms, decoded_boxes_nms = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default, overlap=0.5, threshold=0.8)
        
#         # ----- keep only the best box per class (for nicer visualization) -----
#         num_boxes, num_classes = pred_confidence_nms.shape
#         num_real_classes = num_classes - 1  # 0,1,2 = cat,dog,person

#         filtered_conf = np.zeros_like(pred_confidence_nms)

#         for c in range(num_real_classes):
#             column = pred_confidence_nms[:, c]      # scores for class c
#             if column.max() <= 0.0:
#                 continue                            # nothing survived NMS for this class
#             best_idx = column.argmax()              # index of best box for this class
#             filtered_conf[best_idx, c] = column[best_idx]

#         # from now on, use filtered_conf instead of pred_confidence_nms
#         pred_conf_to_use = filtered_conf

#         #TODO: save predicted bounding boxes and classes to a txt file.
#         #you will need to submit those files for grading this assignment
        
#         img_name = dataset_test.img_names[i]               
#         out_name = os.path.splitext(img_name)[0] + ".txt"  
#         out_path = os.path.join("predictions", out_name)

#         # with open(out_path, "w") as f:
#         #     num_boxes, num_classes = pred_confidence_nms.shape
#         #     real_classes = num_classes - 1   # ignore background
#         #     for b in range(decoded_boxes_nms.shape[0]):

#         #         cls = np.argmax(pred_confidence_nms[b, :3])   # best class
#         #         score = pred_confidence_nms[b, cls]

#         #         if score < 0.5:       
#         #             continue
                
#         #         x_min, y_min, x_max, y_max = decoded_boxes_nms[b]

#         #         f.write(f"{cls} {score:.4f} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}\n")

#         with open(out_path, "w") as f:
#             num_boxes, num_classes = pred_conf_to_use.shape
#             real_classes = num_classes - 1

#             for b in range(num_boxes):
#                 cls = np.argmax(pred_conf_to_use[b, :3])
#                 score = pred_conf_to_use[b, cls]

#                 if score <= 0.5:      # your display threshold
#                     continue

#                 x_min, y_min, x_max, y_max = decoded_boxes_nms[b]
#                 f.write(f"{cls} {score:.4f} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}\n")


            
#         viz_name = f"test_{i:04d}"

#         visualize_pred(viz_name, pred_conf_to_use,  pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
#         cv2.waitKey(1000)        
        # visualize_pred(
        #         viz_name,
        #         filtered_scores,
        #         filtered_boxes,
        #         ann_confidence_[0].numpy(),
        #         ann_box_[0].numpy(),
        #         images_[0].numpy(),
        #         boxs_default=None
        #     )

# else:
#     # TEST
#     dataset_test = COCO("data/test/images/", "data/test/annotations/",
#                         class_num, boxs_default, train=False, image_size=320)
#     dataloader_test = torch.utils.data.DataLoader(
#         dataset_test, batch_size=1, shuffle=False, num_workers=0
#     )

#     # load weights onto CPU
#     network.load_state_dict(torch.load('network.pth', map_location=device))
#     network.to(device)
#     network.eval()
    
#     os.makedirs("predictions", exist_ok=True)
#     os.makedirs("viz", exist_ok=True)

#     with torch.no_grad():
#         for i, data in enumerate(dataloader_test, 0):
#             images_, ann_box_, ann_confidence_ = data

#             # move tensors to CPU device
#             images = images_.to(device)
#             ann_box = ann_box_.to(device)
#             ann_confidence = ann_confidence_.to(device)

#             pred_confidence, pred_box = network(images)
#             pred_confidence = F.softmax(pred_confidence, dim=-1)

#             pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
#             pred_box_ = pred_box[0].detach().cpu().numpy()
        
#             pred_confidence_nms = non_maximum_suppression(
#                 pred_confidence_, pred_box_, boxs_default,
#                 overlap=0.5, threshold=0.65
#             )
        
#             # save predicted bounding boxes and classes to txt
#             img_name = dataset_test.img_names[i]               # e.g. "000123.jpg"
#             out_name = os.path.splitext(img_name)[0] + ".txt"  # -> "000123.txt"
#             out_path = os.path.join("predictions", out_name)

#             with open(out_path, "w") as f:
#                 num_boxes, num_classes = pred_confidence_nms.shape
#                 real_classes = num_classes - 1   # ignore background

#                 for b in range(num_boxes):
#                     for c in range(real_classes):   # 0:cat, 1:dog, 2:person
#                         score = pred_confidence_nms[b, c]
#                         if score <= 0.0:
#                             continue  # suppressed / not confident enough

#                         # decode predicted box for this default box b
#                         px, py, pw, ph = boxs_default[b, :4]   # default box center + size
#                         dx, dy, dw, dh = pred_box_[b]         # predicted offsets

#                         cx = pw * dx + px
#                         cy = ph * dy + py
#                         w_box = pw * np.exp(dw)
#                         h_box = ph * np.exp(dh)

#                         x_min = cx - w_box / 2.0
#                         y_min = cy - h_box / 2.0
#                         x_max = cx + w_box / 2.0
#                         y_max = cy + h_box / 2.0

#                         # save: class_id, score, x_min, y_min, x_max, y_max (all normalized [0,1])
#                         f.write(f"{c} {score:.4f} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}\n")

#             viz_name = f"test_{i:04d}"
#             visualize_pred(
#                 viz_name,
#                 pred_confidence_nms,
#                 pred_box_,
#                 ann_confidence_[0].numpy(),
#                 ann_box_[0].numpy(),
#                 images_[0].cpu().numpy(),   # back to numpy on CPU
#                 boxs_default
#             )
#             cv2.waitKey(1000)



