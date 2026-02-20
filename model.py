import os
import random
import numpy as np

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




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box, neg_weight=3.0):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    Bp, Np, Cp = pred_confidence.shape
    Bg, Ng, Cg = ann_confidence.shape
    
    assert Bp == Bg, "Batch size mismatch between predictions and GT"
    assert Cp == Cg, "Class count mismatch between predictions and GT"

    N = min(Np, Ng)
    
    if Np != Ng:
      print(f"[SSD_loss] WARNING: pred boxes = {Np}, GT boxes = {Ng} -> using N = {N}")

    pred_confidence = pred_confidence[:, :N, :]   # [B, N, C]
    pred_box        = pred_box[:, :N, :]         # [B, N, 4]
    ann_confidence  = ann_confidence[:, :N, :]   # [B, N, C]
    ann_box         = ann_box[:, :N, :]         # [B, N, 4]

    B, N, C = pred_confidence.shape   # now N is consistent

    # Use the common number of boxes (GT has 540, preds currently 552)


    # -----------------------------------------------
    # 1 FLATTEN EVERYTHING
    # -----------------------------------------------
    pred_conf_flat = pred_confidence.reshape(B * N, C)      # [B*N, 4]
    pred_box_flat  = pred_box.reshape(B * N, 4)             # [B*N, 4]

    ann_conf_flat  = ann_confidence.reshape(B * N, C)       # [B*N, 4]
    ann_box_flat   = ann_box.reshape(B * N, 4)

    # -----------------------------------------------
    # 2 GET POSITIVE ANCHOR MASK
    # ann_conf is ONE-HOT so background = index 3
    # -----------------------------------------------
    # positive if NOT background
    gt_idx = ann_conf_flat.argmax(dim=-1)                # [B*N]
    bg_id  = C - 1   
    
    pos_mask = gt_idx != bg_id                # foreground boxes
    neg_mask = ~pos_mask

    num_pos = pos_mask.sum().clamp(min=1)     # prevent div by zero
    num_neg = neg_mask.sum().clamp(min=1)

    # -----------------------------------------------
    # 3 CLASSIFICATION LOSS (CrossEntropy)
    # -----------------------------------------------

    # convert one-hot target class index
    #target_labels = ann_conf_flat.argmax(dim=1)   # [B*N]

    # --------------------------------------------------
    # 4 Classification loss
    # --------------------------------------------------
    class_weights = torch.tensor([4.0, 2.0, 1.0, 1.0], device=pred_conf_flat.device)

    ce = F.cross_entropy(pred_conf_flat, gt_idx, reduction='none', weight=class_weights)

    pos_ce = ce[pos_mask]
    neg_ce = ce[neg_mask]

    cls_loss = pos_ce.mean() + neg_weight * neg_ce.mean()


    if pos_mask.any():
            pb = pred_box_flat[pos_mask]
            tb = ann_box_flat[pos_mask]
            box_loss = F.smooth_l1_loss(pb, tb, reduction='mean')
    else:
        box_loss = torch.zeros((), device=pred_box.device)

    return cls_loss + box_loss



class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        def CBR(cin, cout, k=3, s=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=k//2, bias=True),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True)
            )

        self.conv1  = CBR(3,   64, 3, 2)    # 320 -> 160
        self.conv2  = CBR(64,  64, 3, 1)
        self.conv3  = CBR(64, 128, 3, 2)    # 160 -> 80
        self.conv4  = CBR(128,128,3, 1)
        self.conv5  = CBR(128,256,3, 2)     # 80 -> 40
        self.conv6  = CBR(256,256,3, 1)
        self.conv7  = CBR(256,512,3, 2)     # 40 -> 20
        self.conv8  = CBR(512,512,3, 1)
        self.conv9  = CBR(512,512,3, 2)     # 20 -> 10

        # ============================================================
        # EXTRA SSD FEATURE MAPS
        # ============================================================
        self.conv10 = CBR(512,256,3,1)      # 10×10
        self.conv11 = CBR(256,256,3,2)      # 5×5
        self.conv12 = CBR(256,256,3,2)      # 3×3
        self.conv13 = CBR(256,256,3,2)      # 1×1

        self.pool1  = nn.AdaptiveAvgPool2d((1, 1))

        # ============================================================
        # PREDICTION HEADS
        # 4 boxes × 4 values = 16 channels  (!!!!)
        # ============================================================
        self.loc10  = nn.Conv2d(256, 16, 3, 1, 1, bias=True)
        self.conf10 = nn.Conv2d(256, 16, 3, 1, 1, bias=True)

        self.loc5   = nn.Conv2d(256, 16, 3, 1, 1, bias=True)
        self.conf5  = nn.Conv2d(256, 16, 3, 1, 1, bias=True)

        self.loc3   = nn.Conv2d(256, 16, 3, 1, 1, bias=True)
        self.conf3  = nn.Conv2d(256, 16, 3, 1, 1, bias=True)

        self.loc1   = nn.Conv2d(256, 16, 3, 1, 1, bias=True)
        self.conf1  = nn.Conv2d(256, 16, 3, 1, 1, bias=True)

        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        #x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        # ----------------------------------------------------
        # BACKBONE FEATURE EXTRACTION
        # ----------------------------------------------------
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)     # -> [B,512,10,10]

        # ---------------- SSD FEATURE MAPS ------------------
        f10 = self.conv10(x)     # [B,256,10,10]
        f5  = self.conv11(f10)   # [B,256,5,5]
        f3  = self.conv12(f5)    # [B,256,3,3]
        f1  = self.conv13(f3)    # [B,256,1,1]
        f1  = self.pool1(f1)     # 🔹 [B,256,1,1]  <-- FORCE 1x1

        # ---------------- LOC HEADS -------------------------
        loc10 = self.loc10(f10)
        loc5  = self.loc5(f5)
        loc3  = self.loc3(f3)
        loc1  = self.loc1(f1)

        # ---------------- CONF HEADS ------------------------
        conf10 = self.conf10(f10)
        conf5  = self.conf5(f5)
        conf3  = self.conf3(f3)
        conf1  = self.conf1(f1)

        # ============================================================
        # reshape / reorder into [B , NUM_BOXES , 4]
        # ============================================================
        def reshape_head(t):
            B, C, H, W = t.shape
            t = t.permute(0, 2, 3, 1)       # [B, H, W, 16]
            return t.reshape(B, -1, 4)      # [B, H*W*4 , 4]

        bbox10 = reshape_head(loc10)   # 10×10×4 = 400
        bbox5  = reshape_head(loc5)    # 5×5×4  = 100
        bbox3  = reshape_head(loc3)    # 3×3×4  = 36
        bbox1  = reshape_head(loc1)    # 1×1×4  = 4

        conf10 = reshape_head(conf10)
        conf5  = reshape_head(conf5)
        conf3  = reshape_head(conf3)
        conf1  = reshape_head(conf1)

        # ============================================================
        # CONCATENATE IN CORRECT ORDER (MUST MATCH DEFAULT BOX GEN)
        # ============================================================
        bboxes = torch.cat([bbox10, bbox5, bbox3, bbox1], dim=1)          # [B, 540, 4]
        confidence = torch.cat([conf10, conf5, conf3, conf1], dim=1)      # [B, 540, 4]
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        return confidence,bboxes










