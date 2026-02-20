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
import numpy as np
import os
import cv2
import albumentations as A
#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    # total number of cells over all layers
    num_cells = sum([L * L for L in layers])      # 10*10 + 5*5 + 3*3 + 1*1 = 135
    # 4 boxes per cell, 8 attributes per box
    boxes = np.zeros((num_cells, 4, 8), dtype=np.float32)

    cell_idx = 0  # index over all cells (across all layers)

    for li, grid_size in enumerate(layers):
        lsize = large_scale[li]
        ssize = small_scale[li]

        # widths and heights for the 4 boxes in each cell
        w_list = [
            ssize,
            lsize,
            lsize * np.sqrt(2.0),
            lsize / np.sqrt(2.0)
        ]
        h_list = [
            ssize,
            lsize,
            lsize / np.sqrt(2.0),
            lsize * np.sqrt(2.0)
        ]

        # loop over all cells of this grid
        for i in range(grid_size):        # row (y)
            for j in range(grid_size):    # column (x)
                # center of this cell in normalized coords
                cx = (j + 0.5) / grid_size
                cy = (i + 0.5) / grid_size

                for k in range(4):
                    w = w_list[k]
                    h = h_list[k]

                    # corners before clipping
                    x_min = cx - w / 2.0
                    y_min = cy - h / 2.0
                    x_max = cx + w / 2.0
                    y_max = cy + h / 2.0

                    # clip to [0, 1]
                    x_min = max(0.0, x_min)
                    y_min = max(0.0, y_min)
                    x_max = min(1.0, x_max)
                    y_max = min(1.0, y_max)

                    # fill in attributes: [cx, cy, w, h, x_min, y_min, x_max, y_max]
                    boxes[cell_idx, k, 0] = cx
                    boxes[cell_idx, k, 1] = cy
                    boxes[cell_idx, k, 2] = w
                    boxes[cell_idx, k, 3] = h
                    boxes[cell_idx, k, 4] = x_min
                    boxes[cell_idx, k, 5] = y_min
                    boxes[cell_idx, k, 6] = x_max
                    boxes[cell_idx, k, 7] = y_max

                cell_idx += 1

    # reshape from [num_cells, 4, 8] -> [box_num, 8]
    boxes = boxes.reshape(-1, 8)
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    cat_id = int(cat_id)  # force cast to int
    assert isinstance(cat_id, (int, np.integer)), f"cat_id is NOT an int: {cat_id}, type={type(cat_id)}"
    assert 0 <= cat_id < ann_confidence.shape[1], f"cat_id {cat_id} out of bounds (max {ann_confidence.shape[1]-1})"

    
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    ious_true = ious>threshold
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    if not np.any(ious_true):
        ious_true = np.zeros_like(ious, dtype=bool)
        ious_true[np.argmax(ious)] = True

    # convert ground truth box to (center_x, center_y, width, height)
    gx = (x_min + x_max) / 2.0
    gy = (y_min + y_max) / 2.0
    gw = x_max - x_min
    gh = y_max - y_min

    #ious_true = np.argmax(ious)
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    pos_indices = np.where(ious_true)[0]

    for idx in pos_indices:
        # default box parameters (px, py, pw, ph)
        assert isinstance(idx, (int, np.integer)), f"idx not int: {idx}, type={type(idx)}"
        assert 0 <= idx < ann_confidence.shape[0], f"idx {idx} out of range"
        assert 0 <= cat_id < ann_confidence.shape[1], f"cat_id {cat_id} out of range"
        px, py, pw, ph = boxs_default[idx, 0:4]

        # relative regression targets
        tx = (gx - px) / pw
        ty = (gy - py) / ph
        tw = np.log(gw / pw + 1e-8)
        th = np.log(gh / ph + 1e-8)

        # update ann_box
        ann_box[idx, 0] = tx
        ann_box[idx, 1] = ty
        ann_box[idx, 2] = tw
        ann_box[idx, 3] = th

        # update ann_confidence (one-hot for this class)
        ann_confidence[idx, :] = 0.0
        ann_confidence[idx, cat_id] = 1.0


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        # Get sorted image list
        self.img_names = sorted(os.listdir(self.imgdir))
        
        split_idx = int(0.9 * len(self.img_names))
        if self.train:
            self.img_names = self.img_names[:split_idx]  # 90% for training
        else:
            self.img_names = self.img_names[split_idx:]  # 10% for validation

        print(f"[COCO] {len(self.img_names)} samples loaded ({'train' if train else 'val'})")
        
        if self.train:
            # self.transform = None   # temporarily disable Albumentations
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.BBoxSafeRandomCrop(p=0.3),
                    A.OneOf([
                      A.ColorJitter(p=0.3),
                      A.HueSaturationValue(p=0.3),
                      A.RandomGamma(gamma_limit=(80, 120), p=0.3),

                    ])
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",        # [x_min, y_min, x_max, y_max]
                    label_fields=["class_labels"]
                )
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        img = cv2.imread(img_name)                  # BGR, H x W x 3
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        orig_h, orig_w, _ = img.shape

        gt_boxes = []   # list of [x_min, y_min, x_max, y_max]
        gt_labels = []  # list of category ids

        # later we resize to self.image_size x self.image_size


        # ---------------------------------------------------------
        # 2. Read annotation file and update ann_box / ann_confidence
        #    Assume each line: class_id x_min y_min x_max y_max (in pixels)
        # ---------------------------------------------------------
        if os.path.exists(ann_name):
            with open(ann_name, "r") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    parts = line.split()
                    # class id: 0-cat, 1-dog, 2-person
                                        # class id: 0-cat, 1-dog, 2-person
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= (self.class_num - 1):  # only 0,1,2 are valid
                        # optional debug print
                        # print(f"[WARN] skip invalid class {class_id} in {ann_name}")
                        continue
                    # annotation format: x, y, w, h  (top-left corner + width/height)
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    # convert to corner coordinates
                    x_min = x
                    y_min = y
                    x_max = x + w
                    y_max = y + h

                    # clip to valid image region
                    x_min = max(0.0, x_min)
                    y_min = max(0.0, y_min)
                    x_max = min(orig_w - 1.0, x_max)
                    y_max = min(orig_h - 1.0, y_max)

                    gt_boxes.append([x_min, y_min, x_max, y_max])
                    gt_labels.append(class_id)

        if self.train and self.transform is not None and len(gt_boxes) > 0:
            augmented = self.transform(
                image=img,
                bboxes=gt_boxes,
                class_labels=gt_labels
            )
            img = augmented['image']
            gt_boxes = augmented['bboxes']
            gt_labels = augmented['class_labels']
        # -------------------------------------------------
        # 3. Use match() to assign this GT box to anchors
        # -------------------------------------------------
        cur_h, cur_w = img.shape[:2]
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0        # normalize to [0,1]
        # transpose to [C, H, W]
        img = np.transpose(img, (2, 0, 1))
        image = torch.from_numpy(img)               # tensor [3, 320, 320]
        
        for (class_id, box) in zip(gt_labels, gt_boxes):
            x_min, y_min, x_max, y_max = box
            
            scale_x = self.image_size / cur_w
            scale_y = self.image_size / cur_h

            x_min_norm = (x_min * scale_x) / self.image_size
            y_min_norm = (y_min * scale_y) / self.image_size
            x_max_norm = (x_max * scale_x) / self.image_size
            y_max_norm = (y_max * scale_y) / self.image_size
            # # compute resize scale
            # scale_x = self.image_size / orig_w
            # scale_y = self.image_size / orig_h

            # # resize GT box to match resized image
            # x_min_resized = x_min * scale_x
            # y_min_resized = y_min * scale_y
            # x_max_resized = x_max * scale_x
            # y_max_resized = y_max * scale_y

            # # normalize using resized coordinates
            # x_min_norm = x_min_resized / self.image_size
            # y_min_norm = y_min_resized / self.image_size
            # x_max_norm = x_max_resized / self.image_size
            # y_max_norm = y_max_resized / self.image_size

                    
            match(ann_box,
                  ann_confidence,
                  self.boxs_default,
                  self.threshold,
                  class_id,
                  x_min_norm,
                  y_min_norm,
                  x_max_norm,
                  y_max_norm)
        



        return image, ann_box, ann_confidence
