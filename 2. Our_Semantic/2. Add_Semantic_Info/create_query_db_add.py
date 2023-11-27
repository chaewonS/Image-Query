import os
import numpy as np
from PIL import Image
import sys
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from panopticfcn.config import add_panopticfcn_config
import sqlite3
from detectron2.data import MetadataCatalog
import shutil
from detectron2.structures import Boxes
import json
from detectron2.structures import Boxes
from pycocotools import mask as coco_mask
import cv2

config_file = "/home/ubuntu/hm/pan_FCN/cityscapes/tools_d2_cityscapes/configs/cityscapes/PanopticFCN-R50-400-3x-FAST.yaml"
weights_file = "/home/ubuntu/hm/pan_FCN/cityscapes/tools_d2_cityscapes/output/model_all_1126_transfer.pth"
# weights_file = "/home/ubuntu/hm/pan_FCN/cityscapes/tools_d2_cityscapes/output/model_final_640x480_270000.pth"

class_mapping = {
    0: 'unlabeled',
    1: 'road',
    2: 'sidewalk',
    3: 'building',
    4: 'wall',
    5: 'fence',
    6: 'pole',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    15: 'gas storage',
    16: 'hazard storage'
}

def setup_cfg(config_file, weights_file):
    cfg = get_cfg()
    add_panopticfcn_config(cfg)
    
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_file
    
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").thing_classes = ["person", "car", "truck", "huge truck"] #HM_original
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").set(thing_train_id2contiguous_id={ 0: 11, 1: 12, 2: 13, 3: 14 }) #HM_change_trainId
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").stuff_classes = ["unlabeled", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic sign", "vegetation", "terrain", "sky", "person", "car", "truck", "huge truck", "gas storage", "hazard storage"] #HM_original
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").set(stuff_train_id2contiguous_id={ 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16 }) #HM_change_trainId
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").stuff_colors = [(255, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (0, 0, 142), (0, 0, 70), (13, 208, 131), (170, 60, 10), (230, 180, 70)] 
    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").thing_classes = ["person", "car", "truck", "huge truck"] #HM_original
    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").set(thing_val_id2contiguous_id={0: 11, 1: 12, 2: 13, 3: 14 }) #HM_change_trainId
    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").stuff_classes = ["unlabeled", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic sign", "vegetation", "terrain", "sky", "person", "car", "truck", "huge truck", "gas storage", "hazard storage"] #HM_original
    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").set(stuff_val_id2contiguous_id={ 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16 }) #HM_change_trainId
    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").stuff_colors = [(255, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (0, 0, 142), (0, 0, 70), (13, 208, 131), (170, 60, 10), (230, 180, 70)] 

    cfg.freeze()
    return cfg 

cfg = setup_cfg(config_file, weights_file)
predictor = DefaultPredictor(cfg)

# image_directory = "/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/bag_query"
image_directory = "/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/300ms_query"

# database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query_counts.db"
database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/300ms_query_thing_oneformer2.db"

image_files = os.listdir(image_directory)

conn = sqlite3.connect(database_file)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS images
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   filename TEXT,
                   info_dict TEXT)''')

for image_file in image_files:
    annotations = []
    image_path = os.path.join(image_directory, image_file)
    image = np.array(Image.open(image_path))
    
    predictions = predictor(image)
    
    annotation = {}
    instances = predictions["instances"]
    pred_boxes = instances.pred_boxes if isinstance(instances.pred_boxes, Boxes) else Boxes(instances.pred_boxes)
    masks = instances.pred_masks
    boxes = pred_boxes.tensor.cpu().numpy()
    scores = instances.scores
    classes = instances.pred_classes
    size_list = []
    num_instances = len(predictions["panoptic_seg"][1])
    sem_seg = predictions["sem_seg"].argmax(dim=0).cpu().numpy()    
    
    for sem_seg_val in np.unique(sem_seg):
        mask = (sem_seg == sem_seg_val).astype(np.uint8)
        binary_mask = np.expand_dims(mask, axis=-1)
        rle = coco_mask.encode(np.asfortranarray(binary_mask))
        area = float(np.sum(binary_mask))
        bbox = cv2.boundingRect(mask)
        x, y, w, h = bbox
        bbox_list = list([x,y,w,h])
        counts_decoded = rle[0]['counts'].decode('utf-8')
        counts = ''.join(counts_decoded.split())
        category_id = None
        for i, seg_info in enumerate(predictions['panoptic_seg'][1]):
            if seg_info['category_id'] == sem_seg_val:
                category_id = seg_info['category_id']                    
                break
        if category_id is None:
            continue
        segment_field = {
            'counts': counts,
            'size': [rle[0]['size'][0], rle[0]['size'][1]]
        }
        annotation = {
            "category_id": int(category_id),
            "area": area,
            # "id": int(sem_seg_val),
            "bbox": bbox_list,
            # "image_id": image_id,
            "is_crowd": 0,
            "segmentation": segment_field
        }
        annotations.append(annotation)

    cursor.execute("INSERT INTO images (filename, info_dict) VALUES (?, ?)", (image_file, json.dumps(annotations)))
    conn.commit()

conn.close()
