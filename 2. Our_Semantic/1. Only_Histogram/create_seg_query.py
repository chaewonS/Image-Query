# 히스토그램 정보만을 사용하여 (hist_str, area) 쿼리 데이터베이스 생성하는 코드
import os
import numpy as np
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from panopticfcn.config import add_panopticfcn_config
import sqlite3
from detectron2.data import MetadataCatalog
import shutil

# 설정 파일과 가중치 파일 경로
config_file = "/home/ubuntu/hm/pan_FCN/cityscapes/tools_d2_cityscapes/configs/cityscapes/PanopticFCN-R50-400-3x-FAST.yaml"
weights_file = "/home/ubuntu/hm/pan_FCN/cityscapes/tools_d2_cityscapes/output/model_976_transfer.pth"

# 클래스 매핑
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
    
    # 설정 파일과 가중치 파일 설정
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_file
    
    # 클래스 및 색상 매핑 설정
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

# 설정 초기화
cfg = setup_cfg(config_file, weights_file)

# Predictor 생성
predictor = DefaultPredictor(cfg)

# 이미지 및 히스토그램 저장 디렉토리
image_directory = "/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/mapping_query"
# histogram_directory = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/histogram_0915"
database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/database_query.db"

# 이미지 파일 목록
image_files = os.listdir(image_directory)

# 데이터베이스 연결
conn = sqlite3.connect(database_file)
cursor = conn.cursor()

# 데이터베이스 테이블 생성
cursor.execute('''CREATE TABLE IF NOT EXISTS images
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   filename TEXT,
                   histogram TEXT)''')

# 이미지 처리 및 히스토그램 생성 반복
for image_file in image_files:
    # 이미지 파일 로드
    image_path = os.path.join(image_directory, image_file)
    image = np.array(Image.open(image_path))
    
    # 이미지 세그멘테이션 예측
    outputs = predictor(image)
    sem_seg = outputs["sem_seg"].argmax(dim=0).cpu().numpy()
    
    # 클래스 별 픽셀 수 계산
    hist = np.zeros(len(class_mapping), dtype=int)

    for i, class_id in enumerate(class_mapping.keys()):
        class_pixel_count = np.sum(sem_seg == class_id)
        hist[i] = class_pixel_count

    # 히스토그램을 문자열로 변환
    hist_str = ' '.join(map(str, hist))

    # 히스토그램 정보를 데이터베이스에 저장
    cursor.execute("INSERT INTO images (filename, histogram) VALUES (?, ?)", (image_file, hist_str))
    conn.commit()

# 데이터베이스 연결 종료
conn.close()
