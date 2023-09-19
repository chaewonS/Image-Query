import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from panopticfcn.config import add_panopticfcn_config
from detectron2.data import MetadataCatalog
import shutil

# 설정 파일과 가중치 파일 경로
config_file = "/home/ubuntu/hm/pan_FCN/cityscapes/tools_d2_cityscapes/configs/cityscapes/PanopticFCN-R50-400-3x-FAST.yaml"
weights_file = "/home/ubuntu/hm/pan_FCN/cityscapes/tools_d2_cityscapes/output/model_976_transfer.pth"
# weights_file = "/home/ubuntu/hm/pan_FCN/cityscapes/tools_d2_cityscapes/output/model_final_65000.pth"

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
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").thing_classes = ["person", "car", "truck", "huge truck"]
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").set(thing_train_id2contiguous_id={ 0: 11, 1: 12, 2: 13, 3: 14 })
    
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").stuff_classes = ["unlabeled", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic sign", "vegetation", "terrain", "sky", "person", "car", "truck", "huge truck", "gas storage", "hazard storage"]
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").set(stuff_train_id2contiguous_id={ 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16 })
    MetadataCatalog.get("cityscapes_fine_panoptic_train_separated").stuff_colors = [(255, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (0, 0, 142), (0, 0, 70), (13, 208, 131), (170, 60, 10), (230, 180, 70)] 

    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").thing_classes = ["person", "car", "truck", "huge truck"]
    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").set(thing_val_id2contiguous_id={0: 11, 1: 12, 2: 13, 3: 14 })

    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").stuff_classes = ["unlabeled", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic sign", "vegetation", "terrain", "sky", "person", "car", "truck", "huge truck", "gas storage", "hazard storage"]
    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").set(stuff_val_id2contiguous_id={ 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16 })
    MetadataCatalog.get("cityscapes_fine_panoptic_val_separated").stuff_colors = [(255, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (0, 0, 142), (0, 0, 70), (13, 208, 131), (170, 60, 10), (230, 180, 70)] 

    cfg.freeze()
    return cfg 

# 설정 초기화
cfg = setup_cfg(config_file, weights_file)

# Predictor 생성
predictor = DefaultPredictor(cfg)

# 이미지 및 히스토그램 저장 디렉토리
image_directory = "/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/mapping"
histogram_directory = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/histogram_0915"
predictor_directory = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/predictor"

# 이미지 파일 목록
image_files = os.listdir(image_directory)

# 이미지 및 히스토그램 저장 디렉토리 초기화
if os.path.exists(histogram_directory):
    shutil.rmtree(histogram_directory)
os.makedirs(histogram_directory)

if os.path.exists(predictor_directory):
    shutil.rmtree(predictor_directory)
os.makedirs(predictor_directory)

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

    # 히스토그램 시각화
    plt.figure(figsize=(10, 4))

    for i, class_name in enumerate(class_mapping.values()):
        plt.bar(class_name, hist[i], color='b')

    plt.xlabel('Semantic Class', fontsize=8)
    plt.ylabel('Frequency')
    plt.title(f'Semantic Class Histogram for Image: {image_file}')
    plt.xticks(rotation=45, fontsize=8)

    # Semantic Segmentation 결과 이미지 저장
    predictor_filename = f"{os.path.splitext(image_file)[0]}_predictor.png"
    predictor_path = os.path.join(predictor_directory, predictor_filename)
    # Semantic Segmentation 결과 저장
    plt.savefig(predictor_path, bbox_inches='tight')  

    plt.subplot(1, 2, 2)
    plt.imshow(sem_seg, cmap='nipy_spectral')
    plt.title('Semantic Segmentation')
    plt.axis('off')

    # 히스토그램 이미지 저장
    histogram_filename = f"{os.path.splitext(image_file)[0]}_histogram.png"
    histogram_path = os.path.join(histogram_directory, histogram_filename)
    plt.savefig(histogram_path, bbox_inches='tight')
