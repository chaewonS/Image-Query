import os
import numpy as np
from PIL import Image
from torchvision import transforms
from third_party.SuperGluePretrainedNetwork.models.superpoint import SuperPoint
import torch
import sqlite3

# 데이터 경로 설정
database_images_path = '/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/mapping_origin'
database_file = '/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/global_index/database_global_index.db'

# SQLite 데이터베이스 연결 만들기
conn = sqlite3.connect(database_file)
cursor = conn.cursor()

# 테이블 생성 (이미 존재하는 경우 무시)
cursor.execute('''CREATE TABLE IF NOT EXISTS global_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    descriptors BLOB
                  )''')

# 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((320, 240)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

# SuperPoint 모델 초기화
superpoint_model = SuperPoint(config={})

# 데이터베이스 이미지의 로컬 특징 추출 및 SQLite 데이터베이스에 삽입
def process_images(image_folder):
    for root, _, files in os.walk(image_folder):
        for file in files:
            image_path = os.path.join(root, file)
            _, descriptors = extract_local_features(image_path)

            if len(descriptors) > 0:
                # SQLite 데이터베이스에 삽입 (filename과 descriptors만 저장)
                cursor.execute('INSERT INTO global_index (filename, descriptors) VALUES (?, ?)',
                               (file, descriptors.tobytes()))

    # 데이터베이스 이미지의 로컬 특징을 SQLite 데이터베이스에 저장
    conn.commit()

# SuperPoint 모델을 사용하여 로컬 특징 추출
def extract_local_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)
    superpoint_outputs = superpoint_model({'image': image})
    descriptors = superpoint_outputs['descriptors'][0].detach().cpu().numpy()

    return image, descriptors

# 데이터베이스 이미지의 로컬 특징 추출 및 SQLite 데이터베이스에 저장
process_images(database_images_path)

# 연결 종료
conn.close()
