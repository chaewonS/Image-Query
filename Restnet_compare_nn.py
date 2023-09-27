import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# 데이터 경로 설정
database_images_path = '/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/mapping_origin'
query_images_path = '/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/mapping_query'

# 이미지 전처리 및 CNN 모델 준비
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model = resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()

# Database Images 특징과 파일 이름 저장
database_features = []
database_filenames = []
for root, _, files in os.walk(database_images_path):
    for file in files:
        image_path = os.path.join(root, file)
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image)
        image = image.unsqueeze(0)  # 배치 차원 추가
        features = model(image)
        features = features.detach().numpy()  # 기울기를 필요로하지 않도록 수정
        database_features.append(features)
        database_filenames.append(file)

# Query Images 특징 저장
query_features = []
for root, _, files in os.walk(query_images_path):
    for file in files:
        image_path = os.path.join(root, file)
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image)
        image = image.unsqueeze(0)  # 배치 차원 추가
        features = model(image)
        features = features.detach().numpy()  # 기울기를 필요로하지 않도록 수정
        query_features.append(features)

# Nearest Neighbor Search를 위한 준비
database_features = np.concatenate(database_features, axis=0)
query_features = np.concatenate(query_features, axis=0)
nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(database_features)

# 각 Query Image에 대해 Nearest Neighbor Search 수행
for i, query_feature in enumerate(query_features):
    distances, indices = nbrs.kneighbors([query_feature])
    most_similar_image_index = indices[0][0]
    most_similar_image_filename = os.path.basename(database_filenames[most_similar_image_index])
    query_image_filename = os.path.splitext(os.path.basename(files[i]))[0]  # 숫자 부분만 추출
    most_similar_image_id = query_image_filename.split('raw')[-1].split('.')[0]  # 파일 이름에서 숫자 부분 추출
    most_similar_image_filename = os.path.splitext(most_similar_image_filename)[0]  # "raw"와 "jpg" 부분 제거
    print(f"Query 이미지 {i+1} (ID {most_similar_image_id}) : DB에서 가장 유사한 이미지는 ID {most_similar_image_filename.split('raw')[-1]}")
