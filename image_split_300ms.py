import shutil
import os
import random

# 원본 이미지가 있는 디렉토리 경로
source_directory = "/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/230905_jackalmapping_300ms"
# Query 이미지를 저장할 디렉토리 경로
query_directory = "/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/300ms_query_random"

# Reference 이미지를 저장할 디렉토리 경로 
reference_directory = "/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/300ms_reference_random"

# 총 이미지 개수
total_images = 1000
# 선택할 무작위 쿼리 이미지 개수
num_query_images = 200

# 무작위로 쿼리 이미지를 선택
query_image_indices = random.sample(range(total_images), num_query_images)

# Query 이미지 및 Reference 이미지를 분할하는 함수
def split_images():
    for i in range(total_images):
        # source_filename = f"image_raw{i}.jpg"
        source_filename = f"image_raw{i}.jpg"

        # 이미지 번호가 query_image_indices에 속하면 쿼리 이미지로 분할
        if i in query_image_indices:
            destination = os.path.join(query_directory, source_filename)
        else:
            destination = os.path.join(reference_directory, source_filename)

        source_path = os.path.join(source_directory, source_filename)

        # 파일을 복사 또는 이동
        shutil.copy(source_path, destination)  # 이미지를 복사하려면 shutil.copy 대신 shutil.move 사용

# Query 이미지와 Reference 이미지를 분할
split_images()

print("이미지 분할이 완료되었습니다.")
