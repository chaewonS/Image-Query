import numpy as np
import sqlite3
import cv2
import os
import torch
from hloc.matchers.nearest_neighbor import NearestNeighbor

# SQLite 데이터베이스 연결 만들기
database_file = '/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/global_index/database_global_index.db'
query_database_file = '/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/global_index/query_global_index.db'

conn = sqlite3.connect(database_file)
cursor = conn.cursor()

# query 데이터베이스 연결 만들기
query_conn = sqlite3.connect(query_database_file)
query_cursor = query_conn.cursor()

# SQLite 데이터베이스에서 이미지 파일 이름과 디스크립터 정보 가져오기
cursor.execute('SELECT filename, descriptors FROM global_index')
database_rows = cursor.fetchall()

# query SQLite 데이터베이스에서 query 이미지 파일 이름과 디스크립터 정보 가져오기
query_cursor.execute('SELECT filename, descriptors FROM global_index')
query_database_rows = query_cursor.fetchall()

# 디스크립터를 NumPy 배열로 변환
database_descriptors = [np.frombuffer(row[1], dtype=np.uint8) for row in database_rows]
query_descriptors = [np.frombuffer(row[1], dtype=np.uint8) for row in query_database_rows]

# 두 리스트의 길이를 맞추기 위해 짧은 리스트를 0으로 패딩
max_length = max(len(database_descriptors), len(query_descriptors))
database_descriptors_array = np.zeros((max_length, len(database_descriptors[0])), dtype=np.uint8)
query_descriptors_array = np.zeros((max_length, len(query_descriptors[0])), dtype=np.uint8)

database_descriptors_array[:len(database_descriptors)] = database_descriptors
query_descriptors_array[:len(query_descriptors)] = query_descriptors

# NearestNeighbor 모델 초기화
conf = {
    'ratio_threshold': None,
    'distance_threshold': None,
    'do_mutual_check': True,
}
model = NearestNeighbor(conf=conf)

# NumPy 배열을 텐서로 변환
query_descriptors_tensor = torch.tensor(query_descriptors_array, dtype=torch.uint8)
database_descriptors_tensor = torch.tensor(database_descriptors_array, dtype=torch.uint8)

# 모델에 입력 데이터 설정
input_data = {
    'descriptors0': query_descriptors_tensor,
    'descriptors1': database_descriptors_tensor,
}

# 이미지 검색 실행
output = model(input_data)

# 결과 처리
for q_idx, matches in enumerate(output['matches0']):
    best_match = matches.item()
    if best_match >= 0:
        best_filename = database_rows[best_match][0]
        print(f"Query 이미지 {q_idx + 1}의 가장 유사한 이미지: {best_filename}")
    else:
        print(f"Query 이미지 {q_idx + 1}: 일치하는 이미지가 없습니다.")

# 연결 종료
conn.close()
query_conn.close()
