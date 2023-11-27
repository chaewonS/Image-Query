# Hloc에서 h5 형식의 파일을 db 형식의 파일로 변환하는 코드
import h5py
import sqlite3
import numpy as np

# H5 파일 경로
h5_file_path = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/bag-reference-feats-netvlad.h5'

# SQLite DB 파일 경로
db_file_path = '/home/ubuntu/cw/Hierarchical-Localization/outputs/integrate/combined_bag_refer.db'

# .db 파일을 열어 연결
conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()

# 테이블 생성 (예: 이미지 정보를 저장할 images 테이블)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        filename TEXT PRIMARY KEY,
        global_descriptor BLOB
    )
''')

# H5 파일에서 데이터 읽기
with h5py.File(h5_file_path, 'r') as h5_file:
    for key in h5_file.keys():
        if key.startswith('image_raw'):
            # 이미지 파일 이름
            filename = key.split('/')[-1]
            # global_descriptor 데이터
            global_descriptor = h5_file[key]['global_descriptor'][...]  
            global_descriptor_str = ' '.join(map(str, global_descriptor))
            cursor.execute('INSERT INTO images (filename, global_descriptor) VALUES (?, ?)', (filename, global_descriptor_str))

# 변경 사항 저장 및 연결 닫기
conn.commit()
conn.close()
