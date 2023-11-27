# 레퍼런스/쿼리 DB 파일 비교해서 유사 이미지 ID 출력하는 코드
import sqlite3
import numpy as np

database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/database_origin.db"
query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/database_query.db" 

# 데이터베이스 연결
conn = sqlite3.connect(database_file)
cursor = conn.cursor() 

# 쿼리 이미지 DB 연결
query_conn = sqlite3.connect(query_database_file)
query_cursor = query_conn.cursor()

# 히스토그램 차이를 계산하는 함수
def histogram_difference(hist1, hist2):
    hist1 = np.array(hist1.split(), dtype=int)
    hist2 = np.array(hist2.split(), dtype=int)
    diff = np.abs(hist1 - hist2)
    return np.sum(diff)

# 쿼리 이미지 DB에서 쿼리 이미지 정보 가져오기
query_cursor.execute("SELECT filename, histogram FROM images")
query_images = query_cursor.fetchall()

# 쿼리 이미지들을 순회하며 각각의 가장 유사한 이미지 찾기
for query_idx, (query_filename, query_histogram_str) in enumerate(query_images):
    # 숫자 부분 추출
    query_image_id = int(''.join(filter(str.isdigit, query_filename)))

    # 최소 히스토그램 차이를 추적하기 위한 변수 초기화
    min_difference = float('inf')
    most_similar_image_filename = ""
    most_similar_image_id = ""

    # 원본 이미지 테이블에서 모든 이미지를 가져와서 히스토그램 차이 계산
    cursor.execute("SELECT filename, histogram FROM images")
    images = cursor.fetchall()

    # 쿼리 이미지와 가장 유사한 이미지 찾기
    for filename, histogram_str in images:
        difference = histogram_difference(query_histogram_str, histogram_str)
        image_id = int(''.join(filter(str.isdigit, filename)))

        if difference < min_difference:
            min_difference = difference
            most_similar_image_filename = filename
            most_similar_image_id = image_id

    # 결과 출력
    print(f"쿼리 이미지 {query_idx + 1} (ID {query_image_id}): DB에서 가장 유사한 이미지는 ID {most_similar_image_id}")

# 데이터베이스 연결 종료
conn.close()
query_conn.close()
