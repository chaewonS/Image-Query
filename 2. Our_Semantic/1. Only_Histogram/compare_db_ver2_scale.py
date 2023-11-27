# 레퍼런스/쿼리 DB 파일을 비교하여 distance 기반으로 정확도 출력하는 코드
# output에 스케일링 수행
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import time

database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference.db"
query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query.db"

conn = sqlite3.connect(database_file)
cursor = conn.cursor()

query_conn = sqlite3.connect(query_database_file)
query_cursor = query_conn.cursor()

def histogram_difference(hist1, hist2):
    hist1 = np.array(hist1.split(), dtype=int)
    hist2 = np.array(hist2.split(), dtype=int)
    diff = np.abs(hist1 - hist2)
    return np.sum(diff)

query_cursor.execute("SELECT filename, histogram FROM images")
query_images = query_cursor.fetchall()

true_count = 0
false_count = 0
result_list = []
threshold = 0.2

def scale_value(value, old_min, old_max, new_min, new_max):
    if old_min == old_max:
        return new_min
    old_range = old_max - old_min
    new_range = new_max - new_min
    scaled_value = (value - old_min) / old_range * new_range + new_min
    return scaled_value

start_time = time.time()  # 코드 실행 시작 시간 기록

for query_idx, (query_filename, query_histogram_str) in enumerate(query_images):
    query_image_id = int(''.join(filter(str.isdigit, query_filename)))

    min_difference = float('inf')
    most_similar_image_filename = ""
    most_similar_image_id = ""

    cursor.execute("SELECT filename, histogram FROM images")
    images = cursor.fetchall()

    for filename, histogram_str in images:
        difference = histogram_difference(query_histogram_str, histogram_str)
        image_id = int(''.join(filter(str.isdigit, filename)))

        if difference < min_difference:
            min_difference = difference
            most_similar_image_filename = filename
            most_similar_image_id = image_id

    id_difference = abs(query_image_id - most_similar_image_id)
    result_list.append(id_difference)
    
    histogram_min = min(result_list)
    histogram_max = max(result_list)
    scaled_histogram_results = [scale_value(value, histogram_min, histogram_max, 0, 1) for value in result_list]

    final_accuracy = 0
    for histogram_diff in scaled_histogram_results:
        if histogram_diff <= threshold:
            final_accuracy += 1

end_time = time.time()  # 코드 실행 종료 시간 기록
total_execution_time = end_time - start_time  # 실행 시간 계산
     
conn.close()
query_conn.close()

total_queries = len(query_images)
accuracy_percentage = (final_accuracy / total_queries) * 100
print("\n")
print(f"Final Accuracy: {final_accuracy}/{total_queries} ({accuracy_percentage:.2f}%)")
print("\n")
print(f"Total Execution Time: {total_execution_time:.2f} seconds")
