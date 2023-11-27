# 레퍼런스/쿼리 DB 파일을 비교하여 distance 기반으로 정확도 출력하는 코드
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

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
threshold = 1.5

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
    
    final_accuracy = 0
    for histogram_diff in result_list:
        if histogram_diff <= threshold:
            final_accuracy += 1
            
conn.close()
query_conn.close()

total_queries = len(query_images)
accuracy_percentage = (final_accuracy / total_queries) * 100
print("\n")
print(f"Final Accuracy: {final_accuracy}/{total_queries} ({accuracy_percentage:.2f}%)")
print("\n")
