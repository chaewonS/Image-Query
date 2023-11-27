import sqlite3
import numpy as np
import json

# database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference_add.db"
database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/300ms_reference_oneformer.db"

# query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query_add.db"
query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/300ms_query_oneformer.db"

conn = sqlite3.connect(database_file)
cursor = conn.cursor()

query_conn = sqlite3.connect(query_database_file)
query_cursor = query_conn.cursor()

def histogram_difference(hist1, hist2):
    hist1 = np.array(hist1.split(), dtype=int)
    hist2 = np.array(hist2.split(), dtype=int)
    diff = np.abs(hist1 - hist2)
    return np.sum(diff)

query_cursor.execute("SELECT filename, info_dict FROM images")
query_images = query_cursor.fetchall()

true_count = 0
false_count = 0

for query_idx, (query_filename, query_info_dict) in enumerate(query_images):
    query_image_id = int(''.join(filter(str.isdigit, query_filename)))
    query_info_dict = json.loads(query_info_dict)
    min_difference = float('inf')
    most_similar_image_filename = ""
    most_similar_image_id = ""
    
    cursor.execute("SELECT filename, info_dict FROM images")
    images = cursor.fetchall()
    
    for filename, info_dict in images:
        info_dict = json.loads(info_dict)
        difference = histogram_difference(query_info_dict['hist_str'], info_dict['hist_str'])
        image_id = int(''.join(filter(str.isdigit, filename)))
        
        num_instances_difference = abs(int(query_info_dict['num_instances']) - int(info_dict['num_instances']))
        
        query_boxes = np.array(query_info_dict.get('pred_boxes', []))
        info_boxes = np.array(info_dict.get('pred_boxes', []))
        
        # query_boxes가 비어 있을 때 boxes_difference를 0으로 처리
        if query_boxes.size == 0:
            boxes_difference = 0
        else:
            # pred_boxes 비교
            query_boxes_count = query_boxes.shape[0]
            info_boxes_count = info_boxes.shape[0]
            boxes_difference = abs(query_boxes_count - info_boxes_count)
            
        # pred_classes 배열의 크기를 맞추는 작업
        query_classes = np.array(query_info_dict.get('pred_classes', []))
        info_classes = np.array(info_dict.get('pred_classes', []))
        max_length = max(len(query_classes), len(info_classes))
        query_classes = np.pad(query_classes, (0, max_length - len(query_classes)), mode='constant')
        info_classes = np.pad(info_classes, (0, max_length - len(info_classes)), mode='constant')
        classes_difference = np.sum(query_classes != info_classes)
        
        # total_difference = difference + num_instances_difference + boxes_difference + classes_difference
        total_difference = num_instances_difference + boxes_difference + classes_difference

        if total_difference < min_difference:
            min_difference = total_difference
            most_similar_image_filename = filename
            most_similar_image_id = image_id
    
    id_difference = abs(query_image_id - most_similar_image_id)
    is_similar = id_difference == 1

    if is_similar:
        true_count += 1
    else:
        false_count += 1
        
# 데이터베이스 연결 종료
conn.close()
query_conn.close()

total_count = true_count + false_count
true_percentage = (true_count / total_count) * 100
false_percentage = (false_count / total_count) * 100
print("\n")
print(f"전체 중 True 비율: {true_percentage:.2f}%")
print("\n")
