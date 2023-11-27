import sqlite3
import numpy as np
import json

# database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference_ver3_stuff.db"
database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/300ms_reference_thing_oneformer2.db"

# query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query_ver3_stuff.db"
query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/300ms_query_thing_oneformer2.db"

conn = sqlite3.connect(database_file)
cursor = conn.cursor()

query_conn = sqlite3.connect(query_database_file)
query_cursor = query_conn.cursor()

query_cursor.execute("SELECT filename, info_dict FROM images")
query_images = query_cursor.fetchall()

cursor.execute("SELECT filename, info_dict FROM images")
reference_images = cursor.fetchall()

def histogram_difference(hist1, hist2):
    hist1 = np.array(hist1.split(), dtype=int)
    hist2 = np.array(hist2.split(), dtype=int)
    diff = np.abs(hist1 - hist2)
    return np.sum(diff)

def bbox_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    
    union_area = bbox1_area + bbox2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

def compare_images(query_info, reference_info):
    total_area_diff = 0
    total_iou_score = 0
    
    query_categories = {obj['category_id']: obj for obj in query_info}
    reference_categories = {obj['category_id']: obj for obj in reference_info}
    
    for category_id, query_obj in query_categories.items():
        if category_id in reference_categories:
            reference_obj = reference_categories[category_id]
            comparison = compare_objects(query_obj, reference_obj)
            if comparison['area_diff'] is not None:
                total_area_diff += comparison['area_diff']
                total_iou_score += comparison['bbox_iou']
    
    return total_area_diff, total_iou_score


def compare_objects(obj1, obj2):
    if obj1['category_id'] == obj2['category_id'] and obj1['is_crowd'] == obj2['is_crowd']:
        area_diff = abs(obj1['area'] - obj2['area'])
        bbox_iou_score = bbox_iou(obj1['bbox'], obj2['bbox'])
        
        return {
            'area_diff': area_diff,
            'bbox_iou': bbox_iou_score
        }
    else:
        return {
            'area_diff': None,
            'bbox_iou': 0
        }

true_count = 0
false_count = 0

for query_idx, (query_filename, query_info_dict) in enumerate(query_images):
    query_info = json.loads(query_info_dict)
    min_difference = float('inf')  # 최소 차이를 저장할 변수 초기화
    most_similar_image_filename = ""
    
    for filename, reference_info_dict in reference_images:
        reference_info = json.loads(reference_info_dict)
        
        # 총 면적 차이와 iou 점수 계산
        total_area_diff, total_iou_score = compare_images(query_info, reference_info)
        
        # 면적 차이와 IOU 점수를 기반으로 가장 유사한 이미지를 찾음
        if total_area_diff < min_difference:
            min_difference = total_area_diff
            most_similar_image_filename = filename
            most_similar_image_id = int(''.join(filter(str.isdigit, filename)))
    
    query_image_id = int(''.join(filter(str.isdigit, query_filename)))
    id_difference = abs(query_image_id - most_similar_image_id)
    
    # ID 차이를 기준으로 유사 여부 판단
    is_similar = id_difference <= 1 
    if is_similar:
        true_count += 1
    else:
        false_count += 1

# 데이터베이스 연결 종료
conn.close()
query_conn.close()

# 결과 출력
total_count = true_count + false_count
true_percentage = (true_count / total_count) * 100
false_percentage = (false_count / total_count) * 100

print("\n")
print(f"전체 중 True 비율: {true_percentage:.2f}%")
# print(f"전체 중 False 비율: {false_percentage:.2f}%")
print("\n")
