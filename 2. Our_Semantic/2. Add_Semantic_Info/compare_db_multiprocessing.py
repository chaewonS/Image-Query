# 시맨틱 정보를 포함한 DB 파일을 비교하는데 시간이 오래 소요되므로,
# 멀티프로세싱 방식을 사용하도록 수정한 코드드
import sqlite3
import numpy as np
import json
from pycocotools import mask as mask_utils
from multiprocessing import Pool, cpu_count

# database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference_counts.db"
database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/300ms_reference_oneformer.db"

# query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query_counts.db"
query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/300ms_query_oneformer.db"

conn = sqlite3.connect(database_file)
cursor = conn.cursor()
query_conn = sqlite3.connect(query_database_file)
query_cursor = query_conn.cursor()

cursor.execute("SELECT filename, info_dict FROM images")
reference_images = cursor.fetchall()
query_cursor.execute("SELECT filename, info_dict FROM images")
query_images = query_cursor.fetchall()

# 멀티프로세싱에서 사용할 함수들 정의
def rle_mask_iou(rle1, rle2):
    mask1 = mask_utils.decode(rle1)
    mask2 = mask_utils.decode(rle2)
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return intersection / union if union > 0 else 0

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

def compare_objects(obj1, obj2):
    if obj1['category_id'] == obj2['category_id'] and obj1['is_crowd'] == obj2['is_crowd']:
        area_diff = abs(obj1['area'] - obj2['area'])
        bbox_iou_score = bbox_iou(obj1['bbox'], obj2['bbox'])
        mask_iou_score = rle_mask_iou(obj1['segmentation'], obj2['segmentation'])

        return {
            'area_diff': area_diff,
            'bbox_iou': bbox_iou_score,
            'mask_iou': mask_iou_score  # 마스크 IoU 추가
        }
    else:
        return {
            'area_diff': None,
            'bbox_iou': 0,
            'mask_iou': 0  # 마스크 IoU 추가
        }
        
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

def process_query_image(data):
    query_filename, query_info_dict, reference_images = data
    query_info = json.loads(query_info_dict)
    min_difference = float('inf')
    most_similar_image_filename = ""
    most_similar_image_id = -1
    
    for filename, reference_info_dict in reference_images:
        reference_info = json.loads(reference_info_dict)
        total_area_diff, total_iou_score = compare_images(query_info, reference_info)
        
        if total_area_diff < min_difference:
            min_difference = total_area_diff
            most_similar_image_filename = filename
            most_similar_image_id = int(''.join(filter(str.isdigit, filename)))
    
    query_image_id = int(''.join(filter(str.isdigit, query_filename)))
    id_difference = abs(query_image_id - most_similar_image_id)
    
    return id_difference <= 1

conn.close()
query_conn.close()

data_for_processing = [(query_filename, query_info_dict, reference_images) for query_filename, query_info_dict in query_images]

if __name__ == '__main__':
    pool = Pool(processes=cpu_count())
    results = pool.map(process_query_image, data_for_processing)
    pool.close()
    pool.join()

    true_count = sum(results)
    total_count = len(results)
    true_percentage = (true_count / total_count) * 100

    print("\n")
    print(f"전체 중 True 비율: {true_percentage:.2f}%")
    print("\n")
