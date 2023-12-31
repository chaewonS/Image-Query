import h5py
import numpy as np
from pathlib import Path
from pprint import pformat
from hloc import extract_features, match_features
import sqlite3

def nearest_neighbor_search(query_descriptor, db_global_descriptors):
    distances = np.linalg.norm(db_global_descriptors - query_descriptor, axis=1)
    most_similar_image_idx = np.argmin(distances)
    return most_similar_image_idx, distances[most_similar_image_idx]

def histogram_difference(hist1, hist2):
    hist1 = np.array(hist1.split(), dtype=int)
    hist2 = np.array(hist2.split(), dtype=int)
    diff = np.abs(hist1 - hist2)
    return np.sum(diff)

def histogram_matching(database_file, query_database_file, query_image_filenames):
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    query_conn = sqlite3.connect(query_database_file)
    query_cursor = query_conn.cursor()

    true_count = 0
    false_count = 0
    histogram_matching_results = []  # 히스토그램 매칭 결과 저장 리스트

    for query_image_filename in query_image_filenames:
        query_cursor.execute("SELECT filename, histogram FROM images WHERE filename=?", (query_image_filename,))
        query_image = query_cursor.fetchone()

        if query_image is not None:
            query_filename, query_histogram_str = query_image
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
            is_similar = id_difference == 1
            histogram_matching_results.append(is_similar)

            if is_similar:
                true_count += 1
            else:
                false_count += 1

    conn.close()
    return true_count, false_count, histogram_matching_results

def main():
    dataset = Path('/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur')
    # dataset = Path('/home/ubuntu/cw/data/kitti/split/image02')
    images_query = dataset / 'bag_query/'
    # images_query = dataset / 'Query_image02/'
    outputs = Path('/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur')
    # outputs = Path('/home/ubuntu/cw/data/kitti/split/image02')

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/bag-reference-feats-netvlad.h5'
    # database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/kitti-image02-refer-3243-feats-netvlad.h5'
    
    query_database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/bag-query-feats-netvlad.h5'
    # query_database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/kitti-image02-query-3243-feats-netvlad.h5'

    db_data = h5py.File(database_file, 'r')
    query_db_data = h5py.File(query_database_file, 'r')

    db_global_descriptors = np.array([db_data[group]['global_descriptor'][:] for group in db_data.keys() if 'global_descriptor' in db_data[group]])
    query_global_descriptors = np.array([query_db_data[group]['global_descriptor'][:] for group in query_db_data.keys() if 'global_descriptor' in query_db_data[group]])

    query_image_names = [group for group in query_db_data.keys() if 'global_descriptor' in query_db_data[group]]

    hm_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference.db"
    # hm_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/kitti_image02_refer_3243.db"
    hm_query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query.db"
    # hm_query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/kitti_image02_query_3243.db"

    true_count_hm, false_count_hm, _ = histogram_matching(hm_database_file, hm_query_database_file, query_image_names)
    
    # 임계값 범위 설정
    threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in threshold_values:
        true_count_hloc = 0
        false_count_hloc = 0

        for query_index in range(len(query_global_descriptors)):
            query_descriptor = query_global_descriptors[query_index]
            most_similar_image_idx, similarity = nearest_neighbor_search(query_descriptor, db_global_descriptors)

            query_image_name = query_image_names[query_index]
            
            db_image_name = list(db_data.keys())[most_similar_image_idx]
            query_image_id = int(''.join(filter(str.isdigit, query_image_name)))
            image_id = int(''.join(filter(str.isdigit, db_image_name)))
            hloc_id_difference = abs(query_image_id - image_id)
            hloc_is_similar = hloc_id_difference == 1
            
            # 히스토그램 매칭 결과 가져오기
            histogram_matching_results = histogram_matching(hm_database_file, hm_query_database_file, [query_image_name])[-1]

            # 가중치 설정
            weight_hloc = 0.8
            weight_histogram = 0.2
            threshold = 0.5

            # HLoc 결과 계산
            hloc_is_similar = 1 if hloc_is_similar else -1
            hloc_weighted_result = hloc_is_similar * weight_hloc

            # 히스토그램 매칭 결과 계산
            histogram_matching_results = 1 if histogram_matching_results == [True] else -1
            histogram_weighted_result = histogram_matching_results * weight_histogram

            # 가중 평균 계산
            combined_result = hloc_weighted_result + histogram_weighted_result

            # 최종 결과 출력 (임계값 설정)
            result = 'True' if abs(combined_result) >= threshold else 'False'
            
            # print(f"{query_image_name} hloc: {hloc_is_similar} histogram: {histogram_matching_results} combined_result: {combined_result} T/F Result: {result}")
            
            # 임계값(0.5)에 따라 True 또는 False로 결과를 카운트
            if abs(combined_result) >= threshold:
                true_count_hloc += 1
            else:
                false_count_hloc += 1

        final_true_percentage = (true_count_hloc / len(query_image_names)) * 100
        final_false_percentage = (false_count_hloc / len(query_image_names)) * 100

        print("\n")
        print(f"임계값: {threshold}")
        print(f"전체 중 True 개수: {true_count_hloc}/{len(query_image_names)}, 전체 중 False 개수: {false_count_hloc}/{len(query_image_names)}")
        print(f"True 비율: {final_true_percentage:.2f}%")
        print("\n")

if __name__ == "__main__":
    main()
