import h5py
import numpy as np
from pathlib import Path
from pprint import pformat
from hloc import extract_features, match_features
import sqlite3
import matplotlib.pyplot as plt
import os
import time

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
    histogram_matching_results = []

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

    conn.close()
    return true_count, false_count, id_difference

def scale_value(value, old_min, old_max, new_min, new_max):
    if old_min == old_max:
        return new_min
    old_range = old_max - old_min
    new_range = new_max - new_min
    scaled_value = (value - old_min) / old_range * new_range + new_min
    return scaled_value

def main():
    start_time = time.time()  # 코드 실행 시작 시간 기록
    dataset = Path('/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur')
    images_query = dataset / 'bag_query/'
    outputs = Path('/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur')

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/bag-reference-feats-netvlad.h5'
    
    query_database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/bag-query-feats-netvlad.h5'

    db_data = h5py.File(database_file, 'r')
    query_db_data = h5py.File(query_database_file, 'r')

    db_global_descriptors = np.array([db_data[group]['global_descriptor'][:] for group in db_data.keys() if 'global_descriptor' in db_data[group]])
    query_global_descriptors = np.array([query_db_data[group]['global_descriptor'][:] for group in query_db_data.keys() if 'global_descriptor' in query_db_data[group]])

    query_image_names = [group for group in query_db_data.keys() if 'global_descriptor' in query_db_data[group]]

    hm_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference.db"
    hm_query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query.db"

    true_count_hm, false_count_hm, _ = histogram_matching(hm_database_file, hm_query_database_file, query_image_names)
    
    final_results = []

    weight_hloc = 0.9
    weight_histogram = 0.1
    threshold = 0.15
    hloc_results = []
    result_list = []
    weighted_diff_list = []
    final_accuracy = 0
    

    for query_index in range(len(query_global_descriptors)):
        query_descriptor = query_global_descriptors[query_index]
        most_similar_image_idx, similarity = nearest_neighbor_search(query_descriptor, db_global_descriptors)

        query_image_name = query_image_names[query_index]

        db_image_name = list(db_data.keys())[most_similar_image_idx]
        query_image_id = int(''.join(filter(str.isdigit, query_image_name)))
        image_id = int(''.join(filter(str.isdigit, db_image_name)))
        hloc_id_difference = abs(query_image_id - image_id)
        hloc_is_similar = hloc_id_difference == 1
        hloc_results.append(hloc_id_difference)
        
        histogram_matching_results = histogram_matching(hm_database_file, hm_query_database_file, [query_image_name])[-1]
        result_list.append(histogram_matching_results)

        hloc_min = min(hloc_results)
        hloc_max = max(hloc_results)
        histogram_min = min(result_list)
        histogram_max = max(result_list)
        scaled_hloc_results = [scale_value(value, hloc_min, hloc_max, 0, 1) for value in hloc_results]
        scaled_histogram_results = [scale_value(value, histogram_min, histogram_max, 0, 1) for value in result_list]

        weighted_scaled_diff_list = [(weight_hloc * hloc + weight_histogram * histogram) for hloc, histogram in zip(scaled_hloc_results, scaled_histogram_results)]

        final_accuracy = 0
        for weighted_diff in weighted_scaled_diff_list:
            if weighted_diff <= threshold:
                final_accuracy += 1
                
    total_queries = len(query_global_descriptors)
    accuracy_percentage = (final_accuracy / total_queries) * 100
    weighted_average = np.mean(weighted_scaled_diff_list)
    
    end_time = time.time()
    total_execution_time = end_time - start_time

    print("\n")
    print(f"Final Accuracy: {final_accuracy}/{total_queries} ({accuracy_percentage:.2f}%)")
    print("\n")
    print(f"Total Execution Time: {total_execution_time:.2f} seconds") 

if __name__ == "__main__":
    main()
