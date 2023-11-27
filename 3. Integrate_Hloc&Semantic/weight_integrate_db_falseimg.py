# hloc + histogram weight 매칭 결과
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

def histogram_matching(database_file, query_database_file, false_query_images):
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    query_conn = sqlite3.connect(query_database_file)
    query_cursor = query_conn.cursor()

    def histogram_difference(hist1, hist2):
        hist1 = np.array(hist1.split(), dtype=int)
        hist2 = np.array(hist2.split(), dtype=int)
        diff = np.abs(hist1 - hist2)
        return np.sum(diff)

    true_count = 0
    false_count = 0

    for query_image_filename in false_query_images:
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

            if is_similar:
                true_count += 1
            else:
                false_count += 1

    conn.close()
    return true_count

def main():
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

    false_query_images = []

    for query_index in range(len(query_global_descriptors)):
        query_descriptor = query_global_descriptors[query_index]
        most_similar_image_idx, similarity = nearest_neighbor_search(query_descriptor, db_global_descriptors)

        query_image_name = query_image_names[query_index]
        
        # 리스트로 변환 후 인덱싱
        db_image_name = list(db_data.keys())[most_similar_image_idx]
        query_image_id = int(''.join(filter(str.isdigit, query_image_name)))
        image_id = int(''.join(filter(str.isdigit, db_image_name)))
        id_difference = abs(query_image_id - image_id)
        is_similar = id_difference == 1

        if not is_similar:
            false_query_images.append(query_image_name)

    db_data.close()
    query_db_data.close()

    true_count = len(query_global_descriptors) - len(false_query_images)
    false_count = len(false_query_images)

    database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference.db"
    query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query.db"


    true_count_hm = histogram_matching(database_file, query_database_file, false_query_images)
    false_count_hm = len(false_query_images) - true_count_hm
    final_true_count = 0.8 * true_count + 0.2 * true_count_hm
    final_false_count = false_count_hm

    final_true_percentage = (final_true_count / (final_true_count + final_false_count)) * 100
    final_false_percentage = (final_false_count / (final_true_count + final_false_count)) * 100
    print(f"Hloc 깃허브 테스트 결과, True 개수: {true_count}, False 개수: {false_count}")
    print(f"Histogram 매칭 테스트 결과, True 개수: {true_count_hm}, False 개수: {false_count_hm}")
    print(f"weight 기반 합성 알고리즘 결과, True 개수: {final_true_count}")
    print(f"weight 기반 합성 알고리즘 결과, False 개수: {final_false_count}")
    print(f"weight 기반 합성 알고리즘 결과, True 비율: {final_true_percentage:.2f}%")

if __name__ == "__main__":
    main()
