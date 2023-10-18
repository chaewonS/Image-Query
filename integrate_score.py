import h5py
import numpy as np
from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization
import sqlite3
import numpy as np

def nearest_neighbor_search(query_descriptor, db_global_descriptors, query_image_names, db_image_names):
    distances = np.linalg.norm(db_global_descriptors - query_descriptor, axis=1)
    most_similar_image_idx = np.argmin(distances)
    return most_similar_image_idx, distances[most_similar_image_idx]

def main():
    dataset = Path('/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur')
    images = dataset / '300ms_reference/'
    images_query = dataset / '300ms_query/'

    outputs = Path('/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur')
    sfm_pairs = outputs / 'pairs-factory.txt'
    loc_pairs = outputs / 'pairs-query-factory.txt'
    print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
    print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/300ms-random-reference-feats-netvlad.h5'
    query_database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/300ms-random-query-feats-netvlad.h5'

    db_data = h5py.File(database_file, 'r')
    query_db_data = h5py.File(query_database_file, 'r')

    db_global_descriptors = np.array([db_data[group]['global_descriptor'][:] for group in db_data.keys() if 'global_descriptor' in db_data[group]])
    query_global_descriptors = np.array([query_db_data[group]['global_descriptor'][:] for group in query_db_data.keys() if 'global_descriptor' in query_db_data[group]])

    query_image_names = [group for group in query_db_data.keys() if 'global_descriptor' in query_db_data[group]]

    db_image_names = [group for group in db_data.keys() if 'global_descriptor' in db_data[group]]
      
    true_count = 0
    false_count = 0
    false_query_images = []
    
    for query_index in range(len(query_global_descriptors)):
        query_descriptor = query_global_descriptors[query_index]
        most_similar_image_idx, similarity = nearest_neighbor_search(query_descriptor, db_global_descriptors, query_image_names, db_image_names)
        
        query_image_name = query_image_names[query_index]
        db_image_name = db_image_names[most_similar_image_idx]
        query_image_id = int(''.join(filter(str.isdigit, query_image_name)))
        image_id = int(''.join(filter(str.isdigit, db_image_name)))
        id_difference = abs(query_image_id - image_id)
        is_similar = id_difference == 1

        if not is_similar:
            false_query_images.append(query_image_name)
        if is_similar:
            true_count += 1
        else:
            false_count += 1
    print(false_query_images)

    db_data.close()
    query_db_data.close()

    total_count = true_count + false_count
    true_percentage = (true_count / total_count) * 100
    false_percentage = (false_count / total_count) * 100

    true_count_hm = histogram_matching(dataset, false_query_images, db_image_names)
    false_count_hm = total_count - true_count_hm - true_count
    final_true_count = true_count + true_count_hm
    final_false_count = false_count_hm

    final_true_percentage = (final_true_count / (final_true_count + final_false_count)) * 100
    final_false_percentage = (final_false_count / (final_true_count + final_false_count)) * 100
    print(f"Final True 개수: {final_true_count}")
    print(f"Final False 개수: {final_false_count}")

    print(f"Final True 비율: {final_true_percentage:.2f}%")
    
    print(f"{true_percentage:.2f}% 에서 {final_true_percentage:.2f}만큼 정확도 상승")

def histogram_matching(dataset, false_query_images, db_image_names):
    # 데이터베이스 파일 경로 설정
    database_file = "/home/ubuntu/cw/Hierarchical-Localization/outputs/integrate/refer_combined_final.db"
    query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/outputs/integrate/query_combined_final.db"

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

    # 결과 초기화
    true_count = 0
    false_count = 0

    # 각 쿼리 이미지에 대한 히스토그램 매칭 실행
    for query_image_filename in false_query_images:
        # 쿼리 이미지 DB에서 쿼리 이미지 정보 가져오기
        query_cursor.execute("SELECT filename, histogram FROM images WHERE filename=?", (query_image_filename,))
        query_image = query_cursor.fetchone()

        if query_image is not None:
            query_filename, query_histogram_str = query_image

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

            # ID 차이 계산
            id_difference = abs(query_image_id - most_similar_image_id)

            # 결과 출력
            is_similar = id_difference == 1

            # True 및 False 카운트 업데이트
            if is_similar:
                true_count += 1
            else:
                false_count += 1

    # 데이터베이스 연결 종료
    conn.close()

    return true_count

if __name__ == "__main__":
    main()
