# NN Search로 레퍼런스/쿼리 DB 비교하는 코드 (Hloc 단독 정확도 출력)
# DB에 있는 연속된 ID 기준
# True/False로 strict 검증 추가
import h5py
import numpy as np
from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization

def nearest_neighbor_search(query_descriptor, db_global_descriptors, query_image_names, db_image_names):
    # 각 쿼리 이미지의 global descriptor와 DB의 global descriptor 간의 L2 거리 계산
    distances = np.linalg.norm(db_global_descriptors - query_descriptor, axis=1)
    # L2 거리가 가장 작은 이미지의 인덱스 찾기
    most_similar_image_idx = np.argmin(distances)
    # 가장 유사한 이미지와 거리 반환
    return most_similar_image_idx, distances[most_similar_image_idx]

def main():
    # dataset = Path('/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur')
    dataset = Path('/home/ubuntu/cw/data/nuScenes/asia/split')
    # images = dataset / 'mapping_origin/'
    # images = dataset / '300ms_reference/'
    # images = dataset / '500ms_reference/'
    # images = dataset / '500ms_reference2/'
    # images = dataset / '300ms_reference_random2/'
    images = dataset / 'Refer_CAM_BACK_SPLIT_3000/'

    # images_query = dataset / 'mapping_query/'
    # images_query = dataset / '300ms_query/'
    # images_query = dataset / '500ms_query/'
    # images_query = dataset / '500ms_query2/'
    # images_query = dataset / '300ms_query_random2/'
    images_query = dataset / 'Query_CAM_BACK_SPLIT_3000/'

    outputs = Path('/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur')
    sfm_pairs = outputs / 'pairs-factory.txt'
    loc_pairs = outputs / 'pairs-query-factory.txt'
    print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
    print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    # Database 파일과 Query 파일을 생성하고 읽기
    # database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/origin-global-feats-netvlad.h5'
    # database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/300ms-reference-feats-netvlad.h5'
    # database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/500ms-reference-feats-netvlad.h5'
    # database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/500ms-2-reference-feats-netvlad.h5'
    # database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/300ms-2-random-reference-feats-netvlad.h5'
    database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/open-reference-3000-feats-netvlad.h5'

    # query_database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/query-global-feats-netvlad.h5'
    # query_database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/300ms-query-feats-netvlad.h5'
    # query_database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/500ms-query-feats-netvlad.h5'
    # query_database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/500ms-2-query-feats-netvlad.h5'
    query_database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/open-query-3000-feats-netvlad.h5'

    db_data = h5py.File(database_file, 'r')
    query_db_data = h5py.File(query_database_file, 'r')

    # DB와 Query 이미지의 global descriptor를 가져오기
    db_global_descriptors = np.array([db_data[group]['global_descriptor'][:] for group in db_data.keys() if 'global_descriptor' in db_data[group]])
    query_global_descriptors = np.array([query_db_data[group]['global_descriptor'][:] for group in query_db_data.keys() if 'global_descriptor' in query_db_data[group]])

    # Query 이미지 파일명 가져오기
    query_image_names = [group for group in query_db_data.keys() if 'global_descriptor' in query_db_data[group]]

    # DB 이미지 파일명 가져오기
    db_image_names = [group for group in db_data.keys() if 'global_descriptor' in db_data[group]]
    # 변수 초기화
    true_count = 0
    false_count = 0
    # 각 쿼리 이미지에 대한 NN 검색 수행
    for query_index in range(len(query_global_descriptors)):
        query_descriptor = query_global_descriptors[query_index]
        most_similar_image_idx, similarity = nearest_neighbor_search(query_descriptor, db_global_descriptors, query_image_names, db_image_names)
        
        # Query 이미지 파일명과 DB 이미지 파일명, 결과 출력
        query_image_name = query_image_names[query_index]
        db_image_name = db_image_names[most_similar_image_idx]
        query_image_id = int(''.join(filter(str.isdigit, query_image_name)))
        image_id = int(''.join(filter(str.isdigit, db_image_name)))
        id_difference = abs(query_image_id - image_id)
        is_similar = id_difference == 1

                # True 및 False 카운트 업데이트
        if is_similar:
            true_count += 1
        else:
            false_count += 1

        print(f'Query 이미지 {query_image_name}의 가장 유사한 이미지는 DB 이미지 {db_image_name} (유사도: {similarity:.4f})')

    # 파일 닫기
    db_data.close()
    query_db_data.close()

    # 결과 출력
    total_count = true_count + false_count
    true_percentage = (true_count / total_count) * 100
    false_percentage = (false_count / total_count) * 100
    print(f"전체 중 True 개수: {true_count}")
    print(f"전체 중 False 개수: {false_count}")
    print(f"전체 중 True 비율: {true_percentage:.2f}%")
    print(f"전체 중 False 비율: {false_percentage:.2f}%")
if __name__ == "__main__":
    main()
