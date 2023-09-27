import h5py
import numpy as np
from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization

def nearest_neighbor_search(query_descriptor, db_global_descriptors):
    # 각 쿼리 이미지의 global descriptor와 DB의 global descriptor 간의 L2 거리 계산
    distances = np.linalg.norm(db_global_descriptors - query_descriptor, axis=1)
    
    # L2 거리가 가장 작은 이미지의 인덱스 찾기
    most_similar_image_idx = np.argmin(distances)
    
    # 가장 유사한 이미지와 거리 반환
    return most_similar_image_idx, distances[most_similar_image_idx]

def main():
    dataset = Path('/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur')
    images = dataset / 'mapping_origin/'
    images_query = dataset / 'mapping_query/'

    outputs = Path('/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur')
    sfm_pairs = outputs / 'pairs-factory.txt'
    loc_pairs = outputs / 'pairs-query-factory.txt'
    print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
    print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    # Database 파일과 Query 파일을 생성하고 읽기
    database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/global-feats-netvlad.h5'
    query_database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/query-feats-netvlad.h5'
    
    db_data = h5py.File(database_file, 'r')
    query_db_data = h5py.File(query_database_file, 'r')

    # DB와 Query 이미지의 global descriptor를 가져오기
    db_global_descriptors = np.array([db_data[group]['global_descriptor'][:] for group in db_data.keys() if 'global_descriptor' in db_data[group]])
    query_global_descriptors = np.array([query_db_data[group]['global_descriptor'][:] for group in query_db_data.keys() if 'global_descriptor' in query_db_data[group]])

    # 각 쿼리 이미지에 대한 NN 검색 수행
    for query_index in range(len(query_global_descriptors)):
        query_descriptor = query_global_descriptors[query_index]
        most_similar_image_idx, similarity = nearest_neighbor_search(query_descriptor, db_global_descriptors)
        
        # 결과 출력
        print(f'Query 이미지 {query_index + 1}의 가장 유사한 이미지는 DB 이미지 {most_similar_image_idx + 1} (유사도: {similarity:.4f}) 입니다.')

    # 파일 닫기
    db_data.close()
    query_db_data.close()

if __name__ == "__main__":
    main()
