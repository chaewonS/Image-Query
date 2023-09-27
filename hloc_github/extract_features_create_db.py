from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization

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

    # 같은 이름으로 생성되므로 한 번씩 실행해야 함 (추후 수정 필요)
    # DB 이미지에서 global descriptor 추출
    # db_global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    # Query 이미지에서 global descriptor 추출
    db_global_descriptors = extract_features.main(retrieval_conf, images_query, outputs)

if __name__ == "__main__":
    main()

