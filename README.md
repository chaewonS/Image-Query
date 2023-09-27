# Image-Query
Visual Localization

- 기존 Hloc 알고리즘의 한계 극복
    - feature points에 의존
- Semantic Segmentation 정보 활용해 성능 향상
    - segmentation 정보 기반 term 추가
- 기존 Hloc 알고리즘 성능 테스트
    - 로컬 특징 추출 및 Query/Reference 이미지로 Global Descriptor 정보를 포함한 DB 생성, DB 파일로 이미지 검색 및 매칭 수행
- Semantic Segmentation 성능 테스트
    - weight, config 파일 활용, Query/Reference 이미지로 Semantic segmentation 정보를 포함한 DB 생성, DB 파일로 이미지 검색 및 매칭 수행
- Histogram Matching 성능 평가 방식
    - https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
    - 각 클래스에 해당하는 픽셀의 수를 히스토그램으로 표현 (only stuff class)
    - Semantic 영역에 대한 히스토그램 생성, 코사인 거리 계산
    - Query Image와 유사한 위치에서 찍힌 이미지를 찾아내는 것이 목표 (이미지 ID 출력)
- 정확도 코딩 (성능 평가 방법)
- Query/Reference 이미지 Split 코딩

---  

*Histogram 시각화
- visualize_histogram.py
  
*H5 파일 읽기
- read_h5.py

1. Our Semantic Term
    - pan_FCN 밑에서 실행 (pan_FCN 가상환경)
    - Semantic Segmentation 정보 추가 및 DB 생성
        - create_seg_db.py
        - create_seg_query.py
    - DB 파일로 이미지 검색 및 매칭
        - compare_db.py

3. Github Hloc Term
    - Hierarchical-Localization 밑에서 실행 (Hloc 가상환경)
    - 로컬 특징 추출 및 DB 생성
        - extract_features_create_db.py
    - DB 파일로 이미지 검색 및 매칭
        - netvlad_compare_NN_h5.py
        - netvlad_compare_NN_h5_id.py (ID 출력)
          
---  

![DB_이미지 삽입 가능한지](https://github.com/chaewonS/Image-Query/assets/81732426/c61112ca-746d-4d2f-819f-b4a59ee9370d)
- DB 생성 (filename / Semantic class 픽셀 빈도 수)


![히스토그램 매칭 테스트 결과2](https://github.com/chaewonS/Image-Query/assets/81732426/40ebb5b2-b58a-40ce-8d0a-f40beb83103e)
- Histogram 매칭 테스트 결과
