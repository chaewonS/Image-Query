# Image-Query
Visual Localization

- 기존 Hloc 알고리즘의 한계 극복
    - feature points에 의존
- Semantic Segmentation 정보 활용해 성능 향상
    - segmentation 정보 기반 term 추가
- Histogram Matching 성능 평가 방식
    - https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
    - 각 클래스에 해당하는 픽셀의 수를 히스토그램으로 표현 (only stuff class)
    - Semantic 영역에 대한 히스토그램 생성, 코사인 거리 계산
    - Query Image와 유사한 위치에서 찍힌 이미지를 찾아내는 것이 목표 (이미지 ID 출력)

---  

- DB 생성
    - create_seg_db.py
    - create_seg_query.py
- Histogram 시각화
    - visualize_histogram.py
- Histogram DB 파일로 Image Query
    - compare_histogram.py
      
