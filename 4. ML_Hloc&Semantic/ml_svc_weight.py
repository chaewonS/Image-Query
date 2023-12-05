from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import h5py
import numpy as np
from pathlib import Path
from pprint import pformat
from hloc import extract_features, match_features
import sqlite3
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
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

import sqlite3

def prepare_data(db_data, query_db_data, hm_database_file, hm_query_database_file, query_image_names, db_global_descriptors, query_global_descriptors):
    features = []
    labels = []

    def get_histogram_from_db(database_file, image_name):
        with sqlite3.connect(database_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT histogram FROM images WHERE filename=?", (image_name,))
            result = cursor.fetchone()
            return np.array(result[0].split(), dtype=int) if result else np.array([])

    for query_index, query_name in enumerate(query_image_names):
        query_descriptor = query_global_descriptors[query_index]
        most_similar_image_idx, _ = nearest_neighbor_search(query_descriptor, db_global_descriptors)

        db_image_name = list(db_data.keys())[most_similar_image_idx]
        hloc_id_difference = abs(int(''.join(filter(str.isdigit, query_name))) - 
                                 int(''.join(filter(str.isdigit, db_image_name))))
        hloc_label = 1 if hloc_id_difference <= 1 else 0

        _, _, id_difference = histogram_matching(hm_database_file, hm_query_database_file, [query_name])
        histogram_label = 1 if id_difference <= 1 else 0
        histogram_array = get_histogram_from_db(hm_query_database_file, query_name)

        hloc_descriptor = db_global_descriptors[most_similar_image_idx]
        feature_vector = np.concatenate((hloc_descriptor, histogram_array))
        features.append(feature_vector)
        labels.append(1 if hloc_label or histogram_label else 0)

    return np.array(features), np.arrayW(labels)


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

    hm_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference.db"
    hm_query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query.db"

    true_count_hm, false_count_hm, _ = histogram_matching(hm_database_file, hm_query_database_file, query_image_names)
    
    # Hloc 결과에 대한 평가
    hloc_accuracies = []
    for query_index in range(len(query_global_descriptors)):
        query_descriptor = query_global_descriptors[query_index]
        most_similar_image_idx, _ = nearest_neighbor_search(query_descriptor, db_global_descriptors)

        query_image_name = query_image_names[query_index]
        db_image_name = list(db_data.keys())[most_similar_image_idx]
        hloc_id_difference = abs(int(''.join(filter(str.isdigit, query_image_name))) - 
                                 int(''.join(filter(str.isdigit, db_image_name))))
        hloc_accuracies.append(hloc_id_difference <= 1)

    hloc_accuracy = np.mean(hloc_accuracies) * 100

    # 히스토그램 매칭 결과에 대한 평가
    histogram_accuracies = []
    for query_image_name in query_image_names:
        _, _, id_difference = histogram_matching(hm_database_file, hm_query_database_file, [query_image_name])
        histogram_accuracies.append(id_difference <= 1)

    histogram_accuracy = np.mean(histogram_accuracies) * 100

    # 머신러닝 모델 결과에 대한 평가
    X, y = prepare_data(db_data, query_db_data, hm_database_file, hm_query_database_file, query_image_names, db_global_descriptors, query_global_descriptors)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    # model = SVC(C="0.1", gamma="1", kernel="rbf")
    model = grid.best_estimator_
    predictions = model.predict(X_test)
    ml_model_accuracy = accuracy_score(y_test, predictions) * 100

    print(f"Hloc Accuracy: {hloc_accuracy:.2f}%")
    print(f"Histogram Matching Accuracy: {histogram_accuracy:.2f}%")
    print(f"Combined Model Accuracy: {ml_model_accuracy:.2f}%")
    improvement = ml_model_accuracy - max(hloc_accuracy, histogram_accuracy)
    print(f"Improvement over individual methods: {improvement:.2f}%")

if __name__ == "__main__":
    main()
