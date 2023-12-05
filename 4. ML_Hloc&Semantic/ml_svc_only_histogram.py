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

    results = []

    for query_image_filename in query_image_filenames:
        query_cursor.execute("SELECT filename, histogram FROM images WHERE filename=?", (query_image_filename,))
        query_image = query_cursor.fetchone()

        if query_image is not None:
            query_histogram_str = query_image[1]
            query_image_id = int(''.join(filter(str.isdigit, query_image[0])))
            min_difference = float('inf')
            most_similar_histogram = None

            cursor.execute("SELECT filename, histogram FROM images")
            images = cursor.fetchall()

            for filename, histogram_str in images:
                difference = histogram_difference(query_histogram_str, histogram_str)
                image_id = int(''.join(filter(str.isdigit, filename)))
                if difference < min_difference:
                    min_difference = difference
                    most_similar_histogram = histogram_str
                    most_similar_image_id = image_id

            id_difference = abs(query_image_id - most_similar_image_id)
            results.append((most_similar_histogram, id_difference))

    conn.close()
    return results

def prepare_data(hm_database_file, hm_query_database_file, query_image_names):
    matching_results = histogram_matching(hm_database_file, hm_query_database_file, query_image_names)
    features = []
    labels = []
    for hist_str, id_diff in matching_results:
        if hist_str:  # 히스토그램 데이터가 있는 경우에만 처리
            histogram_array = np.array(hist_str.split(), dtype=int)
            features.append(histogram_array)
            labels.append(1 if id_diff <= 1 else 0)

    return np.array(features), np.array(labels)


def get_query_image_names(query_database_file):
    conn = sqlite3.connect(query_database_file)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM images")
    query_image_names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return query_image_names

def main():
    dataset = Path('/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur')
    images_query = dataset / 'bag_query/'
    outputs = Path('/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur')

    hm_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference.db"
    hm_query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query.db"
    
    query_image_names = get_query_image_names(hm_query_database_file)
    
    matching_results = histogram_matching(hm_database_file, hm_query_database_file, query_image_names)

    histogram_accuracies = []
    for hist_str, id_diff in matching_results:
        histogram_accuracies.append(id_diff <= 1)
    histogram_accuracy = np.mean(histogram_accuracies) * 100

    X, y = prepare_data(hm_database_file, hm_query_database_file, query_image_names)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    # model = SVC()
    # model.fit(X_train, y_train)
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    model = grid.best_estimator_
    predictions = model.predict(X_test)
    ml_model_accuracy = accuracy_score(y_test, predictions) * 100
    
    print(f"Histogram Matching Accuracy: {histogram_accuracy:.2f}%")
    print(f"ML Model Accuracy: {ml_model_accuracy:.2f}%")
    improvement = ml_model_accuracy - histogram_accuracy
    print(f"Improvement over individual methods: {improvement:.2f}%")

if __name__ == "__main__":
    main()
