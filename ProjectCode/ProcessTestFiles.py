import os
import welly
from welly.plot import plot_well
import matplotlib.pyplot as plt
import os
import lasio
import pandas as pd
import numpy as np


def main(transformer, model, folder_with_test_files, learning_columns, target_column):
    learning_columns.remove(target_column)
    for root, folders, files in os.walk(folder_with_test_files):
        for file in files:
            if file.upper().endswith(".LAS"):
                file_path = os.path.join(root, file)
                las = process_file(file_path, transformer, model, learning_columns, target_column)
                las.write(os.path.join(file_path))

    folder_for_graphs = os.path.join(folder_with_test_files, "01.CurvesPlots")
    ensure_folder_existence(folder_for_graphs)
    for root, folders, files in os.walk(folder_with_test_files):
        for file in files:
            if file.upper().endswith(".LAS"):
                las_file_path = os.path.join(root, file)
                result_file_path = os.path.join(folder_for_graphs, f"{file}.png")
                well = welly.read_las(las_file_path)[0]
                target_track_names = [target_column, f"{target_column}_PREDICTED"]
                tracks = learning_columns.copy()
                tracks.append(target_track_names)
                fig = plot_well(well, tracks=tracks, extents="curves")
                fig.savefig(result_file_path)


def process_file(file_path, transformer, model, columns_for_prediction, target_column_name):
    """Based on .las file located in file_path, creates result .las file, containing predicted curve data"""
    las = lasio.read(file_path)
    all_las_data = las.df()
    curves_for_prediction = all_las_data[columns_for_prediction]
    depth_data = curves_for_prediction.index
    prediction_curves_exist_mask = curves_for_prediction.isna().any(axis=1)
    depth_index = depth_data[~prediction_curves_exist_mask]
    data_for_prediction = curves_for_prediction[~prediction_curves_exist_mask]
    all_las_data = all_las_data[~prediction_curves_exist_mask]
    depth_index, data_for_prediction, data_for_prediction = add_features(data_for_prediction,
                                                                         depth_index,
                                                                         all_las_data,
                                                                         add_log=True, add_exp=True,
                                                                         add_sqrt=True)
    X_transformed = transformer.transform(data_for_prediction)
    predicted_curve = model.predict(X_transformed)
    predicted_curve = pd.DataFrame(predicted_curve)
    predicted_curve.columns = [f"{target_column_name}_predicted"]
    predicted_curve.index = depth_index
    resulting_data = pd.concat([all_las_data, predicted_curve], axis=1)
    resulting_data.index.name = "DEPTH"
    las.set_data_from_df(resulting_data)
    return las


def get_las_data(las_file_path, needed_curves):
    las = lasio.read(las_file_path)
    try:
        las_data = las.df()[needed_curves]
        las_data.dropna(inplace=True)
        return las_data
    except KeyError as e:
        print(f"Error {e} with file {os.path.basename(las_file_path)}")


def add_features(X, depth_index, initial_data_for_prediction, add_log=True, add_exp=True, add_sqrt=True):
    start_columns = X.columns
    # Надо переписать на формирование единой маски, ее возврат, и потом обрезку всех значений по ней, а не колхоз
    # этот весь
    if add_log:
        log_X_data = np.log(X[start_columns])
        log_X_data.columns = [column_name + "_log" for column_name in X[start_columns]]
        X = pd.concat([X, log_X_data], axis=1)
        mask = X.isin([np.Inf, np.NINF]).any(axis=1)
        X = X[~mask]
        depth_index = depth_index[~mask]
        initial_data_for_prediction = initial_data_for_prediction[~mask]
    if add_exp:
        exp_X_data = np.exp(X[start_columns])
        exp_X_data.columns = [column_name + "_exp" for column_name in X[start_columns]]
        X = pd.concat([X, exp_X_data], axis=1)
        mask = X.isin([np.Inf, np.NINF]).any(axis=1)
        X = X[~mask]
        depth_index = depth_index[~mask]
        initial_data_for_prediction = initial_data_for_prediction[~mask]

    if add_sqrt:
        sqrt_X_data = np.sqrt(X[start_columns])
        sqrt_X_data.columns = [column_name + "_sqrt" for column_name in X[start_columns]]
        X = pd.concat([X, sqrt_X_data], axis=1)
        mask = X.isin([np.Inf, np.NINF]).any(axis=1)

        X = X[~mask]
        depth_index = depth_index[~mask]
        initial_data_for_prediction = initial_data_for_prediction[~mask]

    mask = X.isna().any(axis=1)
    X = X[~mask]
    depth_index = depth_index[~mask]
    initial_data_for_prediction = initial_data_for_prediction[~mask]
    return depth_index, initial_data_for_prediction, X


def ensure_folder_existence(folder_path):
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print(f"Folder already exists: \n {folder_path}")
