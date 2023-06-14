import os
import pickle
import sklearn
import pandas as pd
import numpy as np
import lasio


def process_las_files(folder_with_files, transformer, model, columns_for_prediction, target_column_name):
    """Function processes .las files located in folder_with_files
       uses pretrained transformer and model, for making result files"""
    paths_list = make_paths_list(folder_with_files)
    result_folder_path = os.path.join(folder_with_files, "!_ResultFiles")
    manage_folder_existence(result_folder_path)
    for file_path in paths_list:
        print(f"Processing file {os.path.basename(file_path)}")
        las = process_file(file_path, transformer, model, columns_for_prediction, target_column_name)
        las.write(os.path.join(result_folder_path, os.path.basename(file_path)))


def load_model(model_path):
    """The function loads the model located at model_path"""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def make_paths_list(folder_with_files):
    paths_list = []
    for root, folders, files in os.walk(folder_with_files):
        for file in files:
            if file.upper().endswith(".LAS"):
                file_path = os.path.join(root, file)
                paths_list.append(file_path)
    return paths_list


def manage_folder_existence(folder_path):
    """Created a folder, if it doesn't exist already"""
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass


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
    # data_for_prediction.index = depth_index
    predicted_curve.index = depth_index
    resulting_data = pd.concat([all_las_data, predicted_curve], axis=1)
    resulting_data.index.name = "DEPTH"
    print(resulting_data)
    las.set_data_from_df(resulting_data)
    return las


def get_las_data(las_file_path, columns_for_prediction):
    las = lasio.read(las_file_path)
    print(las.df().columns)
    try:
        las_data = las.df()[columns_for_prediction]
        las_data.dropna(inplace=True)
        print(las_data)
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


if __name__ == "__main__":
    folder_with_files = None
    transformer = None
    model = None
    columns_for_prediction = []
    target_column_name = []
    process_las_files(folder_with_files, transformer, model, columns_for_prediction, target_column_name)
