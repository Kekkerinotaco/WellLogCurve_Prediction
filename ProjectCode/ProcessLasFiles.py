import os
# from tensorflow import keras
import pickle
import sklearn
from sklearn.preprocessing import PolynomialFeatures
import welly
import pandas as pd
from welly import Curve
import numpy as np
import lasio

print(sklearn.__version__)


def main():
    """Starts program execution"""
    transformer_path = r"/Users/gymoroz/01.MyFiles/02.Coding/01.CodeForO&G/05.NKTDPrediction/Models/ColumnTransformer.pkl"
    model_path = r"/Users/gymoroz/01.MyFiles/02.Coding/01.CodeForO&G/05.NKTDPrediction/Models/xgb_reg.pkl"
    folder_with_files = r"/Users/gymoroz/Desktop/ЛасФайлыДляТеста/РабочаяПапка"
    transformer = load_model(transformer_path)
    model = load_model(model_path)
    process_las_files(folder_with_files, transformer, model)
    print("Successfully completed")


def load_model(model_path):
    """The function loads the model located at model_path"""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def process_las_files(folder_with_files, transformer, model):
    """Function processes .las files located in folder_with_files
       uses pretrained transformer and model, for making result files"""
    paths_list = make_paths_list(folder_with_files)
    result_folder_path = os.path.join(folder_with_files, "!_ResultFiles")
    manage_folder_existence(result_folder_path)
    for file_path in paths_list:
        print(f"Processing file {os.path.basename(file_path)}")
        las = process_file(file_path, transformer, model)
        las.write(os.path.join(result_folder_path, os.path.basename(file_path)))


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


def process_file(file_path, transformer, model):
    """Based on .las file located in file_path, creates result .las file, containing predicted curve data"""
    las = lasio.read(file_path)
    depth_data = pd.Series(las.curves["DEPT"].data)
    gk_curve = pd.Series(las.curves["GK"].data)
    bk_curve = pd.Series(las.curves["BK"].data)

    initial_data_for_prediction = pd.concat([gk_curve, bk_curve], axis=1)
    initial_data_for_prediction.columns = ["GK", "BK"]
    mask = initial_data_for_prediction.isna().any(axis=1)

    depth_index = depth_data[~mask]
    initial_data_for_prediction = initial_data_for_prediction[~mask]

    data_for_prediction = add_features(initial_data_for_prediction, add_log=True, add_exp=True, add_sqrt=True)
    X_transformed = transformer.transform(data_for_prediction)
    NKTD_curve = model.predict(X_transformed)
    NKTD_curve = pd.DataFrame(NKTD_curve)
    NKTD_curve.columns = ["NKTD_predicted"]
    initial_data_for_prediction.index = depth_index
    NKTD_curve.index = depth_index
    resulting_data = pd.concat([initial_data_for_prediction, NKTD_curve], axis=1)
    resulting_data.index.name = "DEPTH"
    las.set_data_from_df(resulting_data)
    return las

def filter_data_for_prediction(well):
    gr_curve = well.data["GK"].df
    bk_curve = well.data["BK"].df
    data_for_prediction = pd.concat([gr_curve, bk_curve], axis=1)
    mask = data_for_prediction[~(data_for_prediction == -999.25).any(axis=1)]

def add_features(X, add_log=True, add_exp=True, add_sqrt=True):
    start_columns = X.columns
    exp = np.exp(X)
    if add_log:
        log_X_data = np.log(X[start_columns])
        log_X_data.columns = [column_name + "_log" for column_name in X[start_columns]]
        X = pd.concat([X, log_X_data], axis=1)
        X = X[X != np.NINF]
    if add_exp:
        exp_X_data = np.exp(X[start_columns])
        exp_X_data.columns = [column_name + "_exp" for column_name in X[start_columns]]
        X = pd.concat([X, exp_X_data], axis=1)
        X = X[X != np.NINF]
    if add_sqrt:
        sqrt_X_data = np.sqrt(X[start_columns])
        sqrt_X_data.columns = [column_name + "_sqrt" for column_name in X[start_columns]]
        X = pd.concat([X, sqrt_X_data], axis=1)
        X = X[X != np.NINF]
    mask = X.isna().any(axis=1)
    X = X[~mask]
    return X


if __name__ == "__main__":
    main()
