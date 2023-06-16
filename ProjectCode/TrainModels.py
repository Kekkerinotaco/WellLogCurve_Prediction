import lasio
import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from ProjectCode import MakePredictionQualityStats
from sklearn.linear_model import LinearRegression
import PlotInitialDataStatistics
import ProcessTestFiles
import matplotlib.pyplot as plt
date_time_str = time.strftime("run_%Y_%m_%d-%H_%M_%S")


def main(folder_with_learn_files, folder_with_test_files, learning_columns, target_column):
    global date_time_str
    start = time.time()
    learning_data = get_learning_data_from_lases(folder_with_learn_files, learning_columns, target_column)
    learning_data = preprocess_data(learning_data, learning_columns)
    PlotInitialDataStatistics.main(learning_data, date_time_str, learning_columns, target_column)
    print(f"Data loading time: {time.time() - start}")
    print(f"The shape of the learning Data: {learning_data.shape}")
    X_train, X_test, y_train, y_test, full_pipeline = transform_data(learning_data, target_column=target_column,
                                                                     add_log=True, add_exp=True, add_sqrt=True)
    # prediction_model = train_xgb_model(X_train, X_test, y_train, y_test)
    prediction_model = train_LinReg_model(X_train, X_test, y_train, y_test)
    ProcessTestFiles.main(full_pipeline, prediction_model, folder_with_test_files, learning_columns, target_column)
    plot_data_amount_tendency(prediction_model, X_train, y_train, X_test, y_test)
    return full_pipeline, prediction_model


def get_learning_data_from_lases(folder_path, needed_curves, target_column):
    summary_data = None
    needed_curves.append(target_column)
    for root, folders, files in os.walk(folder_path):
        for file in files:
            if file.upper().endswith(".LAS"):
                print(f"Getting the data from file: {file}")
                las_file_path = os.path.join(root, file)
                if summary_data is None:
                    summary_data = get_las_data(las_file_path, needed_curves)
                else:
                    current_file_data = get_las_data(las_file_path, needed_curves)
                    summary_data = pd.concat([summary_data, current_file_data], axis=0)
    return summary_data


def transform_data(learning_data, target_column, add_log=True, add_exp=True, add_sqrt=True):
    X = learning_data.drop(columns=target_column)
    y = learning_data[target_column]
    cat_attribs = []
    num_initial_attribs = X.columns.drop(cat_attribs)
    X, y = add_features(X, y, add_log=True, add_exp=True, add_sqrt=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    num_added_attribs = X_train.columns.drop(cat_attribs).drop(num_initial_attribs)
    num_initial_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly_adder", PolynomialFeatures(degree=2)),
        ("std_scaler", StandardScaler())
    ])
    num_added_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler())
    ])
    full_pipeline = ColumnTransformer([
        ("num_initial", num_initial_pipeline, num_initial_attribs),
        ("num_added", num_added_pipeline, num_added_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])
    X_train = full_pipeline.fit_transform(X_train)
    X_test = full_pipeline.transform(X_test)
    save_model(full_pipeline, "Transformer.pkl")
    return X_train, X_test, y_train, y_test, full_pipeline


def get_las_data(las_file_path, needed_curves):
    las = lasio.read(las_file_path)
    # print(las.df().columns)
    try:
        las_data = las.df()[needed_curves]
        las_data.dropna(inplace=True)
        # print(las_data)
        return las_data
    except KeyError as e:
        print(f"Error {e} with file {os.path.basename(las_file_path)}")


def preprocess_data(learning_data, learning_columns):
    data_dropped_outliers = drop_outliers(learning_data, learning_columns, 3)
    data_dropped_correlations = drop_corr(data_dropped_outliers, corr_coef=0.8)
    return data_dropped_correlations


def drop_outliers(df, columns_to_clear, n_of_std_away):
    start_shape = df.shape
    for column in columns_to_clear:
        column_mean = df[column].mean()
        column_STD = df[column].std()
        df["n_std_away"] = np.abs((df[column] - column_mean) / column_STD)
        df = df[df["n_std_away"] < float(n_of_std_away)]
        # df.loc[:, "n_std_away"] = np.abs((df[column] - column_mean) / column_STD)
    df = df.drop(columns="n_std_away")
    result_shape = df.shape
    result_string = "Initial df shape: {}, \n Result df shape: {}, \n N of dropped examples: {}".format(start_shape,
                                                                                                        result_shape,
                                                                                                        start_shape[0] -
                                                                                                        result_shape[0])
    print(result_string)
    return df


def drop_corr(df, corr_coef):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_coef)]
    df = df.drop(columns=to_drop)
    return df


def add_features(X, y, add_log=True, add_exp=True, add_sqrt=True):
    start_columns = X.columns
    if add_log:
        log_X_data = np.log(X[start_columns])
        log_X_data.columns = [column_name + "_log" for column_name in X[start_columns]]
        X = pd.concat([X, log_X_data], axis=1)
        X = X[X != np.NINF]
        X = X[X != np.Inf]
    if add_exp:
        exp_X_data = np.exp(X[start_columns])
        exp_X_data.columns = [column_name + "_exp" for column_name in X[start_columns]]
        X = pd.concat([X, exp_X_data], axis=1)
        X = X[X != np.NINF]
        X = X[X != np.Inf]

    if add_sqrt:
        sqrt_X_data = np.sqrt(X[start_columns])
        sqrt_X_data.columns = [column_name + "_sqrt" for column_name in X[start_columns]]
        X = pd.concat([X, sqrt_X_data], axis=1)
        X = X[X != np.NINF]
        X = X[X != np.Inf]

    mask = X.isna().any(axis=1)
    X = X[~mask]
    y = y[~mask]
    return X, y


def train_xgb_model(X_train, X_test, y_train, y_test):
    tree_booster_params = {
        "eta": [0.1, 0.3, 0.5, 0.8],
        "gamma": [10, 30, 50, 70, 90],
        "max_depth": [8, 16, 32],

    }
    linear_booster_params = {
        "lambda": [10, 30, 50, 70, 90],
        "alpha": [10, 30, 50, 70, 90],

    }

    xgb_grid_params = [
        tree_booster_params,
        linear_booster_params,
    ]

    xgb_rand_search = RandomizedSearchCV(XGBRegressor(), xgb_grid_params, cv=5, scoring="r2", n_iter=8, n_jobs=-1)
    print("Started to train XGBRegressor")
    xgb_rand_search.fit(X_train, y_train)
    xgb_reg = xgb_rand_search.best_estimator_
    y_hat = xgb_reg.predict(X_test)
    MakePredictionQualityStats.main(y_test, y_hat, date_time_str)
    save_model(xgb_reg, "XGBRegressionModel.pkl")
    print(r2_score(y_hat, y_test))
    return xgb_reg


def train_LinReg_model(X_train, X_test, y_train, y_test):
    lin_reg_model = LinearRegression(n_jobs=-1, positive=True, fit_intercept=True)
    lin_reg_model.fit(X_train, y_train)
    save_model(lin_reg_model, "linearRegressionModel.pkl")
    y_hat = lin_reg_model.predict(X_test)
    print(f"Linear model R2 score: {r2_score(y_hat, y_test)}")
    MakePredictionQualityStats.main(y_test, y_hat, date_time_str)
    return lin_reg_model


def save_model(model, model_name):
    global date_time_str
    curr_folder_path = os.path.abspath(os.path.curdir)
    models_folder = os.path.join(curr_folder_path, "02.Models")
    current_run_folder = os.path.join(models_folder, date_time_str)
    ensure_folder_existence(models_folder)
    ensure_folder_existence(current_run_folder)
    saved_model_path = os.path.join(current_run_folder, model_name)
    with open(saved_model_path, "wb") as file:
        pickle.dump(model, file)

    print(models_folder)


def plot_data_amount_tendency(model, X_train, y_train, X_test, y_test):
    global date_time_str
    print("Plotting DataAmountTendency")
    statistic_folder = os.path.join(os.path.curdir, "01.InitialDataStatistics")
    ensure_folder_existence(statistic_folder)
    current_run_statistic_folder = os.path.join(statistic_folder, date_time_str)
    ensure_folder_existence(current_run_statistic_folder)
    result_file_path = os.path.join(current_run_statistic_folder, "00.DataAmountTendency.png")
    data_length = len(X_train)
    # step = int(data_length / 20)
    step = 20000
    start_point = data_length - step * (data_length // step)
    data_length_list = []
    train_R2_list = []
    test_R2_list = []
    print(type(X_train))
    print(type(y_train))
    for data_len in range(start_point, data_length + 1, step):
        try:
            print("Current_data_lenth: {}".format(data_len))
            # Параметры лучшего классификатора из grid_search
            current_model = model
            learning_chunk = X_train[:data_len]
            target_chunk = y_train.head(data_len)
            current_model.fit(learning_chunk, target_chunk)
            y_hat_train = current_model.predict(learning_chunk)
            y_hat_test = current_model.predict(X_test)
            data_length_list.append(data_len)
            train_R2_list.append(r2_score(target_chunk, y_hat_train))
            test_R2_list.append(r2_score(y_test, y_hat_test))
        except ValueError:
            continue

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(data_length_list, train_R2_list, label="Train_R2", marker="s", linewidth=3)
    ax.plot(data_length_list, test_R2_list, label="Test_R2", marker="x", linewidth=3)

    ax.set_xlabel("Amount_of_training_data")
    ax.set_ylabel("R2 metric")
    ax.legend()
    fig_copy = fig.get_figure()
    fig_copy.savefig(result_file_path)


def ensure_folder_existence(folder_path):
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print(f"Folder already exists: \n {folder_path}")


if __name__ == '__main__':
    main()
