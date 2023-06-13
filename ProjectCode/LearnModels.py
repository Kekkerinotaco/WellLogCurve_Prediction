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

from ProjectCode import ShowStatistics

date_time_str = time.strftime("run_%Y_%m_%d-%H_%M_%S")
def main(folder_with_files, needed_columns, target_column):
    # folder_with_files = r""
    # needed_columns = ["GK", "BK", "NKTD"]
    # csv_path = os.path.join(folder_with_files, "!_LearningData.csv")
    start = time.time()
    # learning_data = pd.read_csv(csv_path, index_col=0)
    learning_data = get_learning_data_from_lases(folder_with_files, needed_columns)
    # print(learning_data)

    # Возможно весь этот блок бесполезен, или нужен только при загрузке датасета из .csv, но и там похоже решается
    # указанием колонки-индекса
    try:
        learning_data.index = learning_data["DEPT"]
        learning_data.drop(columns="DEPT", inplace=True)
    except KeyError:
        try:
            learning_data.index = learning_data["DEPTH"]
            learning_data.drop(columns="DEPTH", inplace=True)
        except:
            pass

    learning_data = preprocess_data(learning_data)
    learning_data = learning_data.head(1000)
    X_train, X_test, y_train, y_test, full_pipeline = transform_data(learning_data, target_column=target_column,
                                                                     add_log=True, add_exp=True, add_sqrt=True)

    print(f"Data loading time: {time.time() - start}")
    prediction_model = train_xgb_model(X_train, X_test, y_train, y_test)
    return full_pipeline, prediction_model


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


def get_learning_data_from_lases(folder_path, needed_curves):
    summary_data = None
    for root, folders, files in os.walk(folder_path):
        for file in files:
            if file.upper().endswith(".LAS"):
                print(file)
                las_file_path = os.path.join(root, file)
                if summary_data is None:
                    summary_data = get_las_data(las_file_path, needed_curves)
                else:
                    current_file_data = get_las_data(las_file_path, needed_curves)
                    summary_data = pd.concat([summary_data, current_file_data], axis=0)
    return summary_data


def get_las_data(las_file_path, needed_curves):
    las = lasio.read(las_file_path)
    try:
        las_data = las.df()[needed_curves]
        las_data.dropna(inplace=True)
        print(las_data)
        return las_data
    except KeyError as e:
        print(f"Error {e} with file {os.path.basename(las_file_path)}")


def preprocess_data(learning_data):
    learning_columns = ["GK", "BK"]
    # print(learning_data)
    data_dropped_outliers = drop_outliers(learning_data, learning_columns, 3)
    data_dropped_correlations = drop_corr(data_dropped_outliers, corr_coef=0.8)
    # print(data_dropped_outliers)
    # print(data_dropped_outliers.shape)
    return data_dropped_outliers


def drop_outliers(df, columns_to_clear, n_of_std_away):
    start_shape = df.shape
    for column in columns_to_clear:
        column_mean = df[column].mean()
        column_STD = df[column].std()
        df["n_std_away"] = np.abs((df[column] - column_mean) / column_STD)
        df = df[df["n_std_away"] < float(n_of_std_away)]
    df = df.drop(columns="n_std_away")
    result_shape = df.shape
    result_string = "Initial df shape: {}, \n Result df shape: {}, \n N of dropped examples: {}".format(start_shape,
                                                                                                        result_shape,
                                                                                                        start_shape[0] - result_shape[0])
    print(result_string)
    return df


def drop_corr(df, corr_coef):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_coef)]
    df = df.drop(columns = to_drop)
    return df


def add_features(X, y, add_log=True, add_exp=True, add_sqrt=True):
    start_columns = X.columns
    exp = np.exp(X)
    # X.isin([np.Inf, np.NINF]).any(axis=1)
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

    xgb_grid_search = RandomizedSearchCV(XGBRegressor(), xgb_grid_params, cv=5, scoring="r2", n_iter=8, n_jobs=-1)
    print("Started to train XGBRegressor")
    xgb_grid_search.fit(X_train, y_train)
    xgb_reg = xgb_grid_search.best_estimator_
    y_hat = xgb_reg.predict(X_test)
    ShowStatistics.main(y_test, y_hat, date_time_str)
    # np.savetxt("./Yhat.csv", y_hat)
    # y_hat.to_csv()
    # y_test.iloc[:, 1].to_csv("./Ytest.csv")
    save_model(xgb_reg, "XGBRegressionModel.pkl")
    return xgb_reg


# def show_statistics(y_test, y_hat):
#     global date_time_str
#     folder_for_statistics = os.path.join(os.path.curdir, "01.PredictionQualityCheck")
#     ensure_folder_existance(folder_for_statistics)
#     folder_for_current_run_statistics = os.path.join(folder_for_statistics, date_time_str)
#     ensure_folder_existance(folder_for_current_run_statistics)
#     median_test_value = y_test.median()
#     fig = plt.scatter(y_test, y_hat)
#     fig_path = os.path.join(folder_for_current_run_statistics, "01.Pred_VS_True.jpg")
#     fig_copy = fig.get_figure()
#     fig_copy.savefig(fig_path, dpi=500)
#     plt.clf()
#     print(f"Type Ytest: {type(y_test)}")
#     print(f"Type Yhat: {type(y_hat)}")
#     print(r2_score(y_test, y_hat))
#     errors = y_test - y_hat
#     median_error = round(sum(errors) / len(errors), 5)
#     plt.hist(errors, bins=150)
#     fig_path = os.path.join(folder_for_current_run_statistics, "02.ErrorsHist.jpg")
#     # fig_copy = fig.get_figure()
#     plt.savefig(fig_path, dpi=500)
#     median_percent_error = round((median_error * 100 / median_test_value), 3)
#     text_data_filepath = os.path.join(folder_for_current_run_statistics, "Stats.txt")
#     stat_string = f"Median NKTD value: {median_test_value}, \n meadian prediction error: {median_error} or {median_percent_error}%"
#     with open(text_data_filepath, "w") as file:
#         file.write(stat_string)


def save_model(model, model_name):
    global date_time_str
    curr_folder_path = os.path.abspath(os.path.curdir)
    models_folder = os.path.join(curr_folder_path, "Models")
    current_run_folder = os.path.join(models_folder, date_time_str)
    ensure_folder_existance(models_folder)
    ensure_folder_existance(current_run_folder)
    saved_model_path = os.path.join(current_run_folder, model_name)
    with open(saved_model_path, "wb") as file:
        pickle.dump(model, file)

    print(models_folder)


def ensure_folder_existance(folder_path):
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print(f"Folder already exists: \n {folder_path}")


if __name__ == '__main__':
    main()

