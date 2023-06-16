import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import pandas as pd


def main(y_test, y_hat, date_time_str):

    statistic_folder = os.path.join(os.path.curdir, "03.PredictionQualityCheck")
    ensure_folder_existance(statistic_folder)
    current_run_statistic_folder = os.path.join(statistic_folder, date_time_str)
    ensure_folder_existance(current_run_statistic_folder)

    median_test_value, errors = get_stat_data(y_test, y_hat)

    plot_TrueVsPred(y_test, y_hat, current_run_statistic_folder)
    plot_ErrorsHist(errors, current_run_statistic_folder)
    white_stat_txt(errors, median_test_value, current_run_statistic_folder)


def ensure_folder_existance(folder_path):
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print(f"Folder already exists: \n {folder_path}")


def get_stat_data(y_test, y_hat):
    median_test_value = y_test.median()
    errors = y_test - y_hat
    return median_test_value, errors


def plot_TrueVsPred(y_test, y_hat, current_run_statistic_folder):
    fig = plt.scatter(y_test, y_hat)
    fig_path = os.path.join(current_run_statistic_folder, "01.Pred_VS_True.jpg")
    fig_copy = fig.get_figure()
    fig_copy.savefig(fig_path, dpi=500)
    plt.clf()


def plot_ErrorsHist(errors, current_run_statistic_folder):
    plt.hist(errors, bins=150)
    fig_path = os.path.join(current_run_statistic_folder, "02.ErrorsHist.jpg")
    plt.savefig(fig_path, dpi=500)
    plt.clf()


def white_stat_txt(errors, median_test_value, current_run_statistic_folder):
    median_error = round(sum(errors) / len(errors), 5)
    median_percent_error = round((median_error * 100 / median_test_value), 3)
    text_data_filepath = os.path.join(current_run_statistic_folder, "Stats.txt")
    stat_string = f"Median NKTD value: {median_test_value}, \n meadian prediction error: {median_error} or {median_percent_error}%"
    with open(text_data_filepath, "w") as file:
        file.write(stat_string)


if __name__ == "__main__":
    main()