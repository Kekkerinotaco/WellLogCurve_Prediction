import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


def main(data, date_time_string, learning_curves, target_curve):
    # data = pd.read_csv(data_path, index_col=0)
    statistic_folder = os.path.join(os.path.curdir, "01.InitialDataStatistics")
    ensure_folder_existance(statistic_folder)
    current_run_statistic_folder = os.path.join(statistic_folder, date_time_string)
    ensure_folder_existance(current_run_statistic_folder)

    # If the df is too large, it will take a while to make plots,
    # so we need to sample only a part of it
    sample_number = 5000
    if len(data) > sample_number:
        data = data.sample(sample_number)
    save_txt_stats(data, current_run_statistic_folder)
    save_corrs_with_target(data, current_run_statistic_folder, learning_curves, target_curve)
    plot_df_histograms(data, current_run_statistic_folder)


def plot_df_histograms(data, folder_for_statistics):
    columns_counter = 0
    for column in data.columns:
        print(f"Plotting the distribution histogram for curve: {column}")
        columns_counter += 1
        result_graph_name = os.path.join(folder_for_statistics, f"{columns_counter}.{column}_statistics.png")
        # fig = sns.histplot(data=data, x=column)
        plt.savefig(result_graph_name)
        plt.clf()
        plt.close()


def ensure_folder_existence(folder_path):
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print(f"Folder already exists: \n {folder_path}")


def save_txt_stats(data, folder_for_statistics):
    print(f"Saving a txt describing data statistics")
    result_file_path = os.path.join(folder_for_statistics, "00.DataStats.csv")
    data.describe().to_csv(result_file_path, float_format='%.2f')


def save_corrs_with_target(data, result_folder, learning_curves, target_curve):
    print(f"Saving a correlation between learning curves and the target curve")
    result_file_path = os.path.join(result_folder, "00.CorrelationsWithTarget.png")
    sns.pairplot(data, x_vars=learning_curves, y_vars=target_curve, diag_kind="hist")
    plt.savefig(result_file_path)
    plt.clf()


def ensure_folder_existance(folder_path):
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print(f"Folder already exists: \n {folder_path}")


if __name__ == "__main__":
    data_path = r"/Users/gymoroz/Desktop/LasCurvesPredictionData/!_LearningData.csv"
    data = pd.read_csv(data_path, index_col=0)
    folder_for_statistics = r"/Users/gymoroz/Desktop/Statistics"
    learning_curves = "GK", "BK"
    target_curve = "NKTD"
    main(data, folder_for_statistics, learning_curves, target_curve)