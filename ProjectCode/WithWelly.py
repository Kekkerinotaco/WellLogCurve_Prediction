import os
from tensorflow import keras
import pickle
import sklearn
import welly
import pandas as pd
from welly import Curve


def main():
    """Starts program execution"""
    transformer_path = r""
    model_path = r""
    folder_with_files = r""
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
    result_folder_path = os.path.join(folder_with_files, "!_ResultFiles")
    manage_folder_existence(result_folder_path)
    for root, folders, files in os.walk(folder_with_files):
        for file in files:
            if file.upper().endswith(".LAS"):
                file_path = os.path.join(root, file)
                processed_file = process_file(file_path, transformer, model)
                processed_file.to_las(os.path.join(result_folder_path, file))


def manage_folder_existence(folder_path):
    """Created a folder, if it doesn't exist already"""
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass


def process_file(file_path, transformer, model):
    """Based on .las file located in file_path, creates result .las file, containing predicted curve data"""
    project = welly.read_las(file_path)
    well = project[0]
    gr_curve = well.data["GK"].df
    bk_curve = well.data["BK"].df
    data_for_prediction = pd.concat([gr_curve, bk_curve], axis=1)
    X_transformed = transformer.transform(data_for_prediction)
    NKTD_curve = model.predict(X_transformed)
    params = {"mnemonic": "NKTD_predicted"}
    c = Curve(NKTD_curve, index=well.data["GK"].df.index, **params)
    well.data["NKTD_pred"] = c
    return well


if __name__ == "__main__":
    main()
