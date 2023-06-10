import lasio
import numpy as np
from tensorflow import keras
import sklearn
import pickle
import welly
import pandas as pd
# import lasio
from welly import Curve

transformer_path = r"C:\Users\USER\Desktop\NKTD_pred\02.Models\run_2023_02_11-17_32_39\ColumnTransformer.pkl"

with open(transformer_path, "rb") as file:
    transformer = pickle.load(file)

model_path = r"C:\Users\USER\Desktop\NKTD_pred\02.Models\run_2023_02_11-17_32_39\xgb_reg.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

las_file_path = r"C:\Users\USER\Desktop\NKTD_pred\00.ExampleFiles\15R_V_S_continuous.las"

las = lasio.read(las_file_path)
print(las.curves["DEPT"])
depth_data = las.curves["DEPT"]
gk_curve = pd.Series(las.curves["GK"].data)
bk_curve = pd.Series(las.curves["BK"].data)

data_for_prediction = pd.concat((gk_curve, bk_curve), axis=1)
print(data_for_prediction)
data_for_prediction.columns = ["GK", "BK"]
X_transformed = transformer.transform(data_for_prediction)

NKTD_curve = model.predict(X_transformed)

print(NKTD_curve)

las.add_curve("NKTD_predicted", NKTD_curve)
result_path = r"C:\Users\USER\Desktop\NKTD_pred\00.ExampleFiles\15R_V_S_continuous_NKTD.las"
las.write(result_path)
