from tensorflow import keras
import pickle
import sklearn
import welly
import pandas as pd
# import lasio
from welly import Curve

transformer_path = r"path_to_transformer"

with open(transformer_path, "rb") as file:
    transformer = pickle.load(file)

model_path = r"path_to_model"
with open(model_path, "rb") as file:
    model = pickle.load(file)

las_file_path = r"path_to_a_las_file"

project = welly.read_las(las_file_path)
well = project[0]

gr_curve = well.data["GK"].df
bk_curve = well.data["BK"].df

print(gr_curve)

print(bk_curve)
data_for_prediction = pd.concat([gr_curve, bk_curve], axis=1)

print(data_for_prediction)

X_transformed = transformer.transform(data_for_prediction)

print(X_transformed)

NKTD_curve = model.predict(X_transformed)

params = {"mnemonic": "NKTD_predicted"}

c = Curve(NKTD_curve, index=well.data["GK"].df.index, **params)

print(c)

well.data["NKTD_pred"] = c

print(well.data)

out_path = r"../../02.OtherNKTDProject/Processed.las"
well.to_las(out_path)
