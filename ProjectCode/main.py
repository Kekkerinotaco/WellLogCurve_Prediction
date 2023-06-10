import pickle
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures

import sklearn

print(sklearn.__version__)
project_root_path = r"F:\! Code\08.CalculateNKTD"

transformer_path = os.path.join(project_root_path, "Models", "ColumnTransformer.pkl")
with open(transformer_path, "rb") as file:
    transformer = pickle.load(file)

model_path = os.path.join(project_root_path, "Models", "xgb_reg.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

