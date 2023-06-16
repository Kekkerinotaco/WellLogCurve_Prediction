import TrainModels
import MakeTargetCurvePredictions


# GK BK -> NKTD
learning_lases_path = r"/Users/gymoroz/Desktop/LasCurvesPredictionData/02.WorkingDirectory/01.Learn"
folder_with_test_files = r"/Users/gymoroz/Desktop/LasCurvesPredictionData/02.WorkingDirectory/02.Test"
prediction_lases_path = r"/Users/gymoroz/Desktop/LasCurvesPredictionData/02.WorkingDirectory/03.ToPredict"
learning_columns = ["GK", "BK"]
target_column = "NKTD"

# Files set 1
# learning_lases_path = r"/Users/gymoroz/Desktop/DimaTestFiles/02.WorkingFolder/Обучить"
# prediction_lases_path = r"/Users/gymoroz/Desktop/DimaTestFiles/02.WorkingFolder/здесь надо сгенерить"
# learning_columns = ["Poro", "Swat", "PERMEABILITY_CORE"]
# target_column = "GK"


# Files set 2
# learning_lases_path = r"/Users/gymoroz/Desktop/Rigis->GIS/Learn"
# prediction_lases_path = r"/Users/gymoroz/Desktop/Rigis->GIS/Predict"
# learning_columns = ["Poro", "Prob_COLL"]
# target_column = "GK"

learning_columns = [learning_column.upper() for learning_column in learning_columns]
target_column = target_column.upper()
learning_columns_copy = learning_columns.copy()
transformer, model = TrainModels.main(learning_lases_path, folder_with_test_files, learning_columns, target_column)

print("Started to predict")
MakeTargetCurvePredictions.process_las_files(prediction_lases_path, transformer, model, learning_columns_copy, target_column)
print("Program execution is finished")
