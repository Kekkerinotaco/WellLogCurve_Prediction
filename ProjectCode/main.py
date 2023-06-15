import LearnModels
import ProcessLasFiles


# GK BK -> NKTD
learning_lases_path = r"/Users/gymoroz/Desktop/LasCurvesPredictionData/02.WorkingDirectory/ToLearn"
prediction_lases_path = r"/Users/gymoroz/Desktop/LasCurvesPredictionData/02.WorkingDirectory/ToPredict"
learning_columns = ["GK", "BK"]
target_column = "NKTD"

# Dima pred 1
# learning_lases_path = r"/Users/gymoroz/Desktop/DimaTestFiles/02.WorkingFolder/Обучить"
# prediction_lases_path = r"/Users/gymoroz/Desktop/DimaTestFiles/02.WorkingFolder/здесь надо сгенерить"
# learning_columns = ["Poro", "Swat", "PERMEABILITY_CORE"]
# target_column = "GK"


# Dima Pred 2
# learning_lases_path = r"/Users/gymoroz/Desktop/Rigis->GIS/Learn"
# prediction_lases_path = r"/Users/gymoroz/Desktop/Rigis->GIS/Predict"
# learning_columns = ["Poro", "Prob_COLL"]
# target_column = "GK"

learning_columns = [learning_column.upper() for learning_column in learning_columns]
target_column = target_column.upper()
learning_columns_copy = learning_columns.copy()
transformer, model = LearnModels.main(learning_lases_path, learning_columns, target_column)

print("Started to predict")
ProcessLasFiles.process_las_files(prediction_lases_path, transformer, model, learning_columns_copy, target_column)
print("Program execution is finished")
