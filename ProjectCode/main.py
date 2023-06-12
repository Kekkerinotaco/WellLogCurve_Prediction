import LearnModels
import ProcessLasFiles

print("Started To Train model")
folder_with_lases_to_learn = r"/Users/gymoroz/Desktop/Test1/ToLearn"
needed_columns = ["GK", "BK", "NKTD"]
target_column = "NKTD"
transformer, model = LearnModels.main(folder_with_lases_to_learn, needed_columns, target_column)

# print("Started to predict")
folder_with_lases_to_predict = r"/Users/gymoroz/Desktop/Test1/ToPredict"
ProcessLasFiles.main(folder_with_lases_to_predict, transformer, model)
print("Program execution is finished")
