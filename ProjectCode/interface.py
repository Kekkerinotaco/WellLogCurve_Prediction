import customtkinter
from tkinter import messagebox


class Interface(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("800x600")
        self.title("Moroz.GY WellLogPredictionTool[V_0.1]")
        self.minsize(300, 200)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.left_side_frame = customtkinter.CTkFrame(self, width=200, corner_radius=0)
        self.left_side_frame.grid(row=0, column=0, padx=20, rowspan=6, columnspan=3,  pady=10, sticky="nsew")

        self.lases_to_learn_label = customtkinter.CTkLabel(self.left_side_frame,
                                                           text="Put a link to a folder with .las files to learn:\n"
                                                                    "[If you can't paste the link, try to change a lang]")
        self.lases_to_learn_label.grid(row=0, column=0, padx=20, pady=(20, 0), columnspan=2, sticky="nsew")

        self.lases_to_learn_entry = customtkinter.CTkEntry(master=self.left_side_frame)
        self.lases_to_learn_entry.grid(row=1, column=0, rowspan=1, columnspan=2, padx=20, sticky="we")

        self.learning_curves_label = customtkinter.CTkLabel(self.left_side_frame,
                                                            text="Enter the curves to learn from, in a format:\n"
                                                                 "[Curve1, Curve2, Curve3, Curve4, etc.]")

        self.learning_curves_label.grid(row=2, column=0, rowspan=1, columnspan=1, padx=20, sticky="we")

        self.learning_curves_entry = customtkinter.CTkEntry(master=self.left_side_frame)
        self.learning_curves_entry.grid(row=3, column=0, rowspan=1, columnspan=1, padx=20, sticky="we")

        self.target_curve_label = customtkinter.CTkLabel(self.left_side_frame,
                                                            text="Enter the curves to learn from, in a format:\n"
                                                                 "Curve")

        self.target_curve_label.grid(row=2, column=1, rowspan=1, columnspan=1, padx=20, sticky="we")

        self.target_curve_entry = customtkinter.CTkEntry(master=self.left_side_frame)
        self.target_curve_entry.grid(row=3, column=1, rowspan=1, columnspan=1, padx=20, sticky="we")

        self.lases_to_predict_label = customtkinter.CTkLabel(self.left_side_frame,
                                                            text="Put a link to a folder with .las files to predict:\n"
                                                                    "[If you can't paste the link, try to change a lang]")
        self.lases_to_predict_label.grid(row=4, column=0, padx=20, pady=(20, 0), columnspan=2,  sticky="nsew")
        self.lases_to_learn_entry = customtkinter.CTkEntry(master=self.left_side_frame)
        self.lases_to_learn_entry.grid(row=5, column=0, rowspan=1, columnspan=2, padx=20, sticky="we")

        self.run_button = customtkinter.CTkButton(master=self.left_side_frame, text="Run",
                                                  command=self.run_button_command)

        self.run_button.grid(row=6, column=0, padx=20, rowspan=2,  pady=10, sticky="ew")

        self.info_textbox = customtkinter.CTkTextbox(master=self.left_side_frame, height=100)
        self.info_textbox.grid(row=6, column=1, padx=20, rowspan=2,  pady=10, sticky="nsew")

        self.info_textbox.insert("insert",
                             text="WellLog Curve prediction tool. \nIf you have any problems, please contact: \nMoroz.GYu@gazpromneft-ntc.ru\ngrighoriim@mail.ru")

    def run_button_command(self):
        learning_lases_folder = self.lases_to_learn_entry.get().strip('"').strip("'")
        learning_curves = self.learning_curves_entry.get().strip('"').strip("'")
        target_curve = self.target_curve_entry.get().strip('"').strip("'")
        
        print("MakingSomeActions")


if __name__ == "__main__":
    interface = Interface()
    interface.mainloop()
