import customtkinter
# import ProcellWellDocs
from tkinter import messagebox


class SaveTXTFrame(customtkinter.CTkFrame):
    def __init__(self, *args, header_name="", **kwargs):
        super().__init__(*args, **kwargs)

        self.header_name = header_name

        self.header = customtkinter.CTkLabel(master=self, text=self.header_name)
        self.header.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.radio_button_var = customtkinter.StringVar(value="")

        self.radio_button_1 = customtkinter.CTkRadioButton(master=self, text="Сохранить",
                                                           value=True,
                                                           variable=self.radio_button_var)
        self.radio_button_1.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.radio_button_2 = customtkinter.CTkRadioButton(master=self, text="Не сохранять",
                                                           value=False,
                                                           variable=self.radio_button_var)

        self.radio_button_2.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

    def get_value(self):
        return self.radio_button_var.get()


class SaveTagFrame(customtkinter.CTkFrame):
    def __init__(self, *args, header_name="", **kwargs):
        super().__init__(*args, **kwargs)

        self.header_name = header_name

        self.header = customtkinter.CTkLabel(master=self, text=self.header_name)
        self.header.grid(row=0, column=0, padx=10, pady=10)

        self.radio_button_var = customtkinter.StringVar(value="")

        self.radio_button_1 = customtkinter.CTkRadioButton(master=self, text="Сохранить",
                                                           value=True,
                                                           variable=self.radio_button_var)
        self.radio_button_1.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.radio_button_2 = customtkinter.CTkRadioButton(master=self, text="Не сохранять",
                                                           value=False,
                                                           variable=self.radio_button_var)
        self.radio_button_2.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

    def get_value(self):
        return self.radio_button_var.get()


class Interface(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.geometry("800x600")
        self.title("Moroz.GY WellDocsProcessingTool")
        self.minsize(300, 200)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7), weight=1)

        self.left_sidebar_frame_one = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.left_sidebar_frame_one.grid(row=0, column=0, padx=20, pady=10, rowspan=3, sticky="nsew")

        self.input_link_label = customtkinter.CTkLabel(self.left_sidebar_frame_one,
                                                       text="Вставьте ссылку на папку с файлами для обработки:\n"
                                                            "[Если не вставляется, попробуйте сменить язык]")
        self.input_link_label.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="n")

        self.input_link_entry = customtkinter.CTkEntry(master=self.left_sidebar_frame_one)
        self.input_link_entry.grid(row=1, column=0, rowspan=1, columnspan=2, padx=20, sticky="we")

        self.left_sidebar_frame_two = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.left_sidebar_frame_two.grid(row=4, column=0, padx=20, pady=10, rowspan=3, columnspan=1, sticky="nsew")

        self.output_link_label = customtkinter.CTkLabel(master=self.left_sidebar_frame_two,
                                                        text="В какой папке сохранить результат\n [Папка не должна быть в папке инпута]:")
        self.output_link_label.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 0), sticky="ns")

        self.output_link_entry = customtkinter.CTkEntry(master=self.left_sidebar_frame_two)
        self.output_link_entry.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="nsew")

        self.searched_tag_label = customtkinter.CTkLabel(master=self.left_sidebar_frame_two,
                                                         text="Введите искомый тег:")
        self.searched_tag_label.grid(row=2, column=0, padx=(20, 0), pady=0, sticky="ns")

        self.searched_tag_entry = customtkinter.CTkEntry(master=self.left_sidebar_frame_two)
        self.searched_tag_entry.grid(row=3, column=0, padx=20, pady=0, sticky="w")

        self.run_button = customtkinter.CTkButton(self.left_sidebar_frame_two, text="Провести расчет",
                                                  command=self.run_button_command)
        self.run_button.grid(row=3, column=1, padx=20, pady=10, sticky="ew")

        self.right_sidebar_frame_two = customtkinter.CTkFrame(master=self, width=140, corner_radius=0)

        self.right_sidebar_frame_two.grid(row=0, column=1, rowspan=3, padx=20, pady=10, sticky="nsew")

        self.save_txt_button_frame = SaveTXTFrame(master=self.right_sidebar_frame_two,
                                                  header_name="Сохранение .txt файлов")

        self.save_txt_button_frame.grid(row=0, column=2, rowspan=3, padx=20, pady=10, sticky="n")

        self.right_sidebar_frame_two = customtkinter.CTkFrame(master=self, width=140, corner_radius=0)
        self.right_sidebar_frame_two.grid(row=4, column=1, rowspan=1, padx=20, pady=10, sticky="nsew")

        self.save_no_tag_files = SaveTagFrame(self.right_sidebar_frame_two, header_name="Сохранение файлов без тега")
        self.save_no_tag_files.grid(row=0, column=0, padx=20, pady=10, rowspan=3, sticky="nsew")

        self.textbox = customtkinter.CTkTextbox(master=self, height=100)
        self.textbox.grid(row=7, column=0, padx=20, pady=20, columnspan=2, sticky="ew")
        self.textbox.insert("insert",
                             text="Программа для обработки дел скважин. \nПо вопросам работы можно обращаться: \nMoroz.GYu@gazpromneft-ntc.ru\ngrighoriim@mail.ru")

    def run_button_command(self):
        # Часть .strip('"').strip("'").strip() нужна, чтобы удалить ковычки при копировании через "копировать как путь,
        # или же если пользователь случайно вставил пробелы в начало/конец пути"
        folder_path = self.input_link_entry.get().strip('"').strip("'")
        result_folder_path = self.output_link_entry.get().strip('"').strip("'")
        searched_tag = self.searched_tag_entry.get()
        try:
            save_no_tag = int(self.save_no_tag_files.get_value())
            save_txt = int(self.save_txt_button_frame.radio_button_var.get())
        except ValueError:
            messagebox.showinfo(title="Предупреждение",
                                message="Не выбрана/ны опция/ии сохранения файлов, по умолчанию выбрано [да] [да].")
            save_no_tag = 1
            save_txt = 1
        try:
            ProcellWellDocs.main(folder_path, result_folder_path, self.textbox, searched_tag,
                                 save_empty_pictures_status=save_no_tag,
                                 save_txt_status=save_txt, write_unopened_files=True)
        except FileNotFoundError:
            messagebox.showinfo(title="Ошибка",
                                message="Указан неверный путь, необходимо проверить корректность введенных данных.\n"
                                        "Выполнение программы приостановлено.")
        except NotADirectoryError:
            messagebox.showinfo(title="Ошибка",
                                message="Вероятно, указан путь не до папки, а до файла, необходимо проверить корректность введенных данных.\n"
                                        "Выполнение программы приостановлено.")
        messagebox.showinfo(title="Завершение работы программы", message="Выполнение программы завершено.")


if __name__ == "__main__":
    interface = Interface()
    interface.mainloop()
