#!/usr/bin/python3
# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi
# Yan Zhuang 

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os
import json

class TextFileLoader(tk.Tk):
    def __init__(self):
        super().__init__()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 1)
        self.geometry(f"{window_width}x{window_height}")
        self.title('Report Annotator')

        self.current_file_index = 0
        self.files = []
        self.check_values = [tk.BooleanVar() for _ in range(45)]  # 9 label frames * 5 checkboxes each
        self.dropdown_values = [[tk.StringVar(value="") for _ in range(3)] for _ in range(9)]  # Dropdowns for XXX, YYY, ZZZ
        self.file_check_status = {}  # File-based status data
        self.results_file_path = ''  # Path for the results file

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=10)

        self.label_file_name = tk.Label(self.button_frame, text="", font=('Arial', 12))
        self.label_file_name.pack(side=tk.LEFT)

        # Progress bar
        self.progress = ttk.Progressbar(self.button_frame, orient='horizontal', length=200, mode='determinate')
        self.progress.pack(side=tk.LEFT, pady=10)
        self.progress_label = tk.Label(self.button_frame, text="", font=('Arial', 10))
        self.progress_label.pack(side=tk.LEFT, padx=10)

        
        self.change_folder_button = tk.Button(self.button_frame, text='Open Folder', command=self.load_folder)
        self.change_folder_button.pack(side=tk.LEFT, padx=10)

        self.prev_button = tk.Button(self.button_frame, text='Previous', command=self.load_previous_file)
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(self.button_frame, text='Next', command=self.load_next_file)
        self.next_button.pack(side=tk.LEFT, padx=10)

        # Text widget with a scrollbar
        self.text_frame = tk.Frame(self)
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        self.text_widget = tk.Text(self.text_frame, height=25, width=80)
        self.scrollbar = tk.Scrollbar(self.text_frame, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=self.scrollbar.set)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame for checkboxes
        self.frame_checkboxes = tk.Frame(self)
        self.frame_checkboxes.pack(fill=tk.BOTH, expand=True, pady=1)
        # Names for the label frames
        label_frame_names = ["Spleen", "Liver", "Right-kidney", "Left-kidney", \
            "Stomach", "Pancreas", "Gallbladder", "Small Bowel", "Large Bowel"]
        
        checkbox_names = ["LIM", "SUR", "ENL", "DIF", "FOC"]
        dropdown_options = ["", "normal", "low urgency", "medium urgency", "high urgency"]
     

        self.label_frames = []

        for i, label_frame_name in enumerate(label_frame_names):
            label_frame = tk.LabelFrame(self.frame_checkboxes, text=label_frame_name, padx=1, pady=1)
            label_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1, pady=1)
            self.label_frames.append(label_frame)

            for j, checkbox_name in enumerate(checkbox_names):
                index = i * 5 + j
                frame = tk.Frame(label_frame)
                frame.pack(anchor='w', pady=2)

                # Checkbox
                chk = tk.Checkbutton(frame, text=checkbox_name, variable=self.check_values[index])
                chk.pack(side=tk.LEFT)

                # Dropdown for "XXX," "YYY," and "ZZZ"
                if checkbox_name in ["ENL", "DIF", "FOC"]:
                    dropdown = ttk.Combobox(
                        frame,
                        values=dropdown_options,
                        textvariable=self.dropdown_values[i][["ENL", "DIF", "FOC"].index(checkbox_name)],
                        width=6,
                        state="readonly",
                    )
                    dropdown.pack(side=tk.LEFT, padx=1)

        # Read-only text area
        self.info_frame = tk.Frame(self)
        self.info_frame.pack(fill=tk.X, pady=10)
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        self.info_text = tk.Text(self.info_frame, height=10, width=80, font=('Arial', 8), wrap='word')
        self.info_text.pack(fill=tk.Y, side = tk.LEFT, padx=10, pady=5)
        self.info_text.insert(tk.END, "LIM: Lacks imaging quality, contrast opacification, or is not fully imaged and has limited evaluation; \nSUR: Has postsurgical changes; \nENL: Is enlarged or has atrophy; \nDIF: Has diffuse disease or finding; \nFOC: Has focal disease or finding.")
        self.info_text.insert(tk.END, "\nUrgency level:")
        self.info_text.insert(tk.END, "\nNormal: expected or chronic; \nLow urgency: incidental, unexpected; \nMedium: urgency, significant; \nHigh urgency: critical.")

        self.info_text.configure(state='disabled')  # Make the text read-only


        # Note input area
        self.note_label = tk.Label(self.info_frame, text="Note:", font=('Arial', 8))
        self.note_label.pack(side=tk.TOP)
        self.note_text = tk.Text(self.info_frame, height=2, width=80)
        self.note_text.pack(fill=tk.BOTH, expand=True)



    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder Containing Text Files")
        if folder_path:
            self.files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            self.files = [os.path.join(folder_path, f) for f in self.files]
            self.results_file_path = os.path.join(folder_path, 'results.json')
            self.progress['maximum'] = len(self.files)
            self.current_file_index = 0
            self.file_check_status = {}
            if os.path.exists(self.results_file_path):
                self.load_existing_results()
            self.load_file()
            self.progress['value'] = 0

    def load_existing_results(self):
        with open(self.results_file_path, 'r') as f:
            self.file_check_status = json.load(f)

    def load_file(self):
        if self.files:
            file_path = self.files[self.current_file_index]
            with open(file_path, 'r') as file:
                content = file.read()
                content = content.replace('Â', '\n\n')
                self.text_widget.configure(state='normal')
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(tk.END, content)
                self.text_widget.configure(state='disabled')
            self.label_file_name.config(text=f"Current File: {os.path.basename(file_path)}")
            self.restore_checkboxes_and_notes()
            self.update_progress()

    def restore_checkboxes_and_notes(self):
        file_path = self.files[self.current_file_index]
        if file_path in self.file_check_status:
            data = self.file_check_status[file_path]
            for i, frame_data in enumerate(data.get("label_frames", [])):
                for j, checkbox_state in enumerate(frame_data.get("checkboxes", [])):
                    index = i * 5 + j
                    self.check_values[index].set(checkbox_state)
                for j, dropdown_value in enumerate(frame_data.get("dropdowns", [])):
                    self.dropdown_values[i][j].set(dropdown_value)
            self.note_text.delete(1.0, tk.END)
            self.note_text.insert(tk.END, data.get("note", ""))
        else:
            self.reset_checkboxes()
            self.reset_dropdowns()
            self.note_text.delete(1.0, tk.END)

    def reset_checkboxes(self):
        for val in self.check_values:
            val.set(False)

    def reset_dropdowns(self):
        for dropdown_set in self.dropdown_values:
            for dropdown in dropdown_set:
                dropdown.set("")

    def save_results(self):
        file_path = self.files[self.current_file_index]
        data = {
            "label_frames": [],
            "note": self.note_text.get(1.0, tk.END).strip(),
        }
        for i, label_frame in enumerate(self.label_frames):
            frame_data = {
                "checkboxes": [self.check_values[i * 5 + j].get() for j in range(5)],
                "dropdowns": [self.dropdown_values[i][j].get() for j in range(3)],
            }
            data["label_frames"].append(frame_data)
        self.file_check_status[file_path] = data
        with open(self.results_file_path, 'w') as f:
            json.dump(self.file_check_status, f, indent=4)

    def load_next_file(self):
        self.save_results()
        self.current_file_index = (self.current_file_index + 1) % len(self.files)
        self.load_file()

    def load_previous_file(self):
        self.save_results()
        self.current_file_index = (self.current_file_index - 1) % len(self.files)
        self.load_file()

    def update_progress(self):
        self.progress['value'] = self.current_file_index + 1
        self.progress_label.config(text=f"{self.current_file_index + 1} / {len(self.files)}")

if __name__ == "__main__":
    app = TextFileLoader()
    app.mainloop()
