import json
import os
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Scale

import numpy as np
from align_auto import align
from align_manual import align_manual

class ImageTransparencyAdjuster:
    def __init__(self, master):
        self.master = master

        self.path_well = '...'
        self.path_fast = '...'
        self.path_transformed = '...'
        self.path_merged = '...'
        self.path_masks = '...'
        self.json_file = '...'

        os.makedirs(self.path_transformed, exist_ok=True)
        os.makedirs(self.path_merged, exist_ok=True)

        self.well_images = sorted(os.listdir(self.path_well))
        self.fast_images = sorted(os.listdir(self.path_fast))
        self.index = 82
        self.matrix = np.zeros((3, 3))
        self.merged_image = None
        self.mask = None

        self.master.title("Image Transparency Adjuster")

        self.well, self.fast, self.warped_image = self.load_image_set(self.index)

        frame = tk.Frame(master)
        frame.pack(side=tk.LEFT)

        self.label = tk.Label(frame)
        self.label.pack()

        slider = Scale(master, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_image)
        slider.set(50)
        slider.pack(side=tk.RIGHT, fill=tk.Y)

        self.master.bind('<Key>', self.handle_key)

        self.update_image(50)

    def load_image_set(self, index):
        well = cv2.cvtColor(cv2.imread(os.path.join(self.path_well, self.well_images[index])), cv2.COLOR_BGR2RGB)
        fast = cv2.cvtColor(cv2.imread(os.path.join(self.path_fast, self.fast_images[index])), cv2.COLOR_BGR2RGB)

        try:
            warped_image = cv2.cvtColor(cv2.imread(os.path.join(self.path_transformed, self.fast_images[index])), cv2.COLOR_BGR2RGB)
        except Exception:
            print("Transformed image not found:", self.index, self.fast_images[self.index])
            warped_image = fast
        return well, fast, warped_image

    def update_image(self, alpha):
        alpha = float(alpha) / 100.0
        beta = 1.0 - alpha
        self.merged_image = cv2.addWeighted(self.well, beta, self.warped_image, alpha, 0)
        merged_image = Image.fromarray(self.merged_image)
        merged_image_tk = ImageTk.PhotoImage(merged_image)
        self.label.config(image=merged_image_tk)
        self.label.image = merged_image_tk
        self.first_run = False
        self.master.update()
        self.master.focus_force()

    def transform(self, manual, n):
        if manual:
            self.warped_image, self.matrix = align_manual(self.well, self.fast, n)
        else:
            try:
                self.warped_image, self.matrix = align(self.well, self.fast)
                print("Aligned!")
            except Exception as e:
                print("Failed to align.", e)
        self.update_image(50)

    def save_transformation(self):
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {'matrices': []}

        found = False
        for item in data['matrices']:
            if item['name'] == self.fast_images[self.index]:
                item['matrix'] = self.matrix.tolist()
                found = True
                break
        
        if not found:
            named_matrix = {'name': self.fast_images[self.index], 'matrix': self.matrix.tolist()}
            data['matrices'].append(named_matrix)

        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=4)

    def delete_transformation(self):
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {'matrices': []}

        data['matrices'] = [item for item in data['matrices'] if item['name'] != self.fast_images[self.index]]

        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=4)

    def handle_key(self, event):
        if event.keysym in ['Right', 'Left']:
            self.matrix = np.zeros((3, 3))
            self.index += 1 if event.keysym == 'Right' else -1
            self.well, self.fast, self.warped_image = self.load_image_set(self.index)
            self.update_image(50)
            print(self.index, self.fast_images[self.index])
        elif event.keysym in ['Up', 'Down']:
            self.matrix = np.zeros((3, 3))
            self.warped_image = self.fast if event.keysym == 'Down' else cv2.cvtColor(cv2.imread(os.path.join(self.path_transformed, self.fast_images[self.index])), cv2.COLOR_BGR2RGB)
            self.update_image(50)
        elif event.keysym == 's':
            cv2.imwrite(os.path.join(self.path_transformed, self.fast_images[self.index]), cv2.cvtColor(self.warped_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.path_merged, self.fast_images[self.index]), cv2.cvtColor(self.merged_image, cv2.COLOR_RGB2BGR))
            self.save_transformation()
            print('Saved:', self.index, self.fast_images[self.index])
        elif event.keysym == 'd':
            try:
                os.remove(os.path.join(self.path_transformed, self.fast_images[self.index]))
                self.delete_transformation()
                print('Deleted:', self.index, self.fast_images[self.index])
            except (FileNotFoundError):
                print('File not found:', self.index, self.fast_images[self.index])
        else:
            pts = {'a': 0, '4': 4, '6': 6, '8': 8, '1': 10}
            if event.char in pts:
                self.transform(event.char != 'a', pts[event.char])

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageTransparencyAdjuster(root)
    root.mainloop()
