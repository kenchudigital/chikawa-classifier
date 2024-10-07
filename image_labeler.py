import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import os

def resize_with_padding(image, target_size):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    resized_image = cv2.resize(image, (new_width, new_height))
    
    black_background = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    black_background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    
    return black_background

def process_images(image_path, target_size=100):
    resized_objects = []
    
    image = cv2.imread(image_path)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(thresh, threshold1=30, threshold2=100)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(eroded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [(cv2.contourArea(cnt), cnt) for cnt in contours]
    contour_areas.sort(reverse=True, key=lambda x: x[0])
    
    top_contours = [cnt for area, cnt in contour_areas[:10]]
    
    for idx, contour in enumerate(top_contours):
        x, y, w, h = cv2.boundingRect(contour)
        object_img = image[y:y+h, x:x+w]
        padded_img = resize_with_padding(object_img, target_size)
        resized_objects.append(padded_img)

    return resized_objects

class ImageLabelerApp:
    def __init__(self, root, images):
        self.root = root 
        self.images = images
        self.current_index = 0
        self.names = {}
        self.load_names()
        
        self.root.title("Image Labeler")
        self.root.geometry("800x600")
        
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()
        
        self.name_var = tk.StringVar(root)
        self.name_var.set("Select Name")
        
        self.name_option = tk.OptionMenu(root, self.name_var, *self.names.values())
        self.canvas.create_window(400, 550, window=self.name_option)
        
        self.image_label = tk.Label(root)
        self.canvas.create_window(400, 300, window=self.image_label)
        
        self.name_entry = tk.Entry(root, width=10)
        self.name_entry.bind('<KeyPress>', self.ignore_keys)
        self.canvas.create_window(400, 580, window=self.name_entry)
        
        self.save_button = tk.Button(root, text="Save Name", command=self.save_name)
        self.canvas.create_window(300, 580, window=self.save_button)
        
        self.skip_button = tk.Button(root, text="Skip Image", command=self.skip_image)
        self.canvas.create_window(500, 580, window=self.skip_button)
        
        self.completed_button = tk.Button(root, text="Completed", command=self.complete_task)
        
        self.show_image(self.current_index)
        
        # Bind keys to functions
        self.root.bind('<Right>', self.handle_right_key)
        self.root.bind('<Down>', self.handle_save_key)
        self.root.bind('1', lambda e: self.select_option('0'))
        self.root.bind('2', lambda e: self.select_option('1'))
        self.root.bind('3', lambda e: self.select_option('2'))
        self.root.bind('4', lambda e: self.select_option('3'))
        self.root.bind('5', lambda e: self.select_option('4'))
        self.root.bind('6', lambda e: self.select_option('5'))
        self.root.bind('7', lambda e: self.select_option('6'))
        self.root.bind('8', lambda e: self.select_option('7'))
        self.root.bind('9', lambda e: self.select_option('8'))
        self.root.bind('0', lambda e: self.select_option('9'))
        self.root.bind('<minus>', lambda e: self.select_option('10'))  # For `~` symbol
        
        self.name_entry.bind('<KeyPress>', self.ignore_keys)
        

    def ignore_keys(self, event):
        """Ignore certain key presses in the Entry widget."""
        if event.keysym in  ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'q']:
            return 'break'  # Ignore the event
    
    def handle_right_key(self, event):
        self.skip_image()
        
    def handle_save_key(self, event):
        self.save_name()
        
    def select_option(self, option):
        if int(option) in list(self.names.keys()):
            print('good')
            print(self.names)
            self.name_var.set(self.names[int(option)])
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, self.names[int(option)])
    
    def show_image(self, index):
        if index >= 0 and index < len(self.images):
            self.update_option_menu()
            
            image_rgb = cv2.cvtColor(self.images[index], cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            tk_image = ImageTk.PhotoImage(image_pil)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image
            self.completed_button.pack_forget()
        else:
            self.image_label.config(image='')
            self.completed_button.pack(pady=20)
    
    def save_name(self):
        name = self.name_entry.get().strip().lower()
        if name and self.current_index >= 0 and self.current_index < len(self.images):
            self.save_image(self.current_index, name)
            self.name_entry.delete(0, tk.END)
            
            self.current_index += 1
            if self.current_index < len(self.images):
                self.show_image(self.current_index)
            else:
                self.complete_task()

    def save_image(self, index, name):
        labeled_folder = 'labeled_image'
        os.makedirs(labeled_folder, exist_ok=True)
        image = self.images[index]
        count = 1
        file_name = f"{labeled_folder}/{name}.jpg"
        while os.path.exists(file_name):
            file_name = f"{labeled_folder}/{name}{count}.jpg"
            count += 1
        cv2.imwrite(file_name, image)
    
    def skip_image(self):
        if self.current_index >= 0 and self.current_index < len(self.images):
            self.current_index += 1
            if self.current_index < len(self.images):
                self.show_image(self.current_index)
            else:
                self.complete_task()
    
    def complete_task(self):
        self.root.quit()
        
    def load_names(self):
        self.names = {}
        if os.path.exists('name_list.txt'):
            with open('name_list.txt', 'r') as file:
                for line in file:
                    index, name = line.strip().split(',', 1)
                    self.names[int(index)] = name.strip()

    def update_option_menu(self):
        options = list(self.names.values())
        menu = self.name_option["menu"]
        menu.delete(0, "end")
        for option in options:
            menu.add_command(label=option, command=tk._setit(self.name_var, option, self.on_option_selected))

    def on_option_selected(self, selected_option):
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, selected_option)



def main():
    root = tk.Tk()
    folder_path = 'images/images'
    images = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print('Starting to handle:', image_path)
            processed_images = process_images(image_path)
            images.extend(processed_images)

    app = ImageLabelerApp(root, images)
    root.mainloop()

if __name__ == "__main__":
    main()
