import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import random
import csv
import os
import numpy as np

# Where the results file will be located
RESULTS_FILE = "C:/Users/guill/OneDrive/Bureau/voting_test/second_survey20.csv"

# Where the images file is located
STITCHING_FOLDER = "C:/Users/guill/OneDrive/Bureau/image_samples/"


# Column A parrallax || column B image number || algorithm || choice (good or bad)

# Main Application Class
class VotingApp:
    def __init__(self, root, image_path, max_votes):
        self.root = root
        self.max_votes = max_votes
        self.vote_count = 0
        self.image_path = image_path
        self.folderpaths = []
        self.subfolders_map = {}  # Dictionary: {folder: [list of subfolders]}

        half_votes = max_votes // 2
        self.case_indices = np.array([0] * half_votes + [1] * half_votes)
        np.random.shuffle(self.case_indices)
        
        # Column A parrallax || column B image number || algorithm || choice (good or bad)
        self.results = []
        
        # Collect all top-level folders and their subfolders
        for foldername in os.listdir(image_path):
            folder_path = os.path.join(image_path, foldername)
            if os.path.isdir(folder_path):  # Ensure it's a directory
                self.folderpaths.append(folder_path)
                self.subfolders_map[folder_path] = self._get_subfolders(folder_path)
        
        # GUI Setup
        self.root.title("Voting Application")

        # Create frames for images and buttons
        self.image_frame = tk.Frame(root)
        self.button_frame = tk.Frame(root)

        # Pack frames vertically
        self.image_frame.pack(side=tk.TOP, pady=10)
        self.button_frame.pack(side=tk.TOP, pady=10)

        # Image labels
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(side=tk.TOP, padx=10)

        # Buttons under images
        self.left_button = tk.Button(self.button_frame, text="Good", command=self.vote_left)
        self.right_button = tk.Button(self.button_frame, text="Bad", command=self.vote_right)

        self.left_button.pack(side=tk.LEFT, padx=10)
        self.right_button.pack(side=tk.LEFT, padx=10)

        self.next_images()

    def next_images(self):
        if self.vote_count >= self.max_votes:
            self.save_results_to_csv()
            messagebox.showinfo("Voting Terminated", "You have completed the voting process. Thank You!")
            self.root.quit()
            return
        
        images_infos = self.random_image_selection()
        if images_infos is None:
            self.next_images()  # Exit if no images are found
        
        self.image_path, self.image_name = images_infos
        
        image = Image.open(self.image_path)
        max_size = (600, 400)
        image_resized = resize_with_max_size(image, max_size)
        
        self.photo = ImageTk.PhotoImage(image_resized)
        self.image_label.config(image=self.photo)
        self.vote_count += 1

    def _get_subfolders(self, parent_folder):
        # Private method to collect subfolders in a folder
        subfolder_paths = []
        for foldername in os.listdir(parent_folder):
            subfolder_path = os.path.join(parent_folder, foldername)
            if os.path.isdir(subfolder_path):  # Ensure it's a directory
                subfolder_paths.append(subfolder_path)
        return subfolder_paths

    def vote_left(self):
        self.record_vote("Good")
        self.next_images()

    def vote_right(self):
        self.record_vote("Bad")
        self.next_images()

    def random_case_selection(self):
        case_index = self.case_indices[self.vote_count]
        selected_case = self.folderpaths[case_index]
        return random.choice(self.subfolders_map[selected_case])
    
    def random_image_selection(self):
        case_path = self.random_case_selection()
        image_folder_name = "stitched_images"
        image_path = os.path.join(case_path, image_folder_name)
        
        if not os.path.exists(image_path) or not os.path.isdir(image_path):
            print(f"Image path '{image_path}' does not exist or is not a directory.")
            return None

        image_files = [f for f in os.listdir(image_path)
                       if os.path.isfile(os.path.join(image_path, f))]

        if len(image_files) < 2:
            print(f"Not enough images found in '{image_path}'.")
            return None

        selected_images = random.sample(image_files, 1)

        #full paths and file names returned
        image = os.path.join(image_path, selected_images[0])
        return image, selected_images[0]
    
    def record_vote(self, vote):
        folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.image_path))))  # Two levels up
        parallax = "large parallax" if "large" in folder.lower() else "small parallax"

        # Full results : # parallax | image number| image algo | choice (good or bad)
        img_number = os.path.basename(os.path.dirname(os.path.dirname(self.image_path)))
        algo = os.path.basename(self.image_path).split('.')[0]
        self.results.append((parallax, img_number, algo, vote))

    def save_results_to_csv(self):
        with open(RESULTS_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Parallax", "Image Number", "Image", "Choice"])

            for elements in self.results:
                row = list(elements)
                writer.writerow(row)


def resize_with_max_size(image, max_size):
    """Resize the image to fit within the max_size while maintaining aspect ratio."""
    max_width, max_height = max_size
    img_ratio = image.width / image.height
    max_ratio = max_width / max_height

    if img_ratio > max_ratio:
        # Image is wider than max_size; scale by width
        new_width = max_width
        new_height = int(new_width / img_ratio)
    else:
        # Image is taller than max_size; scale by height
        new_height = max_height
        new_width = int(new_height * img_ratio)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Run the Application
if __name__ == "__main__":
    root = tk.Tk()
    app = VotingApp(root, STITCHING_FOLDER, max_votes=20)
    root.mainloop()
