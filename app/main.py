import tkinter as tk
import util
import cv2
from PIL import Image, ImageTk
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import pickle
from model import EmbeddingNet, TripletNetwork
from torchvision import transforms
import numpy as np
import datetime

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.title("Attendance System")        
        self.main_window.geometry("800x500+400+150")
        self.main_window.configure(background="#FFE863")

        self.login_btn = util.get_button(self.main_window, 'login', '#8736AA', self.login, fg="#F5BD1F")
        self.login_btn.place(x=450, y=300)

        self.register_btn = util.get_button(self.main_window, 'register', '#F5BD1F', self.register, fg="#8736AA")
        self.register_btn.place(x=450, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=0, y=0, width=500, height=500) 

        self.add_webcam(self.webcam_label)

        self.db_dir = 'known_faces'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        # Model loading
        MODEL_PATH = 'best_triplet_network_final.pth'  
        self.triplet_net = torch.load(MODEL_PATH, map_location=torch.device('cpu'))  # Load the full model
        self.triplet_net.eval()  # Set to evaluation mode
        self.embedding_net = self.triplet_net.embedding_net  # Extract the embedding network

        self.log_path = 'attendance.txt'

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()

        if not ret:
            print("Failed to capture frame from webcam. Check if the webcam is properly connected.")
            return
        
        self.most_recent_capture_arr = frame

        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)

        imgTk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)

        self._label.imgTk = imgTk
        self._label.configure(image=imgTk)

        self._label.after(20, self.process_webcam)

    def login(self):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        input_frame = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(input_frame).unsqueeze(0)

        with torch.no_grad():
            detected_embedding = self.embedding_net(input_tensor)

        # Normalize detected embedding
        detected_embedding = normalize(detected_embedding, p=2, dim=1)

        min_distance = float('inf')
        matched_name = "Unknown"

        for file_name in os.listdir(self.db_dir):
            if file_name.endswith('.jpg'):
                file_path = os.path.join(self.db_dir, file_name)
                registered_img = cv2.imread(file_path)
                registered_tensor = preprocess(cv2.cvtColor(registered_img, cv2.COLOR_BGR2RGB)).unsqueeze(0)

                with torch.no_grad():
                    registered_embedding = self.embedding_net(registered_tensor)

                # Normalize registered embedding
                registered_embedding = normalize(registered_embedding, p=2, dim=1)

                # Calculate distance
                distance = torch.norm(detected_embedding - registered_embedding).item()

                if distance < min_distance:
                    min_distance = distance
                    matched_name = os.path.splitext(file_name)[0]

        if min_distance < 0.7:
            util.msg_box('Login Success', f'Hello, {matched_name}!')
            with open(self.log_path, 'a') as f:
                f.write('{}, {}\n'.format(matched_name, datetime.datetime.now()))
                f.close()
        else:
            util.msg_box('Login Failed', 'Face not recognized. Please register or try again.')



    def register(self):
        self.register_win = tk.Toplevel(self.main_window)
        self.register_win.geometry("800x500+410+160")
        self.register_win.configure(background="#FFE863")
        self.register_win.title("Register")        


        self.accept_btn = util.get_button(self.register_win, 'accept', '#8736AA', self.accept, fg="#F5BD1F")
        self.accept_btn.place(x=450, y=300)

        self.discard_btn = util.get_button(self.register_win, 'discard', '#F5BD1F', self.discard, fg="#8736AA")
        self.discard_btn.place(x=450, y=400)

        self.capture_label = util.get_img_label(self.register_win)
        self.capture_label.place(x=0, y=0, width=500, height=500) 

        self.add_img(self.capture_label)

        self.entry_text = util.get_entry_text(self.register_win)
        self.entry_text.place(x=500, y=100)

        self.text_label = util.get_text_label(self.register_win, 'Insert name:')
        self.text_label.place(x=500, y=70)

    def add_img(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_cap = self.most_recent_capture_arr.copy()

    def accept(self):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        name = self.entry_text.get(1.0, "end-1c")

        # Save the captured face to the known_faces directory
        cv2.imwrite(os.path.join(self.db_dir, f'{name}.jpg'), self.register_cap)

        # Reload the model
        MODEL_PATH = 'best_triplet_network_final.pth'  
        triplet_net = torch.load(MODEL_PATH, map_location=torch.device('cpu'))  # Load the full model
        triplet_net.eval()  # Set the model to evaluation mode

        # Use the embedding network for feature extraction
        embedding_net = triplet_net.embedding_net

    # Preprocess the image and generate an embedding
        input_tensor = preprocess(cv2.cvtColor(self.register_cap, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            embedding = embedding_net(input_tensor)
            embedding = normalize(embedding, p=2, dim=1).squeeze(0)

    # Save the embedding
        EMBEDDINGS_FILE = 'embeddings.pkl'
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'rb') as f:
                known_embeddings = pickle.load(f)
        else:
            known_embeddings = {}

        known_embeddings[name] = embedding
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(known_embeddings, f)

        util.msg_box('Success!', 'Thanks for registering!')
        self.register_win.destroy()

    
    def discard(self):
        self.register_win.destroy()

    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()