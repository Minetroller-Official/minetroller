import cv2
import mediapipe as mp
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.preprocessing import StandardScaler
from threading import Thread
import logging
import time
from PIL import Image, ImageTk

# Configure logging
logging.basicConfig(
    filename="minecraft_controller_advanced.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Default Control Mapping
DEFAULT_MAPPING = {
    "walk_forward": "w",
    "walk_backward": "s",
    "move_left": "a",
    "move_right": "d",
    "jump": "space",
    "sneak": "shift",
    "sprint": "ctrl",
    "attack": "left_click",
    "mine": "right_click",
    "inventory": "e",
    "pause_game": "esc",
    "chat": "enter",
}

# Save and Load JSON Functions
def save_json(data, filepath):
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved data to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")


def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"File {filepath} not found, returning empty.")
        return {}
    except Exception as e:
        logging.error(f"Failed to load JSON: {e}")
        return {}


# PyTorch Neural Network Model for Gesture Recognition
class GestureRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GestureRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Gesture Handler
class GestureHandler:
    def __init__(self):
        self.model = GestureRecognitionModel(input_size=33 * 3, hidden_size=128, output_size=len(DEFAULT_MAPPING))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = StandardScaler()
        self.gesture_labels = list(DEFAULT_MAPPING.keys())
        self.gesture_data = []
        self.gesture_targets = []

    def train_model(self, epochs=50):
        data = torch.tensor(self.scaler.fit_transform(self.gesture_data), dtype=torch.float32)
        targets = torch.tensor(self.gesture_targets, dtype=torch.long)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def predict(self, landmarks):
        processed_data = self.scaler.transform([landmarks])
        tensor_data = torch.tensor(processed_data, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(tensor_data)
        predicted_class = torch.argmax(output).item()
        return self.gesture_labels[predicted_class]

    def add_gesture_sample(self, landmarks, label):
        self.gesture_data.append(landmarks)
        self.gesture_targets.append(self.gesture_labels.index(label))


# Splash Screen Class
class SplashScreen:
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)  # Remove title bar
        self.root.geometry("400x300")
        self.center_window(400, 300)

        # Load splash image
        splash_image = Image.open("minetroller_logo.png")  # Add your image file here
        splash_image = splash_image.resize((400, 300), Image.ANTIALIAS)
        self.splash_image = ImageTk.PhotoImage(splash_image)

        self.label = tk.Label(self.root, image=self.splash_image, bg="black")
        self.label.pack()

    def center_window(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def destroy_splash(self):
        self.root.destroy()


# Minecraft Controller App
class MinecraftControllerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Minecraft Motion Controller")
        self.gesture_handler = GestureHandler()
        self.current_mapping = DEFAULT_MAPPING.copy()
        self.calibration_data = {}
        self.running = True

        self.setup_ui()
        self.load_calibration_data()

    def setup_ui(self):
        # Tabs
        self.tab_control = ttk.Notebook(self.root)
        self.main_tab = ttk.Frame(self.tab_control)
        self.calibration_tab = ttk.Frame(self.tab_control)
        self.mapping_tab = ttk.Frame(self.tab_control)

        self.tab_control.add(self.main_tab, text="Main")
        self.tab_control.add(self.calibration_tab, text="Calibration")
        self.tab_control.add(self.mapping_tab, text="Gesture Mapping")
        self.tab_control.pack(expand=1, fill="both")

        # Main Tab
        ttk.Label(self.main_tab, text="Minecraft Motion Controller", font=("Arial", 16)).pack(pady=10)
        ttk.Button(self.main_tab, text="Start Controller", command=self.start_controller).pack(pady=10)
        ttk.Button(self.main_tab, text="Stop Controller", command=self.stop_controller).pack(pady=10)

    def load_calibration_data(self):
        self.calibration_data = load_json("calibration_data.json")

    def start_controller(self):
        self.running = True
        Thread(target=self.run_camera, daemon=True).start()

    def stop_controller(self):
        self.running = False

    def run_camera(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            if results.pose_landmarks:
                landmarks = [lm.x for lm in results.pose_landmarks.landmark] + \
                            [lm.y for lm in results.pose_landmarks.landmark] + \
                            [lm.z for lm in results.pose_landmarks.landmark]
                detected_gesture = self.gesture_handler.predict(landmarks)
                self.perform_action(detected_gesture)

            cv2.imshow("Minecraft Controller Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def perform_action(self, gesture):
        action = self.current_mapping.get(gesture)
        if action == "left_click":
            pyautogui.click(button="left")
        elif action == "right_click":
            pyautogui.click(button="right")
        elif action:
            pyautogui.press(action)


if __name__ == "__main__":
    splash_root = tk.Tk()
    splash = SplashScreen(splash_root)

    def start_main_app():
        time.sleep(3)  # Display splash for 3 seconds
        splash.destroy_splash()

        root = tk.Tk()
        app = MinecraftControllerApp(root)
        root.mainloop()

    Thread(target=start_main_app).start()
    splash_root.mainloop()
