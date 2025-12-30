import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import numpy as np

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

INPUT_DIR = "data"
OUTPUT_DIR = "cropdata"

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_df = []
y_df = []

for folder in os.listdir(INPUT_DIR):
    input_folder = os.path.join(INPUT_DIR, folder)
    
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)
        h, w, _ = frame.shape

        for idx, hand in enumerate(result.hand_landmarks):
            xs = [int(lm.x * w) for lm in hand]
            ys = [int(lm.y * h) for lm in hand]

            x_min = max(min(xs) - 20, 0)
            y_min = max(min(ys) - 20, 0)
            x_max = min(max(xs) + 20, w)
            y_max = min(max(ys) + 20, h)

            hand_crop = frame[y_min:y_max, x_min:x_max]

            hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)

            mask = np.zeros(hand_crop.shape[:2], dtype=np.uint8)
            points = []

            for lm in hand:
                x = int(lm.x * w) - x_min
                y = int(lm.y * h) - y_min
                points.append([x, y])

            points = np.array(points, dtype=np.int32)
            cv2.fillConvexPoly(mask, points, 255)

            hand_crop = cv2.bitwise_and(hand_crop, hand_crop, mask=mask)

            hand_crop_resized = cv2.resize(hand_crop, (28, 28))

            X_df.append(hand_crop_resized)
            y_df.append(folder)  

    print(f"Processed: {folder}")

        
X = np.array(X_df)           
y = np.array(y_df)          

print("X shape:", X.shape)
print("y shape:", y.shape)

np.savez("Dataset/dataset.npz", X=X, y=y)
