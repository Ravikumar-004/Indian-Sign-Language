import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from keras.models import load_model
import numpy as np

# MediaPipe setup
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Load model
model = load_model("model9.h5")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
PRED_EVERY = 5
last_pred = ""

print("Hold a hand in front of the camera... Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)
    h, w, _ = frame.shape

    frame_count += 1

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        xs = [int(lm.x * w) for lm in hand]
        ys = [int(lm.y * h) for lm in hand]

        x_min = max(min(xs) - 20, 0)
        y_min = max(min(ys) - 20, 0)
        x_max = min(max(xs) + 20, w)
        y_max = min(max(ys) + 20, h)

        hand_crop = frame[y_min:y_max, x_min:x_max]

        mask = np.zeros(hand_crop.shape[:2], dtype=np.uint8)
        points = []

        for lm in hand:
            x = int(lm.x * w) - x_min
            y = int(lm.y * h) - y_min
            points.append([x, y])

        points = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 255)

        hand_crop = cv2.bitwise_and(hand_crop, hand_crop, mask=mask)

        if hand_crop.size != 0 and frame_count % PRED_EVERY == 0:
            gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            resized = resized / 255.0

            pred = np.argmax(
                model.predict(resized.reshape(1, 28, 28, 1), verbose=0)
            )
            last_pred = str(pred)

        

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(frame, last_pred, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Smooth Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
