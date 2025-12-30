import cv2
import os
import numpy as np

X = []
y = []

for folder in os.listdir("cropdata"):
    folder_path = os.path.join("cropdata", folder)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        

        img = cv2.resize(img, (28, 28))

        X.append(img)
        y.append(folder)  

    print(f"Processed: {folder}")

X = np.array(X)           
y = np.array(y)          

print("X shape:", X.shape)
print("y shape:", y.shape)

np.savez("Dataset/dataset.npz", X=X, y=y)

# data = np.load("data.npz")
# arr1 = data["X"] 
# arr2 = data["y"]