import cv2
import matplotlib.pyplot as plt
import numpy as np

img_1 = cv2.imread("1.png")
img_2 = cv2.imread("2.png")

img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)


frames = [img_1, img_2]

sum = np.zeros(1)

for f in frames:
    sum += abs(f - img_1) % 256

plt.figure(figsize=(20, 10))
plt.imshow(sum, cmap="gray")
