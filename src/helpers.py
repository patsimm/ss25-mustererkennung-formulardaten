import cv2
import matplotlib.pyplot as plt

def show(img, figsize=(12, 9), title=None):
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if title is not None:
        plt.title(title)
    plt.show()