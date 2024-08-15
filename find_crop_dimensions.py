import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image

def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    # cropped_img = img.crop((x1, y1, x2, y2))
    print('the dimensions are: x1', x1, 'x2', x2, 'y1', y1, 'y2', y2)

# Open an image file
img = Image.open('tester/test0.jpg')
arr = np.asarray(img)

fig, ax = plt.subplots()
ax.imshow(arr)

rs = RectangleSelector(ax, onselect, props=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True))
plt.show()