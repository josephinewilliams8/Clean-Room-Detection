import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image

# https://github.com/josephinewilliams8 

# BEFORE RUNNING THIS CODE: 
# 1) make sure that any necessary paths are updated
# 2) refer to README.md for any other questions

def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    print('The dimensions are as follows:')
    print(f'x1, x2, y1, y2 = {x1},{x2},{y1},{y2}')

# Open an image file
img = Image.open('<INSERT PATH TO SAMPLE FRAME HERE>')
arr = np.asarray(img)

fig, ax = plt.subplots()
ax.imshow(arr)

rs = RectangleSelector(ax, onselect, props=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True))
plt.show()