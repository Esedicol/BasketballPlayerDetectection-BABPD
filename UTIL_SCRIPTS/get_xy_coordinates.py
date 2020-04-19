import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

fig, ax = plt.subplots()
img = cv2.imread('/Users/esedicol/Desktop/Basketball-Shot-Detectection/images/court_extraction2.png')
img  = cv2.resize(img, (500,400))
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
