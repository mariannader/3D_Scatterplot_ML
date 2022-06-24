import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

train = keras.datasets.mnist.load_data()

fig = plt.figure(figsize=(20, 15))
ax = plt.axes(projection='3d')

X = train[0][0][0]
Y = train[0][0][1]
Z = train[0][0][2]

ax.scatter3D(X, Y, Z, color='red')
ax.set_title("3D_scatterplot", pad=25, size=15)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()


