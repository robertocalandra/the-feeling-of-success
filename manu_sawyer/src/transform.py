import numpy as np

"""
This file simply defines the transformation function that take Kinect coordinate to Sawyer coordinates.
"""

# Loading data form calibration
#H = np.load("H.npy")
#t = np.load("t.npy")
H = np.load("/home/manu/ros_ws/src/manu_research/manu_sawyer/src/H.npy")
t = np.load("/home/manu/ros_ws/src/manu_research/manu_sawyer/src/t.npy")

# Transforms (x,y,z) in Kinect coordinate system to
# Sawyer coordinate system
def transform(x,y,z):
    vec = np.array([[x], [y], [z]])
    return H.dot(vec) + t
