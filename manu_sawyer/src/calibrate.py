
import numpy as np
from cvxpy import *

"""
Script used to find H and t for the transformation
x_Sawyer = H x_Kinect + t
"""

# The nine calibration points in the Kinect coordinate system
p_1_kinect = np.array([0.31798916,  0.00325601,  0.76754364]).reshape((3, 1))
p_2_kinect = np.array([-0.06495789, -0.00613411,  0.78532697]).reshape((3, 1))
p_3_kinect = np.array([-0.42200078,  0.01948896,  0.75328325]).reshape((3, 1))
p_4_kinect = np.array([0.32931394, -0.12782843,  0.95595643]).reshape((3, 1))
p_5_kinect = np.array([-0.0298477 , -0.13292019,  0.95944226]).reshape((3, 1))
p_6_kinect = np.array([-0.40676888, -0.12317579,  0.95006945]).reshape((3, 1))
p_7_kinect = np.array([0.33622021, -0.22669912,  1.09418477]).reshape((3, 1))
p_8_kinect = np.array([-0.02905334, -0.23861358,  1.11272965]).reshape((3, 1))
p_9_kinect = np.array([-0.40161079, -0.2231759,  1.09171911]).reshape((3, 1))


p_kinect = [p_1_kinect, p_2_kinect, p_3_kinect, p_4_kinect, p_5_kinect, p_6_kinect, p_7_kinect, p_8_kinect, p_9_kinect]
#p_kinect = [p_1_kinect, p_2_kinect, p_3_kinect, p_4_kinect, p_5_kinect, p_6_kinect]

# The nine corresponding calibration points in the Sawyer coordinate system
p_1_sawyer = np.array([0.643367191677, -0.350339791193, 0.00328037006498]).reshape((3, 1))
p_2_sawyer = np.array([0.6332503943, 0.0292459324126, 0.00330423965907]).reshape((3, 1))
p_3_sawyer = np.array([0.680385933273, 0.37847117307, 0.000283238475192]).reshape((3, 1))
p_4_sawyer = np.array([0.416538421332,  -0.35834409086, 0.00317884107226]).reshape((3, 1))
p_5_sawyer = np.array([0.423452822173, -0.00949250782561,  0.0037321160006]).reshape((3, 1))
p_6_sawyer = np.array([0.442115979626, 0.35916693997,  0.00201009508294]).reshape((3, 1))
p_7_sawyer = np.array([0.248284622296, -0.36458168206,  0.00275356155022]).reshape((3, 1))
p_8_sawyer = np.array([0.24454903839, -0.00529264843613,  0.00249021303604]).reshape((3, 1))
p_9_sawyer = np.array([0.268852273011, 0.351922785944,  0.00141867116927]).reshape((3, 1))


p_sawyer = [p_1_sawyer, p_2_sawyer, p_3_sawyer, p_4_sawyer, p_5_sawyer, p_6_sawyer, p_7_sawyer, p_8_sawyer, p_9_sawyer]
#p_sawyer = [p_1_sawyer, p_2_sawyer, p_3_sawyer, p_4_sawyer, p_5_sawyer, p_6_sawyer]

# Optimization variables
H = Variable(3, 3)
t = Variable(3)

# Optimization constraints
constraints = []

# Optimization objective
temp = []
for i in range(len(p_sawyer)):
        temp.append(norm(H * p_kinect[i] + t - p_sawyer[i]))
objective = Minimize(sum(temp))

# Solve optimization problem
prob = Problem(objective, constraints)
prob.solve()

print("H:\n", H.value)
print("t:\n", t.value)

np.save("H", H.value)
np.save("t", t.value)