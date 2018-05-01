#!/usr/bin/env python
from WSG50_manu import WSG50
import time
import os
import rospy

# run "roslaunch wsg_50_driver wsg_50_tcp_script.launch" in new terminal first

# os.system("for pid in $(ps -ef | grep 'wsg_50_tcp_script' | awk '{print $2}'); do kill -9 $pid; done")
# time.sleep(0.2)
# os.system(
#     'roslaunch wsg_50_driver wsg_50_tcp_script.launch > /home/rcalandra/ros_ws/src/manu_sawyer/temp/wsg_50_driver.txt 2>&1&')
# time.sleep(7)
rospy.init_node('qwe', anonymous=True)

gripper = WSG50()

print("force:" + str(gripper.get_force()))
gripper.set_force(50)
print("force:" + str(gripper.get_force()))
gripper.grasp(pos=5)
print("force:" + str(gripper.get_force()))
