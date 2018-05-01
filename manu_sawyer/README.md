# Sawyer

To init a console use:
```
cd /home/guser/catkin_ws
./intera.sh 
```

To run a piece of code use:
```
rosrun manu_sawyer <namefile>.py
```

To build after a change in the code use:
```
catkin_make
```

To mark file as executable:
go to directory of file
```
chmod +x <namefile>.py
```

To reset after an emergency stop use:
```
rosrun intera_interface enable_robot.py -e

```

To install libuvc_camera in ros (for using gelsight):
```
sudo apt-get install ros-indigo-libuvc-camera
rosdep install libuvc_camera --os=ubuntu:trusty
```

To launch necessary rospy processes: (need to sudo ./intera.sh for gelsight driver)
```
roslaunch manu_sawyer gelsight_driver.launch
roslaunch kinect2_bridge kinect2_bridge.launch
```


# Calibration of KinectA 

Find the coordinates needed for calibration in "src/*_calibration_info.txt"

To obtain the coordinates in the Sawyer frame:
Put the EE right above the calibration points marked on the table and run "src/print_cartesian_position./py"

To obtain the coordinates in the KinectA frame:
Put a small cylindrical object at the center of the calibration points marked on the table and run "scr/grip_and_record/transform_test.py"

Supply/input all the coordinates to "src/calibrate.py".

Use "src/calibrate.py" to find an affine transformation based on these points and the "transform.py" to actually use the transfrom.