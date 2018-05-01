# Research

This is the code used in the paper:
```

```

# Hardware Setting
- Sawyer 7-DOF arm (from now on called Nordri)
- 2 Kinects
- 2 GelSight Sensors

# Software Dependencies

The following packages need to be installed:
- Python
- Tensorflow
- ROS
- The intera sdk need to be installed
- Install https://github.com/OpenKinect/libfreenect2
- Install https://github.com/code-iai/iai_kinect2

# Run code

```
cd  ~/ros_ws
./nordri.sh
roslaunch wsg_50_driver wsg_50_tcp_script.launch
```

```
cd  ~/ros_ws
./nordri.sh
rosrun kinect2_bridge kinect2_bridge
```

```
cd  ~/ros_ws
./nordri.sh
roslaunch manu_sawyer gelsightA_driver.launch
``` 

```
cd  ~/ros_ws
./nordri.sh
roslaunch manu_sawyer gelsightB_driver.launch
```


# HOW TO COLLECT DATA:

- Turn on the robot, gripper, and GelSights (A first and then B, always)    
- open `run_experiment_nordri.py` and select the `name` variable (approx. line 62) so that it corresponds to the object that you what to use.      
- open `run_experiment_nordri.py` and set the `lower_bound_table_dist` variable (approx. line 57) to an appropriate value. Taller objects will require about 0.05 while smaller can be set as low as 0.015.       
- Run `run_experiment_nordri.py` in intera mode    
- Follow the instructions printed

```
cd  ~/ros_ws
./nordri.sh
rosrun manu_sawyer run_experiment_nordri.py
```

### Common problems/errors and fixes:

Problem: How to end the script?    
Solution: Run `killall -9 python` in a new terminal, and delete the two last files in the `Data` directory, if applicable.     

Problem: The GelSightA topics stops, i.e. the terminal where the GelSightA launch file was run gives an error.    
Solution: End the script `run_experiment_nordri.py` and restart the GelSightA. Delete the effected files in the `Data` directory. Then rerun `run_experiment_nordri.py`.

