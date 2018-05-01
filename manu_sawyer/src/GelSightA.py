#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import multiprocessing


class GelSightA:
    def __init__(self, topic='/gelsightA/image_raw'):  # '/gelsightA/image_raw' /image_view_A/output
        # Variable
        self.img = None

        # Used to convert image from ROS to cv2
        self.bridge = CvBridge()

        # The subscriber
        self.subscriber = rospy.Subscriber(topic, Image, self.update_image)

        def spin_thread():
            rospy.spin()

        self.gelsight_process = multiprocessing.Process(target=spin_thread)
        self.gelsight_process.start()

    def get_image(self):
        return self.img

    def update_image(self, data):
        self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def end_process(self):
        self.gelsight_process.terminate()
        self.gelsight_process.join()
