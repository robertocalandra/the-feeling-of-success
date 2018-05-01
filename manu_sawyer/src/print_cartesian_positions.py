#!/usr/bin/env python

import rospy

from sensor_msgs.msg import JointState

from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)

import intera_interface

class Print_fwd_kin(object):
    def __init__(self, rate):

        print("Initializing node... ")
        rospy.init_node("cartestian_joint_printer")

        self.limb = intera_interface.Limb("right")

        limb = 'right'
        self.name_of_service = "ExternalTools/" + limb + "/PositionKinematicsNode/FKService"
        self.fksvc = rospy.ServiceProxy(self.name_of_service, SolvePositionFK)


        self.timer = rospy.Timer(rospy.Duration.from_sec(1.0 / rate), self.run_fwd_kin)



    def run_fwd_kin(self, event):

        fkreq = SolvePositionFKRequest()
        joints = JointState()
        joints.name = self.limb.joint_names()

        joints.position = [self.limb.joint_angle(j)
                        for j in joints.name]

        # Add desired pose for forward kinematics
        fkreq.configuration.append(joints)
        fkreq.tip_names.append('right_hand')


        try:
            rospy.wait_for_service(self.name_of_service, 5)
            resp = self.fksvc(fkreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False

        # Check if result valid
        if (resp.isValid[0]):
            rospy.loginfo("\nFK Joint Angles:\n")
            for i, name in enumerate(joints.name):
                print name + " %f"%(joints.position[i])

            # ADDED CODE
            pose_dict = self.limb.endpoint_pose()
            pose_pos = pose_dict['position']
            pose_orientation = pose_dict['orientation']
            print ("POSITION INFO:")
            for i in pose_pos:
                print (i)
            print ("ORIENTATION INFO:")
            for i in pose_pos:
                print (i)
            # END OF ADDED CODE

            rospy.loginfo("\nFK Cartesian Solution:\n")
            rospy.loginfo("------------------")
            rospy.loginfo("Response Message:\n%s", resp)
        else:
                rospy.loginfo("INVALID JOINTS - No Cartesian Solution Found.")


def main():
    """
    printing the
    response of whether a valid Cartesian solution was found,
    and if so, the corresponding Cartesian pose.
    """

    joint_printer = Print_fwd_kin(rate= 1)
    rospy.spin()


if __name__ == '__main__':
    main()
