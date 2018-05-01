#!/usr/bin/env python
from geometry_msgs.msg import (
    PoseStamped,
    PointStamped,
    Pose,
    Point,
    Quaternion,
)
import rospy

from sensor_msgs.msg import JointState

from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

from std_msgs.msg import Header


def get_joint_angles(pose, limb="right", seed_cmd=None, use_advanced_options=False, verbosity=0):
    ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()

    # Add desired pose for inverse kinematics
    stamped_pose = stamp_pose(pose)
    ikreq.pose_stamp.append(stamped_pose)
    # Request inverse kinematics from base to "right_hand" link
    ikreq.tip_names.append('right_hand')

    # rospy.loginfo("Running IK Service Client.")

    # seed_joints = None
    if use_advanced_options:
        # Optional Advanced IK parameters
        # The joint seed is where the IK position solver starts its optimization
        ikreq.seed_mode = ikreq.SEED_USER

        # if not seed_joints:
        #     seed = JointState()
        #     seed.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
        #                  'right_j4', 'right_j5', 'right_j6']
        #     seed.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # else:
        seed = joint_state_from_cmd(seed_cmd)
        ikreq.seed_angles.append(seed)


    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return False

    # Check if result valid, and type of seed ultimately used to get solution
    if (resp.result_type[0] > 0):
        seed_str = {
                    ikreq.SEED_USER: 'User Provided Seed',
                    ikreq.SEED_CURRENT: 'Current Joint Angles',
                    ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                   }.get(resp.result_type[0], 'None')
        if verbosity > 0:
            rospy.loginfo("SUCCESS - Valid Joint Solution Found from Seed Type: %s" % (seed_str,))
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        if verbosity > 0:
            rospy.loginfo("\nIK Joint Solution:\n%s", limb_joints)
            rospy.loginfo("------------------")
            rospy.loginfo("Response Message:\n%s", resp)
    else:
        rospy.loginfo("INVALID POSE - No Valid Joint Solution Found.")

    return limb_joints


def get_pose(x, y, z, o): #o should be a quaternion
    p =Pose(
            position=Point(
                x=x,
                y=y,
                z=z,
            ),
        orientation=o
        )

    return p


def stamp_pose(pose):
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    p = PoseStamped(
        header=hdr,
        pose=pose
    )
    return p


def joint_state_from_cmd(cmd):
    js = JointState()
    js.name = cmd.keys()
    js.position = cmd.values()
    return js
