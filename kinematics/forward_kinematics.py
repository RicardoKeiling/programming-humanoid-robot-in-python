'''
In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    1. the local_trans has to consider different joint axes and link parameters for different joints
    2. Please use radians and meters as unit.
'''

# add PYTHONPATH
import os
import sys
from numpy.matlib import *

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))
from recognize_posture import PostureRecognitionAgent


class ForwardKinematicsAgent(PostureRecognitionAgent):

    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):

        super(ForwardKinematicsAgent, self).__init__(
            simspark_ip, simspark_port, teamname, player_id, sync_mode
        )

        # global transforms for each joint
        self.transforms = {name: identity(4) for name in self.joint_names}

        # kinematic chains
        self.chains = {
            "Head": ["HeadYaw", "HeadPitch"],
            "LArm": ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"],
            "RArm": ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"],
            "LLeg": ["LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll"],
            "RLeg": ["RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll"]
        }

        # link translations in meters
        self.translation = {
            # head
            "HeadYaw": [0.0, 0.0, 0.1265],
            "HeadPitch": [0.0, 0.0, 0.0],

            # left arm
            "LShoulderPitch": [0.0, 0.098, 0.100],
            "LShoulderRoll": [0.0, 0.0, 0.0],
            "LElbowYaw": [0.105, 0.015, 0.0],
            "LElbowRoll": [0.0, 0.0, 0.0],
            "LWristYaw": [0.05595, 0.0, 0.0],

            # right arm
            "RShoulderPitch": [0.0, -0.098, 0.100],
            "RShoulderRoll": [0.0, 0.0, 0.0],
            "RElbowYaw": [0.105, -0.015, 0.0],
            "RElbowRoll": [0.0, 0.0, 0.0],
            "RWristYaw": [0.05595, 0.0, 0.0],

            # left leg
            "LHipYawPitch": [0.0, 0.05, -0.085],
            "LHipRoll": [0.0, 0.0, 0.0],
            "LHipPitch": [0.0, 0.0, 0.0],
            "LKneePitch": [0.0, 0.0, -0.1],
            "LAnklePitch": [0.0, 0.0, -0.1029],
            "LAnkleRoll": [0.0, 0.0, 0.0],

            # right leg
            "RHipYawPitch": [0.0, -0.05, -0.085],
            "RHipRoll": [0.0, 0.0, 0.0],
            "RHipPitch": [0.0, 0.0, 0.0],
            "RKneePitch": [0.0, 0.0, -0.1],
            "RAnklePitch": [0.0, 0.0, -0.1029],
            "RAnkleRoll": [0.0, 0.0, 0.0]
        }

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_translation(self, joint_name, joint_angle):
        """
        calculate local transformation of one joint
        """

        # initialize transformation
        transform = identity(4)

        c = cos(joint_angle)
        s = sin(joint_angle)

        # rotation matrices
        rot_x = array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s,  c, 0],
            [0, 0, 0, 1]
        ])

        rot_y = array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]
        ])

        rot_z = array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

        # choose rotation based on joint type
        if joint_name.endswith("Pitch"):
            transform = rot_y
        elif joint_name.endswith("Roll"):
            transform = rot_x
        elif joint_name.endswith("Yaw"):
            transform = rot_z
        else:
            print(f"Joint name error: {joint_name}")
            return identity(4)

        # apply translation
        offset = self.translation[joint_name]
        transform[0, 3] = offset[0]
        transform[1, 3] = offset[1]
        transform[2, 3] = offset[2]

        return transform

    def forward_kinematics(self, joints):
        """
        compute forward kinematics for all chains
        """

        for chain in self.chains.values():
            global_transform = identity(4)

            for joint_name in chain:
                joint_angle = joints.get(joint_name, 0.0)
                local_tf = self.local_translation(joint_name, joint_angle)

                global_transform = global_transform @ local_tf
                self.transforms[joint_name] = global_transform


if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
