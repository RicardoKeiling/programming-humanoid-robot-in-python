'''
In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''

import numpy as np
from numpy import array, zeros, matmul, transpose
from numpy.linalg import norm, inv
from numpy.matlib import identity

from forward_kinematics import ForwardKinematicsAgent


class InverseKinematicsAgent(ForwardKinematicsAgent):

    def compute_jacobian(self, chain_joints, joint_state):
        '''
        compute numerical Jacobian for a given kinematic chain
        '''

        num_joints = len(chain_joints)
        jacobian = zeros((6, num_joints))

        # compute current end-effector position
        self.forward_kinematics(joint_state)
        end_effector = chain_joints[-1]
        end_transform = self.transforms[end_effector]
        base_position = end_transform[:3, 3]

        epsilon = 1e-6

        for idx, joint_name in enumerate(chain_joints):
            # copy current joint configuration
            perturbed_state = joint_state.copy()
            base_angle = float(joint_state.get(joint_name, 0.0))
            perturbed_state[joint_name] = base_angle + epsilon

            # forward kinematics with perturbed angle
            self.forward_kinematics(perturbed_state)
            perturbed_transform = self.transforms[end_effector]
            perturbed_position = perturbed_transform[:3, 3]

            # numerical derivative (linear velocity part)
            delta_position = (perturbed_position - base_position) / epsilon
            jacobian[:3, idx] = delta_position.flatten()

            # angular velocity part unused
            jacobian[3:, idx] = 0.0

        return jacobian

    def inverse_kinematics(self, effector_name, transform):
        '''
        solve inverse kinematics using damped least squares
        '''

        solution_angles = []

        max_iterations = 1000
        convergence_tol = 1e-4
        damping_factor = 0.001

        chain_joints = self.chains.get(effector_name, [])
        if not chain_joints:
            print(f"Unknown effector name: {effector_name}")
            return solution_angles

        # initialize joint angles from perception
        joint_state = {
            joint: float(self.perception.joint.get(joint, 0.0))
            for joint in self.joint_names
        }

        target_position = np.array(transform[:3, 3]).flatten()

        for _ in range(max_iterations):
            # forward kinematics with current estimate
            self.forward_kinematics(joint_state)

            end_joint = chain_joints[-1]
            current_transform = self.transforms[end_joint]
            current_position = np.array(current_transform[:3, 3]).flatten()

            position_error = target_position - current_position
            error_norm = norm(position_error)

            if error_norm < convergence_tol:
                break

            jacobian_full = self.compute_jacobian(chain_joints, joint_state)
            jacobian_pos = jacobian_full[:3, :]

            # damped least squares inverse
            JJt = matmul(jacobian_pos, transpose(jacobian_pos))
            damping_matrix = damping_factor * identity(3)
            jacobian_damped_inv = matmul(
                transpose(jacobian_pos),
                inv(JJt + damping_matrix)
            )

            delta_angles = matmul(jacobian_damped_inv, position_error)
            delta_angles = np.array(delta_angles).flatten()

            # update joint configuration
            for idx, joint_name in enumerate(chain_joints):
                joint_state[joint_name] = float(
                    joint_state[joint_name] + delta_angles[idx]
                )

        solution_angles = [joint_state[joint] for joint in chain_joints]
        return solution_angles

    def set_transforms(self, effector_name, transform):
        '''
        compute inverse kinematics and create keyframes
        '''

        joint_angles = self.inverse_kinematics(effector_name, transform)
        chain_joints = self.chains.get(effector_name, [])

        if joint_angles and chain_joints:
            times = [0]
            names = [chain_joints]
            angles = [joint_angles]
            self.keyframes = (times, names, angles)
        else:
            self.keyframes = ([], [], [])


if __name__ == '__main__':
    agent = InverseKinematicsAgent()

    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26

    agent.set_transforms('LLeg', T)

    print("=== DEBUG: angle_interpolation start ===")
    print("keyframes[0] (times):", agent.keyframes[0])
    print("keyframes[1] (joint names):", agent.keyframes[1])
    print("keyframes[2] (angles):", agent.keyframes[2])

    agent.run()