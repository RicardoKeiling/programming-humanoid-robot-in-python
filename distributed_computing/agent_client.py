'''In this file you need to implement remote procedure call (RPC) client

* The agent_server.py has to be implemented first (at least one function is implemented and exported)
* Please implement functions in ClientAgent first, which should request remote call directly
* The PostHandler can be implement in the last step, it provides non-blocking functions, e.g. agent.post.execute_keyframes
 * Hints: [threading](https://docs.python.org/2/library/threading.html) may be needed for monitoring if the task is done
'''

import grpc
import numpy as np

import nao_pb2
import nao_pb2_grpc


class NaoClient:

    def __init__(self, address="localhost:50051"):
        channel = grpc.insecure_channel(address)
        self.stub = nao_pb2_grpc.NaoServiceStub(channel)

    def get_joint_angle(self, joint_name):
        response = self.stub.GetJointAngle(
            nao_pb2.JointRequest(joint_name=joint_name)
        )
        return response.angle

    def set_joint_angle(self, joint_name, angle):
        response = self.stub.SetJointAngle(
            nao_pb2.SetJointAngleRequest(
                joint_name=joint_name,
                angle=angle
            )
        )
        return response.success

    def set_effector_transform(self, effector_name, transform):
        matrix = nao_pb2.Matrix4x4(
            data=list(np.array(transform).flatten())
        )

        response = self.stub.SetEffectorTransform(
            nao_pb2.SetTransformRequest(
                effector_name=effector_name,
                transform=matrix
            )
        )
        return response.success, response.message
