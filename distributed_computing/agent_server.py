'''In this file you need to implement remote procedure call (RPC) server

* There are different RPC libraries for python, such as xmlrpclib, json-rpc. You are free to choose.
* The following functions have to be implemented and exported:
 * get_angle
 * set_angle
 * get_posture
 * execute_keyframes
 * get_transform
 * set_transform
* You can test RPC server with ipython before implementing agent_client.py
'''

# add PYTHONPATH
import grpc
from concurrent import futures
import numpy as np

import nao_pb2
import nao_pb2_grpc


class NaoService(nao_pb2_grpc.NaoServiceServicer):

    def __init__(self, agent):
        self.agent = agent

    def GetJointAngle(self, request, context):
        angle = float(self.agent.perception.joint.get(request.joint_name, 0.0))
        return nao_pb2.JointAngleResponse(angle=angle)

    def SetJointAngle(self, request, context):
        self.agent.set_joint_angle(request.joint_name, request.angle)
        return nao_pb2.StatusResponse(success=True, message="Joint angle set")

    def SetEffectorTransform(self, request, context):
        if len(request.transform.data) != 16:
            return nao_pb2.StatusResponse(
                success=False,
                message="Transform must contain 16 values"
            )

        T = np.array(request.transform.data).reshape((4, 4))
        self.agent.set_transforms(request.effector_name, T)

        return nao_pb2.StatusResponse(
            success=True,
            message="Transform applied"
        )


def serve(agent):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nao_pb2_grpc.add_NaoServiceServicer_to_server(
        NaoService(agent), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()
