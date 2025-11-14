'''In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

'''


from angle_interpolation import AngleInterpolationAgent
from keyframes import hello, wipe_forehead
import pickle
from os import path


class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PostureRecognitionAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.posture = 'unknown'
        self.last_posture = 'unknown'
    
        ROBOT_POSE_CLF = 'robot_pose.pkl'
        self.posture_classifier = pickle.load(open(ROBOT_POSE_CLF, 'rb'))
        
        self.classes = ['Back', 'Belly', 'Crouch', 'Frog', 'HeadBack','Knee', 'Left', 'Right', 'Sit', 'Stand', 'StandInit']

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        if self.posture != self.last_posture:
            print(f"=== POSTURE CHANGED: {self.last_posture} -> {self.posture} ===")
            self.last_posture = self.posture
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = 'unknown'

        # Feature-Vektor
        features = [
            perception.joint['LHipYawPitch'],
            perception.joint['LHipRoll'],
            perception.joint['LHipPitch'],
            perception.joint['LKneePitch'],
            perception.joint['RHipYawPitch'],
            perception.joint['RHipRoll'],
            perception.joint['RHipPitch'],
            perception.joint['RKneePitch'],
            perception.imu[0],  # AngleX
            perception.imu[1]   # AngleY
        ]
        
        # Prediction
        if self.posture_classifier is not None:
            prediction = self.posture_classifier.predict([features])[0]
            posture = self.classes[prediction]
        
        return posture

if __name__ == '__main__':
    agent = PostureRecognitionAgent()
    agent.keyframes = wipe_forehead(0)  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
