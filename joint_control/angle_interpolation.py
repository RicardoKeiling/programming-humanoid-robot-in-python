'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import hello
import time
from scipy.interpolate import CubicSpline


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.start_time = None

    def think(self, perception):
        # Startzeit beim ersten Aufruf
        if self.start_time is None:
            self.start_time = time.time()
        # interpolierter Zielgelenkwinkel
        target_joints = self.angle_interpolation(self.keyframes, perception)
        # Kopiere fehlende Gelenke
        if 'LHipYawPitch' in target_joints:
            target_joints['RHipYawPitch'] = target_joints['LHipYawPitch']
        # Aktualisiere Sollwerte
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        # Spline-Interpolation

        names, times, keys = keyframes
        target_joints = {}

        # aktuelle Zeit relativ zum Start
        t = time.time() - self.start_time

        for name, joint_times, joint_keys in zip(names, times, keys):
            # Winkel
            angles = [k[0] for k in joint_keys]

            # kubische Spline-Interpolation
            spline = CubicSpline(joint_times, angles, bc_type='natural')

            # Begrenze Zeit
            if t < joint_times[0]:
                target_angle = angles[0]
            elif t > joint_times[-1]:
                target_angle = angles[-1]
            else:
                target_angle = float(spline(t))

            target_joints[name] = target_angle

        return target_joints

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()

