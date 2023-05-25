import numpy as np
from scipy.spatial.transform import Rotation
import math
from pylocogym.envs.rewards.utils.mappings import *

def get_observed_motion(observation_raw):
    d_quat = {}
    d_euler = {}
    joint_angles_offset = 6
    d_quat['root_pos'] = np.array(observation_raw[0:3])
    d_euler['root_pos'] = np.array(observation_raw[0:3])
    d_quat['root_rot'] = euler_to_quaternion(observation_raw[[3, 4, 5]], order='yzx', flip_z=True)
    d_euler['root_rot'] = np.array(observation_raw[[3, 4, 5]])
    d_quat['chest_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['chest_rot']]], order='zyx', flip_z=True)
    d_euler['chest_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['chest_rot']]])
    d_quat['neck_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['neck_rot']]], order='zyx', flip_z=True)
    d_euler['neck_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['neck_rot']]])
    d_quat['rhip_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['rhip_rot']]], order='zxy', flip_z=True)
    d_euler['rhip_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['rhip_rot']]])
    d_quat['rknee_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['rknee_rot']]]
    d_euler['rknee_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['rknee_rot']]]
    d_quat['rankle_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['rankle_rot']] + [0]], order='yxz', flip_z=False)
    d_euler['rankle_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['rankle_rot']] + [0]])
    d_quat['rshoulder_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['rshoulder_rot']]], order='zxy', flip_z=True)
    d_euler['rshoulder_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['rshoulder_rot']]])
    d_quat['relbow_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['relbow_rot']]]
    d_euler['relbow_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['relbow_rot']]]
    d_quat['lhip_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['lhip_rot']]], order='zxy', flip_z=True)
    d_euler['lhip_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['lhip_rot']]])
    d_quat['lknee_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['lknee_rot']]]
    d_euler['lknee_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['lknee_rot']]]
    d_quat['lankle_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['lankle_rot']] + [0]], order='yxz', flip_z=False)
    d_euler['lankle_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['lankle_rot']] + [0]])
    d_quat['lshoulder_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['lshoulder_rot']]], order='zxy', flip_z=True)
    d_euler['lshoulder_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['lshoulder_rot']]])
    d_quat['lelbow_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['lelbow_rot']]]
    d_euler['lelbow_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['lelbow_rot']]]
    return d_quat, d_euler

def quatmult(q0, q1):
    return np.array([-q1[1]*q0[1] - q1[2]*q0[2] - q1[3]*q0[3] + q1[0]*q0[0], q1[1]*q0[0] + q1[2]*q0[3] - q1[3]*q0[2] + q1[0]*q0[1],-q1[1]*q0[3] + q1[2]*q0[0] + q1[3]*q0[1] + q1[0]*q0[2],q1[1]*q0[2] - q1[2]*q0[1] + q1[3]*q0[0] + q1[0]*q0[3]])

def get_quaternion_difference(q1, q2):
    q1_inv = q1*[1,-1,-1,-1]
    return quatmult(np.array(q2),np.array(q1_inv))

def interpolate(target_motion, key, frame_idx, num_frames, loop_motion):
    frame_idx_floor = math.floor(frame_idx)
    offset = frame_idx - frame_idx_floor
    if frame_idx_floor < num_frames - 1:
        # interpolate between frame_idx_floor and frame_idx_floor+1
        r1 = np.array(target_motion[frame_idx_floor][key])
        r2 = np.array(target_motion[frame_idx_floor+1][key])
        return (1-offset)*r1 + offset*r2
    else:
        if loop_motion:
            # interpolate between num_frames-1 and 0
            r1 = np.array(target_motion[num_frames-1][key])
            r2 = np.array(target_motion[0][key])
            return (1-offset)*r1+offset*r2
        else:
            # the last frame
            return np.array(target_motion[num_frames-1][key])

def quaternion_to_euler(wxyz, order='xyz', flip_z=True):
    xyzw = np.roll(wxyz, 3)
    r = Rotation.from_quat(xyzw)
    euler = r.as_euler(order, degrees=False)
    if flip_z:
        idx_z = order.index('z')
        euler[idx_z] *= -1
    return euler

def euler_to_quaternion(euler, order='xyz', flip_z=True):
    if flip_z:
        idx_z = order.index('z')
        euler[idx_z] *= -1
    xyzw = Rotation.from_euler(order, euler, degrees=False).as_quat()
    wxyz = np.roll(xyzw, 1)
    return wxyz

def quat_to_axis_angle(quat):
    theta=0
    sin_theta = math.sqrt(1 - quat[0]**2)
    if sin_theta>0.0001:
        theta=2*math.acos(quat[0])
        theta=NormalizeAngle(theta)
    return theta

def NormalizeAngle(theta):
    theta=theta%(2*math.pi)
    if theta>math.pi:
        theta-=2*math.pi
    elif theta<-math.pi:
        theta+=2*math.pi
    return theta


class ObservationData:
    def __init__(self, observation_raw, num_joints, is_obs_fullstate=True):
        self.observation = observation_raw
        self.num_obs = len(observation_raw)
        self.num_additional_obs = self.num_obs - (num_joints * 2 + 9) + 3 * is_obs_fullstate
        self.is_fullstate = is_obs_fullstate

        if is_obs_fullstate:
            """Fully observed observation convention:
                    observation = [ global position, Euler angles, joint angles,
                                    global velocity, angular velocity, joint velocity,
                                    additional observation elements]
                                    
                    position convention: x = left, y = up, z = forward
                    Euler angle convention: (yaw, pitch, roll)
            """

            # position:
            self.pos = observation_raw[0:3]  # x = left, y = up, z = forward
            self.x = observation_raw[0]
            self.y = observation_raw[1]
            self.z = observation_raw[2]

            # orientation:
            self.ori = observation_raw[3:6]  # Euler angles (yaw, pitch, roll)
            self.yaw = observation_raw[3]
            self.pitch = observation_raw[4]
            self.roll = observation_raw[5]

            # joint angles:
            self.joint_angles = observation_raw[6:6 + num_joints]

            # linear velocity:
            self.vel = observation_raw[6 + num_joints:9 + num_joints]  # in global coordinates
            rotation_to_local = Rotation.from_euler('yxz', -self.ori)
            self.local_vel = rotation_to_local.apply(self.vel)

            # angular velocity:
            self.ang_vel = observation_raw[9 + num_joints:12 + num_joints]  # in global coordinates

            # joint velocity:
            self.joint_vel = observation_raw[12 + num_joints:12 + 2 * num_joints]

        else:
            """Partially observed observation convention:
                   observation = [ base height (y), pitch, roll, joint angles,
                                   local velocity, angular velocity, joint velocity,
                                   additional observation elements]
           """

            # position:
            self.y = observation_raw[0]

            # orientation:
            self.pitch = observation_raw[1]
            self.roll = observation_raw[2]

            # joint angles:
            self.joint_angles = observation_raw[3:3 + num_joints]

            # linear velocity:
            self.local_vel = observation_raw[3 + num_joints: 6 + num_joints]  # in local coordinates

            # angular velocity:
            self.ang_vel = observation_raw[6 + num_joints:9 + num_joints]  # in local coordinates

            # joint velocity:
            self.joint_vel = observation_raw[9 + num_joints:9 + 2 * num_joints]

        def print_obs():
            print(observation_raw)


def calc_rms(signal_vector):
    """ Given a vector of signals, returns a vector of RMS (Root Mean Squared) of signals"""
    return np.sqrt(np.mean(signal_vector ** 2, axis=1))


def calc_max_abs(signal_vector):
    """ Given a vector of signals, returns a vector of the elements with max magnitude for each signal"""
    return np.absolute(signal_vector).max(1)


def calc_rms_torque(torques, num_joints):
    """ Given a vector of signals, returns a vector of RMS (Root Mean Squared) of signals"""
    rms_torque = np.zeros(num_joints)
    num_sim_steps_per_loop = int(len(torques) / num_joints)
    for i in range(num_sim_steps_per_loop):
        rms_torque += np.power(torques[i * num_joints:(i + 1) * num_joints], 2)
    rms_torque = np.sqrt(rms_torque / num_sim_steps_per_loop)
    return rms_torque


def calc_max_mag_torque(torques, num_joints):
    """ Given a vector of signals, returns a vector of the elements with max magnitude for each signal"""
    max_mag_torque = np.zeros(num_joints)
    num_sim_steps_per_loop = int(len(torques) / num_joints)
    for i in range(num_sim_steps_per_loop):
        for joint_idx in range(num_joints):
            if abs(torques[i * num_joints + joint_idx]) > abs(max_mag_torque[joint_idx]):
                max_mag_torque[joint_idx] = torques[i * num_joints + joint_idx]
    return max_mag_torque


def tail(vector, segment_length):
    return vector[len(vector) - segment_length:]


def calc_derivatives(buffer, dt, num_elements):
    """ Calculate first and second derivative given history buffer.
    buffer = [current, previous, past]
    """
    current = buffer[0:num_elements]
    past = buffer[num_elements:2 * num_elements]
    past_past = buffer[num_elements * 2: 3 * num_elements]
    x_dot = (current - past) / dt
    x_ddot = (current - 2 * past + past_past) / dt ** 2
    return x_dot, x_ddot
