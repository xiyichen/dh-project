"""
Computing reward for Vanilla setup, constant target speed, gaussian kernels
"""
import numpy as np
from numpy.linalg import norm
from pylocogym.envs.rewards.utils.utils import *

import pickle
import time
import math

def squared_norm(v):
    res = 0
    for e in v:
        res += e**2
    return res

def get_phase(dt, i, duration, num_frames, loop_motion):
    elapsed_time = dt*i
    phase = elapsed_time / (duration * num_frames)
    # check if it's a loop motion
    if phase > 1:
        if loop_motion:
            phase -= math.floor(phase)
        else:
            phase = 1
    return phase

def euler_to_quaternion(euler):
    r = Rotation.from_euler('xyz', euler, degrees=False)
    return r.as_quat()

# joint_mappings = {
#     '0 lowerback_x
# 1 lHip_1
# 2 rHip_1
# 3 lowerback_y
# 4 lHip_2
# 5 rHip_2
# 6 lowerback_z
# 7 lHip_torsion
# 8 rHip_torsion
# 9 upperback_x
# 10 lKnee
# 11 rKnee
# 12 upperback_y
# 13 lAnkle_1
# 14 rAnkle_1
# 15 upperback_z
# 16 lAnkle_2
# 17 rAnkle_2
# 18 lowerneck_x
# 19 lScapula_y
# 20 rScapula_y
# 21 lToeJoint
# 22 rToeJoint
# 23 lowerneck_y
# 24 lScapula_z
# 25 rScapula_z
# 26 lowerneck_z
# 27 lShoulder_1
# 28 rShoulder_1
# 29 upperneck_x
# 30 lShoulder_2
# 31 rShoulder_2
# 32 upperneck_y
# 33 lShoulder_torsion
# 34 rShoulder_torsion
# 35 upperneck_z
# 36 lElbow_flexion_extension
# 37 rElbow_flexion_extension
# 38 lElbow_torsion
# 39 rElbow_torsion
# 40 lWrist_x
# 41 rWrist_x
# 42 lWrist_z
# 43 rWrist_z
# '
# }

def get_observed_motion(observation_raw):
    d = {}
    d['root_pos'] = observation_raw[0:3]
    d['root_rot'] = euler_to_quaternion(observation_raw[3:6])
    d['chest_rot'] = euler_to_quaternion(observation_raw[[9, 12, 15]])
    d['neck_rot'] = euler_to_quaternion(observation_raw[[0, 3, 6]])
    d['rhip_rot'] = euler_to_quaternion(observation_raw[[2, 8, 5]])
    d['rknee_rot'] = observation_raw[11]
    d['rankle_rot'] = euler_to_quaternion([observation_raw[14], 0, observation_raw[17]])
    d['rshoulder_rot'] = euler_to_quaternion(observation_raw[[28, 34, 31]])
    # d['relbow_rot'] = euler_to_quaternion([observation_raw[37], observation_raw[39], 0])
    d['relbow_rot'] = observation_raw[37]
    d['lhip_rot'] = euler_to_quaternion(observation_raw[[1, 7, 4]])
    d['lknee_rot'] = observation_raw[10]
    d['lankle_rot'] = euler_to_quaternion([observation_raw[13], 0, observation_raw[16]])
    d['lshoulder_rot'] = euler_to_quaternion(observation_raw[[27, 33, 30]])
    # d['lelbow_rot'] = euler_to_quaternion([observation_raw[36], observation_raw[38], 0])
    d['lelbow_rot'] = observation_raw[36]
    return d

def get_quaternion_difference(q1, q2):
    q1_inv = Rotation.from_quat(q1).inv().as_quat()
    return np.array(q2).dot(np.array(q1_inv))

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
        

def compute_reward(observation_raw, dt, num_joints, params, feet_status, all_torques, action_buffer, is_obs_fullstate,
                   joint_angles_default, nominal_base_height, target_motion, loop_motion):
    
    """
    Compute the reward based on observation (Vanilla Environment).

    :param observation_raw: current observation
    :param dt: control time step size
    :param num_joints: number of joints
    :param params: reward params read from the config file
    :param feet_status: pos, vel and swing status of robot feet
    :param all_torques: torque records during the last control timestep
    :param action_buffer: history of previous actions
    :param is_obs_fullstate: flag to choose full state obs or not.
    :param joint_angles_default: default joint angles
    :return: total reward, reward information (different terms can be passed here to be plotted in the graphs)
    """

    # test_data = {'observation_raw': observation_raw, 'dt': dt, 'num_joints': num_joints, 'params': params,
    #              'feet_status': feet_status, 'all_torques': all_torques, 'action_buffer': action_buffer,
    #              'is_obs_fullstate': is_obs_fullstate, 'joint_angles_default': joint_angles_default,
    #              'nominal_base_height': nominal_base_height}
    
    with open('/local/home/xiychen/Documents/dh-project/n_steps.txt', 'r') as f:
        i = int(f.readline())
    
    num_frames = len(target_motion.keys())
    duration = target_motion[0]['duration'][0]
    phase = get_phase(dt, i, duration, num_frames, loop_motion)
    frame_idx = phase * num_frames # [0, num_frames]
    observed_motion = get_observed_motion(observation_raw)
    pose_reward = 0
    for key in observed_motion:
        if key == 'root_pos':
            continue
        elif key in ['lknee_rot', 'rknee_rot', 'lelbow_rot', 'relbow_rot']:
            r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)[0]
            pose_reward += (observed_motion[key] - r_target)**2
        else:
            r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            pose_reward += get_quaternion_difference(observed_motion[key], r_target)
    pose_reward = np.exp(-2 * pose_reward)
    print(pose_reward)
    exit()
    
    
    # print(frame_idx)

    observation = ObservationData(observation_raw, num_joints, is_obs_fullstate)
    
    

    action_dot, action_ddot = calc_derivatives(action_buffer, dt, num_joints)
    cmd_fwd_vel = params.get("fwd_vel_cmd", 1.0)
    torque = tail(all_torques, num_joints)

    # =============
    # define cost/reward terms here:
    # =============
    # TODO: Implement the rewards here.
    # Hints:
    # - Use function params.get("weight_velocity", 0) to get the value of parameters set in the .conf file.
    
    # forward_vel_reward = params.get("weight_velocity", 0) * np.exp(- (abs(observation.local_vel[2] - cmd_fwd_vel)**2) / (2*params.get("sigma_velocity", 0)**2))

    # height_reward = params.get("weight_height", 0) * np.exp(- abs(observation.y - nominal_base_height)**2/(2*params.get("sigma_height", 0)**2))

    # attitude_reward = params.get("weight_attitude", 0) * np.exp(-(abs(observation.roll)**2+abs(observation.pitch)**2)/(4*params.get("sigma_attitude", 0)**2))

    # torque_reward = params.get("weight_torque", 0) * np.exp(-squared_norm(torque)/(2*num_joints*params.get("sigma_torque", 0)**2))
        
    # smoothness1_reward = params.get("weight_smoothness1", 0) * np.exp(-squared_norm(action_dot)/(2*num_joints*params.get("sigma_smoothness1", 0)**2))
        
    # smoothness2_reward = params.get("weight_smoothness2", 0) * np.exp(-squared_norm(action_ddot)/(2*num_joints*params.get("sigma_smoothness2", 0)**2))

    # joint_reward = params.get("weight_joints", 0) * np.exp(-squared_norm(observation.joint_angles - joint_angles_default)/(2*num_joints*params.get("sigma_joints", 0)**2))

    
    
    
    # =============
    # sum up rewards
    # =============
    smoothness_reward = params.get("weight_smoothness", 0) * (smoothness1_reward + smoothness2_reward)
    reward = forward_vel_reward + smoothness_reward + torque_reward + height_reward + attitude_reward + joint_reward

    info = {
        "forward_vel_reward": forward_vel_reward,
        "height_reward": height_reward,
        "attitude_reward": attitude_reward,

        "torque_reward": torque_reward,
        "smoothness1_reward": smoothness1_reward,
        "smoothness2_reward": smoothness2_reward,
        "smoothness_reward": smoothness_reward,

        "joint_reward": joint_reward
    }

    return reward, info


def punishment(current_step, max_episode_steps, params):  # punishment for early termination
    penalty = params['weight_early_penalty'] * (max_episode_steps - current_step)
    return penalty
