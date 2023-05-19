"""
Computing reward for Vanilla setup, constant target speed, gaussian kernels
"""
import numpy as np
from numpy.linalg import norm
from pylocogym.envs.rewards.utils.utils import *
from pylocogym.envs.rewards.utils.mappings import *

import pickle
import time
import math

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
    d_quat['lshoulder_rot'] = euler_to_quaternion([joint_angles_offset+x for x in rot_mappings['lshoulder_rot']], order='zxy', flip_z=True)
    d_euler['lshoulder_rot'] = np.array([joint_angles_offset+x for x in rot_mappings['lshoulder_rot']])
    d_quat['lelbow_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['lelbow_rot']]]
    d_euler['lelbow_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['lelbow_rot']]]
    return d_quat, d_euler

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
                   joint_angles_default, nominal_base_height, target_motion, loop_motion, phase, num_loops_passed):
    
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
    
    observation = ObservationData(observation_raw, num_joints, is_obs_fullstate)
    
    num_frames = len(target_motion.keys())
    frame_idx = phase * num_frames # [0, num_frames]
    
    observed_motion_quat, observed_motion_euler = get_observed_motion(observation_raw)
    pose_reward = 0
    for key in observed_motion_quat:
        if key == 'root_pos':
            continue
        elif key in['lankle_rot', 'rankle_rot']:
            r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            euler_target = quaternion_to_euler(r_target, order='yxz', flip_z=False)
            diff = abs(euler_target - observed_motion_euler[key])**2
            pose_reward += (diff[0]+diff[1])
        elif key in ['lknee_rot', 'rknee_rot', 'lelbow_rot', 'relbow_rot']:
            r_target = -interpolate(target_motion, key, frame_idx, num_frames, loop_motion)[0]
            pose_reward += (observed_motion_quat[key][0] - r_target)**2
        else:
            r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            pose_reward += get_quaternion_difference(observed_motion_quat[key], r_target)**2
    pose_reward = np.exp(-2 * pose_reward)
    
    velocity_reward = 0
    for key in observed_motion_euler:
        if key == 'root_pos':
            v_target = (target_motion[(math.floor(frame_idx)+1)%num_frames]['root_pos'] - target_motion[math.floor(frame_idx)]['root_pos'])/dt # zyx
            v_observed = observation.vel # xyz
            v_observed = v_observed[[2,1,0]]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).sum()
        elif key == 'root_rot':
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames]['root_rot'], order='yzx', flip_z=True) - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)]['root_rot'], order='yxz', flip_z=True)) / dt
            v_observed = observation.ang_vel
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).sum()
        elif key in ['lknee_rot', 'rknee_rot', 'lelbow_rot', 'relbow_rot']:
            v_target = (target_motion[math.floor(frame_idx)][key] - target_motion[(math.floor(frame_idx)+1)%num_frames][key])[0]/dt # knee and elbow rotations are flipped when mapping from deepmimic to bob
            v_observed = observation_raw[[12+x for x in rot_mappings[key]]][0]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).sum()
        elif key in ['lankle_rot', 'rankle_rot']:
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='yxz', flip_z=False) - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)][key], order='yxz', flip_z=False)) / dt
            v_target = v_target[:2]
            v_observed = observation_raw[[12+x for x in rot_mappings[key]]]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).sum()
        elif key in ['chest_rot', 'neck_rot']:
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='zyx', flip_z=True) - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)][key], order='xyz', flip_z=True)) / dt
            v_observed = observation_raw[[12+x for x in rot_mappings[key]]]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).sum()
        else:
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='zxy', flip_z=True) - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)][key], order='zxy', flip_z=True)) / dt
            v_observed = observation_raw[[12+x for x in rot_mappings[key]]]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).sum()
    velocity_reward = np.exp(-0.1 * velocity_reward)
        
    mean_offset = np.zeros(3)
    for i in range(1, num_frames):
        mean_offset += (target_motion[i]['root_pos']-target_motion[i]['root_pos'])
    mean_offset /= (num_frames-1)
    
    root_pos_target = target_motion[math.floor(frame_idx)]['root_pos'] + mean_offset*(frame_idx-math.floor(frame_idx))
    if loop_motion:
        root_pos_target += mean_offset*num_loops_passed
    root_pos_target = root_pos_target[[2, 1, 0]]
    
    center_of_mass_reward = np.exp(-10*((root_pos_target - observation_raw[:3])**2).sum())

    reward = 0.65*pose_reward + 0.15*velocity_reward + 0.1*center_of_mass_reward

    info = {
        "pose_reward": pose_reward,
        "velocity_reward": velocity_reward,
        "center_of_mass_reward": center_of_mass_reward
    }

    return reward, info


def punishment(current_step, max_episode_steps, params):  # punishment for early termination
    penalty = params['weight_early_penalty'] * (max_episode_steps - current_step)
    return penalty
