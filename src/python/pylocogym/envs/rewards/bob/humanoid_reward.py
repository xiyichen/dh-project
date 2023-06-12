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

def compute_reward(observation_raw, dt, num_joints, params, feet_status, all_torques, action_buffer, is_obs_fullstate,
                   joint_angles_default, nominal_base_height, target_motion, loop_motion, frame_idx, num_loops_passed, init_phase, phase, q_init, target_speed=0, target_heading=0):
    
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
    # frame_idx = phase * num_frames # [0, num_frames]
    
    observed_motion_quat, observed_motion_euler = get_observed_motion(observation_raw)
    # print(phase, init_phase, frame_idx, q_init)
    

    offset = (target_motion[num_frames-1]['root_pos']-target_motion[0]['root_pos'])   
    mean_offset = np.zeros(3)
    for i in range(1, num_frames):
        mean_offset += (target_motion[i]['root_pos']-target_motion[i-1]['root_pos'])
    mean_offset /= (num_frames-1)

    pose_reward = 0
    for key in observed_motion_quat:
        if key == 'root_pos':
            continue
        elif key in['lankle_rot', 'rankle_rot']:
            r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            euler_target = quaternion_to_euler(r_target, order='yxz', flip_z=False)
            diff = abs(euler_target - observed_motion_euler[key])**2
            pose_reward += ((diff[0]+diff[1])/2)
            # print(key, euler_target, observed_motion_euler[key])
            '''
            print(r_target)
            print(euler_target)
            print(observed_motion_euler[key])
            print(diff)
            '''
        elif key in ['lknee_rot', 'rknee_rot']:
            r_target = -interpolate(target_motion, key, frame_idx, num_frames, loop_motion)[0]
            pose_reward += (observed_motion_quat[key][0] - r_target)**2
            # print(key, r_target, observed_motion_euler[key])
            '''
            print(r_target)
            print(observed_motion_quat[key])
            print((observed_motion_quat[key][0] - r_target)**2)
            '''
        elif key in ['lelbow_rot', 'relbow_rot']:
            r_target = -interpolate(target_motion, key, frame_idx, num_frames, loop_motion)[0]
            pose_reward += (observed_motion_quat[key][0] - r_target)**2
        elif key in ['chest_rot', 'neck_rot']:
            # r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            # euler_target = quaternion_to_euler(r_target, order='zyx', flip_z=True)
            # pose_reward += ((observed_motion_euler[key] - euler_target)**2).mean()
            r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            pose_reward += quat_to_axis_angle(get_quaternion_difference(observed_motion_quat[key], r_target))**2
        elif key == 'root_rot':
            # r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            # euler_target = quaternion_to_euler(r_target, order='yzx', flip_z=True)
            # pose_reward += ((observed_motion_euler[key] - euler_target)**2).mean()
            r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            pose_reward += quat_to_axis_angle(get_quaternion_difference(observed_motion_quat[key], r_target))**2
        elif key in ['lhip_rot', 'rhip_rot']:
            r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            pose_reward += quat_to_axis_angle(get_quaternion_difference(observed_motion_quat[key], r_target))**2
        else:
            # r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            # euler_target = quaternion_to_euler(r_target, order='zxy', flip_z=True)
            # pose_reward += ((observed_motion_euler[key] - euler_target)**2).mean()
            r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            pose_reward += quat_to_axis_angle(get_quaternion_difference(observed_motion_quat[key], r_target))**2
            
        # else:
        #     r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
        #     pose_reward += quat_to_axis_angle(get_quaternion_difference(observed_motion_quat[key], r_target))**2
        #     # print(key, r_target, observed_motion_quat[key])
        #     '''print(get_quaternion_difference(observed_motion_quat[key], r_target))
        #     print(observed_motion_quat[key])
        #     print(r_target)'''
            
    # exit()
    pose_reward = np.exp(-2 * pose_reward)
    
    velocity_reward = 0
    t=target_motion[0]['duration'][0]
    for key in observed_motion_euler:
        if key == 'root_pos':
            if (math.floor(frame_idx)+1)%num_frames == 0:
                v_target =mean_offset/t
            else:
                v_target = (target_motion[(math.floor(frame_idx)+1)%num_frames]['root_pos'] - target_motion[math.floor(frame_idx)]['root_pos'])/t # zyx
            v_observed = observation.vel # xyz
            v_observed = v_observed[[2,1,0]]*[-1,1,1]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).mean()
        elif key == 'root_rot':
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames]['root_rot'], order='yzx', flip_z=True) - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)]['root_rot'], order='yzx', flip_z=True)) / t
            v_observed = observation.ang_vel
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).mean()
        elif key in ['lknee_rot', 'rknee_rot', 'lelbow_rot', 'relbow_rot']:
            v_target = (target_motion[math.floor(frame_idx)][key] - target_motion[(math.floor(frame_idx)+1)%num_frames][key])[0]/t # knee and elbow rotations are flipped when mapping from deepmimic to bob
            v_observed = observation.joint_vel[rot_mappings[key]][0]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).mean()
        elif key in ['lankle_rot', 'rankle_rot']:
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='yxz', flip_z=False) - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)][key], order='yxz', flip_z=False)) / t
            v_target = v_target[:2]
            v_observed = observation.joint_vel[rot_mappings[key]]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).mean()
        elif key in ['chest_rot']:
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='zyx', flip_z=True) - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)][key], order='zyx', flip_z=True)) / t
            v_observed = observation.joint_vel[rot_mappings[key]]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).mean()
        elif key in ['neck_rot']:
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='zyx', flip_z=False) - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)][key], order='zyx', flip_z=False)) / t
            v_observed = observation.joint_vel[rot_mappings[key]]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).mean()
        else:
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='zxy', flip_z=True, flip_y=True) - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)][key], order='zxy', flip_z=True, flip_y=True)) / t
            v_observed = observation.joint_vel[rot_mappings[key]]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).mean()
        '''print(key)
        print(v_observed)
        print(v_target)'''
    velocity_reward = np.exp(-0.1 * velocity_reward)

    
    
    root_pos_target = interpolate(target_motion, 'root_pos', frame_idx, num_frames, loop_motion)
    
    if loop_motion:
        root_pos_target += (mean_offset+offset)*num_loops_passed
    root_pos_target = root_pos_target[[2, 1, 0]]*[-1,1,1]
    
    center_of_mass_reward = np.exp(-10*((root_pos_target - observation_raw[:3])**2).sum())
    # center_of_mass_reward = np.exp(-10*((root_pos_target[0] - observation_raw[0])**2+(root_pos_target[2] - observation_raw[2])**2))
    
    height_reward = params.get("weight_height", 0) * np.exp(- abs(observation.y - nominal_base_height)**2/(2*params.get("sigma_height", 0)**2))
    
    if target_speed!=0:
        v_observed = np.array(observation.vel)
        v_target= np.array([math.cos(target_heading),0,math.sin(target_heading)])
        v_proj= np.dot(v_observed,v_target)

        vel_error=max(target_speed- v_proj,0)

        vel_target_reward=math.exp(-vel_error**2)

    reward = 0.65*pose_reward \
             + 0.1*center_of_mass_reward \
             + 0.15*velocity_reward \
            #  + height_reward
    
    
    # reward = pose_reward 
             

    info = {
        "pose_reward": pose_reward,
        "center_of_mass_reward": center_of_mass_reward,
        "velocity_reward": velocity_reward,
        # "height_reward": height_reward
    }

    return reward, info


def punishment(current_step, max_episode_steps, params):  # punishment for early termination
    penalty = params['weight_early_penalty'] * (max_episode_steps - current_step)
    return penalty