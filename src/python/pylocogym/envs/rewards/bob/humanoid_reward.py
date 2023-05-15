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

def quaternion_to_euler(quat, order='xyz'):
    r = Rotation.from_quat(quat)
    return r.as_euler(order, degrees=False)

def euler_to_quaternion(euler, order='xyz'):
    r = Rotation.from_euler(order, euler, degrees=False)
    return r.as_quat()

joint_mappings = {'lowerback_x': 0,
                  'lHip_1': 1,
                  'rHip_1': 2,
                  'lowerback_y': 3,
                  'lHip_2': 4,
                  'rHip_2': 5,
                  'lowerback_z': 6,
                  'lHip_torsion': 7,
                  'rHip_torsion': 8,
                  'upperback_x': 9,
                  'lKnee': 10,
                  'rKnee': 11,
                  'upperback_y': 12,
                  'lAnkle_1': 13,
                  'rAnkle_1': 14,
                  'upperback_z': 15,
                  'lAnkle_2': 16,
                  'rAnkle_2': 17,
                  'lowerneck_x': 18,
                  'lScapula_y': 19,
                  'rScapula_y': 20,
                  'lToeJoint': 21,
                  'rToeJoint': 22,
                  'lowerneck_y': 23,
                  'lScapula_z': 24,
                  'rScapula_z': 25,
                  'lowerneck_z': 26,
                  'lShoulder_1': 27,
                  'rShoulder_1': 28,
                  'upperneck_x': 29,
                  'lShoulder_2': 30,
                  'rShoulder_2': 31,
                  'upperneck_y': 32,
                  'lShoulder_torsion': 33,
                  'rShoulder_torsion': 34,
                  'upperneck_z': 35,
                  'lElbow_flexion_extension': 36,
                  'rElbow_flexion_extension': 37,
                  'lElbow_torsion': 38,
                  'rElbow_torsion': 39,
                  'lWrist_x': 40,
                  'rWrist_x': 41,
                  'lWrist_z': 42,
                  'rWrist_z': 43
                  }
rot_mappings = {'chest_rot': [joint_mappings['upperback_x'], 
                              joint_mappings['upperback_y'], 
                              joint_mappings['upperback_z']],
                'neck_rot': [joint_mappings['lowerback_x'], 
                             joint_mappings['lowerback_y'], 
                             joint_mappings['lowerback_z']],
                'rhip_rot': [joint_mappings['rHip_1'], 
                             joint_mappings['rHip_2'], 
                             joint_mappings['rHip_torsion']],
                'rknee_rot': [joint_mappings['rKnee']],
                'rankle_rot': [joint_mappings['rAnkle_1'],
                               joint_mappings['lAnkle_2']],
                'rshoulder_rot': [joint_mappings['rShoulder_1'], 
                                  joint_mappings['rShoulder_2'], 
                                  joint_mappings['rShoulder_torsion']],
                'relbow_rot': [joint_mappings['rElbow_flexion_extension']],
                'lhip_rot': [joint_mappings['lHip_1'],
                             joint_mappings['lHip_2'],
                             joint_mappings['lHip_torsion']],
                'lknee_rot': [joint_mappings['rHip_torsion']],
                'lankle_rot': [joint_mappings['lAnkle_1'],
                               joint_mappings['lAnkle_2']],
                'lshoulder_rot': [joint_mappings['lShoulder_1'],
                                  joint_mappings['lShoulder_2'],
                                  joint_mappings['lShoulder_torsion']],
                'lelbow_rot': [joint_mappings['lElbow_flexion_extension']]
}

def get_observed_motion(observation_raw):
    d_quat = {}
    d_euler = {}
    joint_angles_offset = 6
    d_quat['root_pos'] = np.array(observation_raw[0:3])
    d_euler['root_pos'] = np.array(observation_raw[0:3])
    d_quat['root_rot'] = euler_to_quaternion(observation_raw[[3, 4, 5]], order='yxz')
    d_euler['root_rot'] = np.array(observation_raw[[3, 4, 5]])
    d_quat['chest_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['chest_rot']]], order='xyz')
    d_euler['chest_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['chest_rot']]])
    d_quat['neck_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['neck_rot']]], order='xyz')
    d_euler['neck_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['neck_rot']]])
    d_quat['rhip_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['rhip_rot']]], order='xzy')
    d_euler['rhip_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['rhip_rot']]])
    d_quat['rknee_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['rknee_rot']]]
    d_euler['rknee_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['rknee_rot']]]
    d_quat['rankle_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['rankle_rot']] + [0]], order='xzy')
    d_euler['rankle_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['rankle_rot']] + [0]])
    d_quat['rshoulder_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['rshoulder_rot']]], order='xzy')
    d_euler['rshoulder_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['rshoulder_rot']]])
    d_quat['relbow_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['relbow_rot']]]
    d_euler['relbow_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['relbow_rot']]]
    d_quat['lhip_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['lhip_rot']]], order='xzy')
    d_euler['lhip_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['lhip_rot']]])
    d_quat['lknee_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['lknee_rot']]]
    d_euler['lknee_rot'] = observation_raw[[joint_angles_offset+x for x in rot_mappings['lknee_rot']]]
    d_quat['lankle_rot'] = euler_to_quaternion(observation_raw[[joint_angles_offset+x for x in rot_mappings['lankle_rot']] + [0]], order='xzy')
    d_euler['lankle_rot'] = np.array(observation_raw[[joint_angles_offset+x for x in rot_mappings['lankle_rot']] + [0]])
    d_quat['lshoulder_rot'] = euler_to_quaternion([joint_angles_offset+x for x in rot_mappings['lshoulder_rot']], order='xzy')
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
                   joint_angles_default, nominal_base_height, target_motion, loop_motion):
    target_motion = target_motion.copy()
    
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
    
    with open('/local/home/xiychen/Documents/dh-project/n_steps.txt', 'r') as f:
        i = int(f.readline())
    
    observation = ObservationData(observation_raw, num_joints, is_obs_fullstate)
    
    num_frames = len(target_motion.keys())
    duration = target_motion[0]['duration'][0]
    phase = get_phase(dt, i, duration, num_frames, loop_motion)
    frame_idx = phase * num_frames # [0, num_frames]
    
    if loop_motion:
        offset_frame_num = math.floor(phase)*num_frames+math.floor(frame_idx)
        # print(phase, frame_idx, offset_frame_num)
        
        mean_offset = np.zeros(3)
        for i in range(1, num_frames):
            mean_offset += (target_motion[i]['root_pos'])
        # print(num_frames, mean_offset)
        mean_offset /= (num_frames-1)
        # target_motion[math.floor(frame_idx)]['root_pos'] += offset_frame_num*mean_offset
        # target_motion[(math.floor(frame_idx)+1)%num_frames]['root_pos'] += (offset_frame_num+1)*mean_offset
    
    # frame_idx_floor = math.floor(frame_idx)
    observed_motion_quat, observed_motion_euler = get_observed_motion(observation_raw)
    pose_reward = 0
    for key in observed_motion_quat:
        if key == 'root_pos':
            continue
        elif key in['lankle_rot', 'rankle_rot']:
            r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            euler_target = quaternion_to_euler(r_target, order='xzy')
            diff = abs(euler_target - observed_motion_euler[key])**2
            pose_reward += (diff[0]+diff[1])
        elif key in ['lknee_rot', 'rknee_rot', 'lelbow_rot', 'relbow_rot']:
            r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)[0]
            pose_reward += (observed_motion_quat[key][0] - r_target)**2
        else:
            r_target = interpolate(target_motion, key, frame_idx, num_frames, loop_motion)
            pose_reward += get_quaternion_difference(observed_motion_quat[key], r_target)**2
    pose_reward = np.exp(-2 * pose_reward)
    
    velocity_reward = 0
    for key in observed_motion_euler:
        if key == 'root_pos':
            v_target = (target_motion[(math.floor(frame_idx)+1)%num_frames]['root_pos'] - target_motion[math.floor(frame_idx)]['root_pos'])/dt
            v_observed = observation.vel
            # print(mean_offset, offset_frame_num, target_motion[(math.floor(frame_idx)+1)%num_frames]['root_pos'])
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).sum()
        elif key == 'root_rot':
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames]['root_rot'], order='yxz') - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)]['root_rot'], order='yxz')) / dt
            v_observed = observation.ang_vel
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).sum()
        elif key in ['lknee_rot', 'rknee_rot', 'lelbow_rot', 'relbow_rot']:
            v_target = (target_motion[(math.floor(frame_idx)+1)%num_frames][key] - target_motion[math.floor(frame_idx)][key])[0]/dt
            v_observed = observation_raw[[12+x for x in rot_mappings[key]]][0]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).sum()
        elif key in ['lankle_rot', 'rankle_rot']:
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='xzy') - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)][key], order='xzy')) / dt
            v_target = v_target[:2]
            v_observed = observation_raw[[12+x for x in rot_mappings[key]]]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).sum()
        elif key in ['chest_rot', 'neck_rot']:
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='xyz') - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)][key], order='xyz')) / dt
            v_observed = observation_raw[[12+x for x in rot_mappings[key]]]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).sum()
        else:
            v_target = (quaternion_to_euler(target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='xzy') - 
                        quaternion_to_euler(target_motion[math.floor(frame_idx)][key], order='xzy')) / dt
            v_observed = observation_raw[[12+x for x in rot_mappings[key]]]
            v_diff = v_target - v_observed
            velocity_reward += (v_diff**2).sum()
    velocity_reward = np.exp(-0.1 * velocity_reward)
            
    
    
    # print(frame_idx)

    
    

    action_dot, action_ddot = calc_derivatives(action_buffer, dt, num_joints)
    cmd_fwd_vel = params.get("fwd_vel_cmd", 1.0)
    torque = tail(all_torques, num_joints)

    # =============
    # define cost/reward terms here:
    # =============
    # TODO: Implement the rewards here.
    # Hints:
    # - Use function params.get("weight_velocity", 0) to get the value of parameters set in the .conf file.
    
    forward_vel_reward = params.get("weight_velocity", 0) * np.exp(- (abs(observation.local_vel[2] - cmd_fwd_vel)**2) / (2*params.get("sigma_velocity", 0)**2))

    height_reward = params.get("weight_height", 0) * np.exp(- abs(observation.y - nominal_base_height)**2/(2*params.get("sigma_height", 0)**2))

    attitude_reward = params.get("weight_attitude", 0) * np.exp(-(abs(observation.roll)**2+abs(observation.pitch)**2)/(4*params.get("sigma_attitude", 0)**2))

    torque_reward = params.get("weight_torque", 0) * np.exp(-squared_norm(torque)/(2*num_joints*params.get("sigma_torque", 0)**2))
        
    smoothness1_reward = params.get("weight_smoothness1", 0) * np.exp(-squared_norm(action_dot)/(2*num_joints*params.get("sigma_smoothness1", 0)**2))
        
    smoothness2_reward = params.get("weight_smoothness2", 0) * np.exp(-squared_norm(action_ddot)/(2*num_joints*params.get("sigma_smoothness2", 0)**2))

    # joint_reward = params.get("weight_joints", 0) * np.exp(-squared_norm(observation.joint_angles - joint_angles_default)/(2*num_joints*params.get("sigma_joints", 0)**2))

    
    
    
    # =============
    # sum up rewards
    # =============
    smoothness_reward = params.get("weight_smoothness", 0) * (smoothness1_reward + smoothness2_reward)
    reward = 0.65*pose_reward + 0.15*velocity_reward

    info = {
        # "forward_vel_reward": forward_vel_reward,
        # "height_reward": height_reward,
        # "attitude_reward": attitude_reward,

        # "torque_reward": torque_reward,
        # "smoothness1_reward": smoothness1_reward,
        # "smoothness2_reward": smoothness2_reward,
        # "smoothness_reward": smoothness_reward,

        # "joint_reward": joint_reward,
        "pose_reward": pose_reward,
        "velocity_reward": velocity_reward
    }

    return reward, info


def punishment(current_step, max_episode_steps, params):  # punishment for early termination
    penalty = params['weight_early_penalty'] * (max_episode_steps - current_step)
    return penalty
