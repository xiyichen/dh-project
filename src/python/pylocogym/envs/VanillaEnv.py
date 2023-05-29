import sys
import numpy as np
import importlib.util
from importlib.machinery import SourceFileLoader

from .PylocoEnv import PylocoEnv
from ..cmake_variables import PYLOCO_LIB_PATH
import os
from scipy.spatial.transform import Rotation
import json
import numpy as np
import math
from pylocogym.envs.rewards.utils.mappings import *
from pylocogym.envs.rewards.utils.utils import *

# importing pyloco
spec = importlib.util.spec_from_file_location("pyloco", PYLOCO_LIB_PATH)
pyloco = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = pyloco
spec.loader.exec_module(pyloco)

def load_target_motion(motion_clip, data_path):
    fn = os.path.join(data_path, 'humanoid3d_{}.txt'.format(motion_clip))
    dict = json.load(open(fn))
    frames = dict['Frames']
    d = {}
    for idx, frame in enumerate(frames):
        if idx not in d:
            d[idx] = {}
        d[idx]['duration'] = np.array(frame[:1])
        d[idx]['root_pos'] = np.array(frame[1:4])
        # d[idx]['root_rot'] = quat_to_euler(frame[4:8])
        d[idx]['root_rot'] = np.array(frame[4:8])
        # d[idx]['chest_rot'] = quat_to_euler(frame[8:12])
        # d[idx]['neck_rot'] = quat_to_euler(frame[12:16])
        # d[idx]['rhip_rot'] = quat_to_euler(frame[16:20])
        d[idx]['chest_rot'] = np.array(frame[8:12])
        d[idx]['neck_rot'] = np.array(frame[12:16])
        d[idx]['rhip_rot'] = np.array(frame[16:20])
        d[idx]['rknee_rot'] = np.array(np.array(frame[20:21]))
        # d[idx]['rankle_rot'] = quat_to_euler(frame[21:25])
        # d[idx]['rshoulder_rot'] = quat_to_euler(frame[25:29])
        d[idx]['rankle_rot'] = np.array(frame[21:25])
        d[idx]['rshoulder_rot'] = np.array(frame[25:29])
        d[idx]['relbow_rot'] = np.array(frame[29:30])
        # d[idx]['lhip_rot'] = quat_to_euler(frame[30:34])
        d[idx]['lhip_rot'] = np.array(frame[30:34])
        d[idx]['lknee_rot'] = np.array(frame[34:35])
        # d[idx]['lankle_rot'] = quat_to_euler(frame[35:39])
        # d[idx]['lshoulder_rot'] = quat_to_euler(frame[39:43])
        d[idx]['lankle_rot'] = np.array(frame[35:39])
        d[idx]['lshoulder_rot'] = np.array(frame[39:43])
        d[idx]['lelbow_rot'] = np.array(frame[43:44])
    # print(d)
    return d


class VanillaEnv(PylocoEnv):

    def __init__(self, max_episode_steps, env_params, reward_params, motion_clip='run'):
        sim_dt = 1.0 / env_params['simulation_rate']
        con_dt = 1.0 / env_params['control_rate']

        if env_params['robot_model'] == "Dog":
            robot_id = 0
        elif env_params['robot_model'] == "Go1":
            robot_id = 1
        elif env_params['robot_model'] == "Bob":
            robot_id = 2

        loadVisuals = False
        super().__init__(pyloco.VanillaSimulator(sim_dt, con_dt, robot_id, loadVisuals), env_params, max_episode_steps)

        self._sim.lock_selected_joints = env_params.get('lock_selected_joints', False)
        self.enable_box_throwing = env_params.get('enable_box_throwing', False)
        self.box_throwing_interval = 100
        self.box_throwing_strength = 2
        self.box_throwing_counter = 0
        self.target_motion = load_target_motion(motion_clip, os.path.join(os.getcwd(), 'data/robots/motions/'))
        self.loop_motion = not ('getup' in motion_clip or 'kick' in motion_clip or 'punch' in motion_clip)
        # self.loop_motion = False
        if "reward_file_path" in reward_params.keys():
            reward_file_path = reward_params["reward_file_path"]

            self.reward_utils = SourceFileLoader('reward_utils', reward_file_path).load_module()
        else:
            raise Exception("Reward file not specified. Please specify via --rewardFile.")

        self.cnt_timestep_size = self._sim.control_timestep_size  # this should be equal to con_dt
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        self.reward_params = reward_params
        self.sum_episode_reward_terms = {}
        self.action_buffer = np.zeros(self.num_joints * 3)  # history of actions [current, previous, past previous]

        self.rng = np.random.default_rng(
            env_params.get("seed", 1))  # create a local random number generator with seed

    def reset(self, seed=None, return_info=False, options=None):
        # super().reset(seed=seed)  # We need this line to seed self.np_random
        prev_episode_length_ratio = self.current_step/self.max_episode_steps
        # print(self.current_step, self.max_episode_steps, prev_episode_length_ratio)
        self.current_step = 0
        self.box_throwing_counter = 0
        
        self._sim.reset()
        self.init_phase=np.random.uniform(0,1)
        observation = self.get_obs(self.init_phase)
        q_init = observation[:50]
        qdot_init = observation[50:-1] 
        
        t = self.target_motion[0]['duration'][0]
        num_frames = len(self.target_motion.keys())
        loop_motion=False
        target_motion=self.target_motion
        frame_idx = self.init_phase * (num_frames-1)
        offset = (target_motion[num_frames-1]['root_pos']-target_motion[0]['root_pos'])   
        mean_offset = np.zeros(3)
        for i in range(1, num_frames):
            mean_offset += (target_motion[i]['root_pos']-target_motion[i-1]['root_pos'])
        mean_offset /= (num_frames-1)
        
        q_init[0] = interpolate(target_motion,'root_pos' , frame_idx, num_frames, loop_motion)[2]
        q_init[1] = interpolate(target_motion, 'root_pos', frame_idx, num_frames, loop_motion)[1]
        q_init[2] = interpolate(target_motion, 'root_pos', frame_idx, num_frames, loop_motion)[0]
        root_euler = quaternion_to_euler(interpolate(target_motion, 'root_rot', frame_idx, num_frames, loop_motion), order='yzx', flip_z=True)
        q_init[3:6] = root_euler
        chest_euler = quaternion_to_euler(interpolate(target_motion, 'chest_rot', frame_idx, num_frames, loop_motion), order='zyx', flip_z=True)
        q_init[[6+x for x in rot_mappings['chest_rot']]] = chest_euler
        neck_euler = quaternion_to_euler(interpolate(target_motion,'neck_rot' , frame_idx, num_frames, loop_motion), order='zyx', flip_z=True)
        q_init[[6+x for x in rot_mappings['neck_rot']]] = neck_euler
        rhip_euler = quaternion_to_euler(interpolate(target_motion,'rhip_rot' , frame_idx, num_frames, loop_motion), order='zxy', flip_z=True)
        q_init[[6+x for x in rot_mappings['rhip_rot']]] = rhip_euler
        q_init[[6+x for x in rot_mappings['rknee_rot']]] = -interpolate(target_motion,'rknee_rot' , frame_idx, num_frames, loop_motion)
        q_init[[6+x for x in rot_mappings['relbow_rot']]] = -interpolate(target_motion, 'relbow_rot', frame_idx, num_frames, loop_motion)
        lhip_euler = quaternion_to_euler(interpolate(target_motion,'lhip_rot' , frame_idx, num_frames, loop_motion), order='zxy', flip_z=True)
        q_init[[6+x for x in rot_mappings['lhip_rot']]] = lhip_euler
        q_init[[6+x for x in rot_mappings['lknee_rot']]] = -interpolate(target_motion,'lknee_rot' , frame_idx, num_frames, loop_motion)
        q_init[[6+x for x in rot_mappings['lelbow_rot']]] = -interpolate(target_motion,'lelbow_rot' , frame_idx, num_frames, loop_motion)
        q_init[[6+x for x in rot_mappings['lankle_rot']]] = quaternion_to_euler(interpolate(target_motion,'lankle_rot' , frame_idx, num_frames, loop_motion), order='yxz', flip_z=False)[:2]
        q_init[[6+x for x in rot_mappings['rankle_rot']]] = quaternion_to_euler(interpolate(target_motion,'rankle_rot' , frame_idx, num_frames, loop_motion), order='yxz', flip_z=False)[:2]
        rshoulder_euler = quaternion_to_euler(interpolate(target_motion,'rshoulder_rot' , frame_idx, num_frames, loop_motion), order='zxy', flip_z=True)
        q_init[[6+x for x in rot_mappings['rshoulder_rot']]] = rshoulder_euler
        lshoulder_euler = quaternion_to_euler(interpolate(target_motion,'lshoulder_rot' , frame_idx, num_frames, loop_motion), order='zxy', flip_z=True)
        q_init[[6+x for x in rot_mappings['lshoulder_rot']]] = lshoulder_euler
        
        # v_target = (self.target_motion[(math.floor(frame_idx)+1)%num_frames]['root_pos'] - self.target_motion[math.floor(frame_idx)]['root_pos'])/t
        if frame_idx < num_frames - 1:
            v_target = (self.target_motion[(math.floor(frame_idx)+1)]['root_pos'] - self.target_motion[math.floor(frame_idx)]['root_pos'])/t
        else:
            v_target = (self.target_motion[0]['root_pos'] + offset + mean_offset - self.target_motion[math.floor(frame_idx)]['root_pos'])/t
        v_target = v_target[[2,1,0]]
        qdot_init[:3] = v_target
        qdot_init[3:6] = (quaternion_to_euler(self.target_motion[(math.floor(frame_idx)+1)%num_frames]['root_rot'], order='yzx', flip_z=True) - 
                          quaternion_to_euler(self.target_motion[math.floor(frame_idx)]['root_rot'], order='yzx', flip_z=True)) / t
        for key in ['lknee_rot', 'rknee_rot', 'lelbow_rot', 'relbow_rot']:
            qdot_init[[6+x for x in rot_mappings[key]]] = (self.target_motion[math.floor(frame_idx)][key] - self.target_motion[(math.floor(frame_idx)+1)%num_frames][key])[0]/t
        for key in ['lankle_rot', 'rankle_rot']:
            v_target = (quaternion_to_euler(self.target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='yxz', flip_z=False) - 
                        quaternion_to_euler(self.target_motion[math.floor(frame_idx)][key], order='yxz', flip_z=False)) / t
            v_target = v_target[:2]
            qdot_init[[6+x for x in rot_mappings[key]]] = v_target
        for key in ['chest_rot', 'neck_rot']:
            qdot_init[[6+x for x in rot_mappings[key]]] = (quaternion_to_euler(self.target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='zyx', flip_z=True) - 
                                                           quaternion_to_euler(self.target_motion[math.floor(frame_idx)][key], order='zyx', flip_z=True)) / t
        for key in ['lshoulder_rot', 'rshoulder_rot', 'lhip_rot', 'rhip_rot']:
            qdot_init[[6+x for x in rot_mappings[key]]] = (quaternion_to_euler(self.target_motion[(math.floor(frame_idx)+1)%num_frames][key], order='zxy', flip_z=True) - 
                                                           quaternion_to_euler(self.target_motion[math.floor(frame_idx)][key], order='zxy', flip_z=True)) / t
        qdot_init *= prev_episode_length_ratio    
        
        self.q_init = q_init
        self._sim.reset(q_init, qdot_init)
        self.sum_episode_reward_terms = {}
        self.action_buffer = np.concatenate(
            (self.joint_angle_default, self.joint_angle_default, self.joint_angle_default), axis=None)

        info = {"msg": "===Episode Reset Done!===\n"}
        return (observation, info) if return_info else observation
    
    def get_phase(self):
        elapsed_time = self.cnt_timestep_size*self.current_step
        num_loops_passed = elapsed_time / (self.target_motion[0]['duration'][0] * len(self.target_motion.keys()))+self.init_phase
                
        # e.g.: 25.6 -> 0.6
        phase = num_loops_passed - math.floor(num_loops_passed)
        num_loops_passed = math.floor(num_loops_passed)
        
        if num_loops_passed >= 1 and not self.loop_motion:
            num_loops_passed = 0
            phase = 1
        
        return phase, num_loops_passed

    def step(self, action):
        if len(action) == 26:
            action_full = np.zeros(44) #16, 44
            action_full[0] = 0
            action_full[1:3] = action[0:2]
            action_full[3] = 0
            action_full[4:6] = action[2:4]
            action_full[6] = 0
            action_full[7:19] = action[4:16]
            action_full[19] = 0
            action_full[20] = 0
            action_full[21] = 0
            action_full[22] = 0
            action_full[23] = action[16]
            action_full[24] = 0
            action_full[25] = 0
            action_full[26:29] = action[17:20]
            action_full[29] = 0
            action_full[30:32] = action[20:22]
            action_full[32] = 0
            action_full[33:35] = action[22:24]
            action_full[35] = 0
            action_full[36:38] = action[24:26]
            action_full[38] = 0
            action_full[39] = 0
            action_full[40] = 0
            action_full[41] = 0
            action_full[42] = 0
            action_full[43] = 0
            action = action_full
            
        # throw box if needed
        if self.enable_box_throwing and self.current_step % self.box_throwing_interval == 0:
            random_start_pos = (self.rng.random(3) * 2 - np.ones(3)) * 2  # 3 random numbers btw -2 and 2
            self._sim.throw_box(self.box_throwing_counter % 3, self.box_throwing_strength, random_start_pos)
            self.box_throwing_counter += 1

        # run simulation
        action_applied = self.scale_action(action)
        self._sim.step(action_applied)
        
        # update variables
        self.current_step += 1
        
        phase, num_loops_passed = self.get_phase()
        num_frames = len(self.target_motion.keys())
        frame_idx = phase * (num_frames-1)
        
        observation = self.get_obs(phase)
        # print(observation[[6, 9, 12, 25, 26, 27, 28, 30, 31, 35, 38, 41, 44, 45, 46, 47, 48, 49]])
        # exit()
        self.action_buffer = np.roll(self.action_buffer, self.num_joints)  # moving action buffer
        self.action_buffer[0:self.num_joints] = action_applied

        # compute reward
        reward, reward_info = self.reward_utils.compute_reward(observation, self.cnt_timestep_size, self.num_joints,
                                                               self.reward_params, self.get_feet_status(),
                                                               self._sim.get_all_motor_torques(), self.action_buffer,
                                                               self.is_obs_fullstate, self.joint_angle_default,
                                                               self._sim.nominal_base_height, self.target_motion, self.loop_motion, frame_idx, num_loops_passed, self.init_phase, phase, self.q_init)
        
        self.sum_episode_reward_terms = {key: self.sum_episode_reward_terms.get(key, 0) + reward_info.get(key, 0) for
                                         key in reward_info.keys()}

        # check if episode is done
        # terminate if phase == 1 and loop_motion == False
        terminated, truncated, term_info = self.is_done(observation, reward_info, phase, self.loop_motion)
        done = terminated | truncated

        # punishment for early termination
        if terminated:
            reward -= self.reward_utils.punishment(self.current_step, self.max_episode_steps, self.reward_params)

        info = {
            "is_success": truncated,
            "termination_info": term_info,
            "current_step": self.current_step,
            "action_applied": action_applied,
            "reward_info": reward_info,
            "TimeLimit.truncated": truncated,
            "msg": "=== 1 Episode Taken ===\n"
        }

        if done:
            mean_episode_reward_terms = {key: self.sum_episode_reward_terms.get(key, 0) / self.current_step for key in
                                         reward_info.keys()}
            info["mean_episode_reward_terms"] = mean_episode_reward_terms

        return observation, reward, done, info

    def filter_actions(self, action_new, action_old, max_joint_vel):
        # with this filter we have much fewer cases that joints cross their limits, but still not zero
        threshold = max_joint_vel * np.ones(self.num_joints)  # max angular joint velocity
        diff = action_new - action_old
        action_filtered = action_old + np.sign(diff) * np.minimum(np.abs(diff), threshold)
        return action_filtered