import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import time
import uuid
import random
import os
from utils import *

from typing import List
from gym import spaces
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torchvision.transforms import Compose
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.midas_net import MidasNet_large
from DPT.dpt.midas_net_custom import MidasNet_small
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from nnmodels.dpt_depth import DPTDepth

DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
DIM_GOAL = 3
DIM_ACTION = 2
BITS = 2

class DPT_depth():
    def __init__(self, device, model_type="dpt_large", model_path=
    os.path.abspath("./") + "/DPT/weights/dpt_large-midas-2f21e586.pt",
                 optimize=True):
        self.optimize = optimize
        self.THRESHOLD = torch.tensor(np.finfo("float").eps).to(device)

        # load network
        if model_type == "dpt_large":  # DPT-Large
            self.net_w = self.net_h = 384
            resize_mode = "minimal"
            self.model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == "dpt_hybrid":  # DPT-Hybrid
            self.net_w = self.net_h = 384
            resize_mode = "minimal"
            self.model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "dpt_hybrid_kitti":
            self.net_w = 1216
            self.net_h = 352
            resize_mode = "minimal"

            self.model = DPTDepthModel(
                path=model_path,
                scale=0.00006016,
                shift=0.00579,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )

            self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == "dpt_hybrid_nyu":
            self.net_w = 640
            self.net_h = 480
            resize_mode = "minimal"

            self.model = DPTDepthModel(
                path=model_path,
                scale=0.000305,
                shift=0.1378,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )

            self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == "midas_v21":  # Convolutional model
            self.net_w = self.net_h = 384

            resize_mode = "upper_bound"
            self.model = MidasNet_large(model_path, non_negative=True)
            self.normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        elif model_type == "midas_v21_small":
            self.net_w = self.net_h = 256
            resize_mode = "upper_bound"
            self.model = MidasNet_small(model_path, non_negative=True)

            self.normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        else:
            assert (
                False
            ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

        # select device
        self.device = device

        self.transform = Compose(
            [
                Resize(
                    self.net_w,
                    self.net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                self.normalization,
                PrepareForNet(),
            ]
        )

        self.model.eval()

        if optimize == True and self.device == torch.device("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = self.model.half()

        self.model.to(self.device)

    def run(self, rgb_img):
        img_input = self.transform({"image": rgb_img})["image"]
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            if self.optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(0),
                    size=(DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            depth_min = prediction.min()
            depth_max = prediction.max()
            if depth_max - depth_min > self.THRESHOLD:
                prediction = (prediction - depth_min) / (depth_max - depth_min)
            else:
                prediction = np.zeros(prediction.shape, dtype=prediction.dtype)

            # plt.imshow(np.uint16(prediction * 65536))
            # plt.show()

        # cv2.imwrite("depth.png", ((prediction*65536).astype("uint16")), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return prediction

class PosChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        self.goal_depthfromwater = msg.read_float32_list()

    def goal_depthfromwater_info(self):
        return self.goal_depthfromwater

    def assign_testpos_visibility(self, data: List[float]) -> None:
        msg = OutgoingMessage()
        msg.write_float32_list(data)
        super().queue_message_to_send(msg)

visibility_constant = 1

class Underwater_navigation():
    def __init__(self, depth_prediction_model, adaptation, randomization, rank, history, start_goal_pos=None, training=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._validate_parameters(adaptation, randomization, start_goal_pos, training)
        self._initialize_depth_model(depth_prediction_model)
        self._initialize_parameters(adaptation, randomization, history, training, start_goal_pos)
        self._setup_unity_env(rank)

    def reset(self):
        self.total_episodes += 1
        self.step_count = 0

        # Adjust the visibility of the environment before resetting
        self._adjust_visibility()
        self.env.reset()

        obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
        self._eval_save(obs_goal_depthfromwater)

        # Stepping with zero action to get the first observation
        obs_img_ray, _, done, _ = self.env.step([0, 0])
        obs_predicted_depth = self.dpt.run(obs_img_ray[0] ** 0.45)

        # Get the minimum value of certain indices in the second channel of obs_img_ray
        indices = [1, 3, 5, 33, 35] # Question: what are these indices?
        values = [obs_img_ray[1][i] for i in indices]
        min_value = np.min(values)

        # Multiply the minimum value by 8 and 0.5 to get the final value for obs_ray
        obs_ray_value = min_value * 8 * 0.5
        obs_ray = np.array([obs_ray_value])

        # Retrive the second observation
        obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
        self._eval_save(obs_goal_depthfromwater)

        # Construct the observations of depth images, goal infos, and rays for consecutive 4 frames
        self.obs_predicted_depths = np.array([obs_predicted_depth.tolist()] * self.history)
        self.obs_goals = np.array([obs_goal_depthfromwater[:3].tolist()] * self.history)
        self.obs_rays = np.array([obs_ray.tolist()] * self.history)
        self.obs_actions = np.array([[0, 0]] * self.history)
        self.obs_visibility = np.reshape(self.visibility_para_Gaussian, [1, 1, 1])

        return self.obs_predicted_depths, self.obs_goals, self.obs_rays, self.obs_actions

    def step(self, action):
        self.time_before = time.time()

        # action[0] controls vertical speed
        # action[1] controls rotation speed
        action_ver, action_rot = action
        action_rot *= self.twist_range

        # observations per frame
        obs_img_ray, _, done, _ = self.env.step([action_ver, action_rot])
        obs_predicted_depth = self.dpt.run(obs_img_ray[0] ** 0.45)
        obs_ray, obstacle_distance, obstacle_distance_vertical = self._get_obs(obs_img_ray)
        obs_goal_depthfromwater = self.pos_info.goal_depthfromwater_info()
        """
            compute reward
            obs_goal_depthfromwater[0]: horizontal distance
            obs_goal_depthfromwater[1]: vertical distance
            obs_goal_depthfromwater[2]: angle from robot's orientation to the goal (degree)
            obs_goal_depthfromwater[3]: robot's current y position
            obs_goal_depthfromwater[4]: robot's current x position            
            obs_goal_depthfromwater[5]: robot's current z position            
        """
        obs_goal_depthfromwater = self.pos_info.goal_depthfromwater_info()
        horizontal_distance = obs_goal_depthfromwater[0]
        vertical_distance = obs_goal_depthfromwater[1]
        vertical_distance_abs = np.abs(vertical_distance)
        angle_to_goal = obs_goal_depthfromwater[2]
        angle_to_goal_abs_rad = np.abs(np.deg2rad(angle_to_goal))
        y_pos = obs_goal_depthfromwater[3]
        x_pos = obs_goal_depthfromwater[4]
        z_pos = obs_goal_depthfromwater[5]
        orientation = obs_goal_depthfromwater[6]

        # 1. give a negative reward when robot is too close to nearby obstacles, seafloor or the water surface
        if obstacle_distance < 0.5:
            reward_obstacle = -10
            done = True
            print("Too close to the obstacle!")
            print("Horizontal distance to nearest obstacle:", obstacle_distance)
        elif np.abs(y_pos) < 0.24:
            reward_obstacle = -10
            done = True
            print("Too close to the seafloor!")
            print("Distance to water surface:", np.abs(y_pos))
        elif obstacle_distance_vertical < 0.12:
            reward_obstacle = -10
            done = True
            print("Too close to the vertical obstacle!")
            print("Vertical distance to nearest obstacle:", obstacle_distance_vertical)
        else:
            reward_obstacle = 0

        # 2. give a positive reward if the robot reaches the goal
        goal_distance_threshold = 0.6 if self.training else 0.8
        if horizontal_distance < goal_distance_threshold:
            reward_goal_reached = (10 - 8 * vertical_distance_abs - angle_to_goal_abs_rad)
            done = True
            print("Reached the goal area!")
            self.reach_goal += 1
        else:
            reward_goal_reached = 0

        # 3. give a positive reward if the robot is reaching the goal
        reward_goal_reaching_vertical = np.abs(action_ver) if vertical_distance * action_ver > 0 else -np.abs(action_ver)

        # 4. give negative rewards if the robot too often turns its direction or is near any obstacle
        reward_goal_reaching_horizontal = (-angle_to_goal_abs_rad + np.pi / 3) / 10
        if 0.5 <= obstacle_distance < 1.0:
            reward_goal_reaching_horizontal *= (obstacle_distance - 0.5) / 0.5
            reward_obstacle -= (1 - obstacle_distance) * 2
        reward = (reward_obstacle + reward_goal_reached + reward_goal_reaching_horizontal + reward_goal_reaching_vertical)
        self.step_count += 1
        if self.step_count > 500:
            done = True
            print("Exceeds the max num_step...")

        # construct the observations of depth images, goal infos, and rays for consecutive 4 frames
        obs_predicted_depth = np.reshape(obs_predicted_depth, (1, self.dpt.depth_image_height, self.dpt.depth_image_width))
        self.obs_predicted_depths = np.append(obs_predicted_depth, self.obs_predicted_depths[: (self.history - 1), :, :], axis=0)

        obs_goal = np.reshape(np.array(obs_goal_depthfromwater[0:3]), (1, DIM_GOAL))
        self.obs_goals = np.append(obs_goal, self.obs_goals[:(self.history - 1), :], axis=0)

        obs_ray = np.reshape(np.array(obs_ray), (1, 1))  # single beam sonar and adaptation representation
        self.obs_rays = np.append(obs_ray, self.obs_rays[:(self.history - 1), :], axis=0)

        obs_action = np.reshape(action, (1, DIM_ACTION))
        self.obs_actions = np.append(obs_action, self.obs_actions[:(self.history - 1), :], axis=0)

        self.time_after = time.time()
        self._eval_save(obs_goal_depthfromwater)
        
        print(f'x: {x_pos}, y: {y_pos}, z: {z_pos}, orientation: {orientation}, horizontal distance: {horizontal_distance}, vertical distance: {vertical_distance}, angle to goal: {angle_to_goal}, reward: {reward}, done: {done}')

        return self.obs_predicted_depths, self.obs_goals, self.obs_rays, self.obs_actions, reward, done, 0

    def _validate_parameters(self, adaptation, randomization, start_goal_pos, training):
        if adaptation and not randomization:
            raise Exception("Adaptation should be used with domain randomization during training")
        if not training and start_goal_pos is None:
            raise AssertionError

    def _initialize_depth_model(self, depth_prediction_model):
        model_path = os.path.abspath("./") + "/DPT/weights/"
        if depth_prediction_model == "dpt":
            model_file = "dpt_large-midas-2f21e586.pt"
            model_type = "dpt_large"
        elif depth_prediction_model == "midas":
            model_file = "midas_v21_small-70d6b9c8.pt"
            model_type = "midas_v21_small"
        self.dpt = DPTDepth(self.device, model_type=model_type, model_path=model_path + model_file)

    def _initialize_parameters(self, adaptation, randomization, history, training, start_goal_pos):
        self.adaptation = adaptation
        self.randomization = randomization
        self.history = history
        self.training = training
        self.start_goal_pos = start_goal_pos

        # Initialize additional class variables
        self.twist_range = 30  # degree
        self.vertical_range = 0.1
        self.action_space = spaces.Box(
            np.array([-self.twist_range, -self.vertical_range]).astype(np.float32),
            np.array([self.twist_range, self.vertical_range]).astype(np.float32),
        )

        # Initialize observation space variables
        self.observation_space_img_depth = (self.history, self.dpt.depth_image_height, self.dpt.depth_image_width)
        self.observation_space_goal = (self.history, DIM_GOAL)
        self.observation_space_ray = (self.history, 1)

        # Initialize performance tracking variables
        self.total_steps = 0
        self.total_correct = 0
        self.total_episodes = 0
        self.reach_goal = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_unity_env(self, rank):
        config_channel = EngineConfigurationChannel()
        self.pos_info = PosChannel()
        unity_env = UnityEnvironment(
            os.path.abspath("./") + "/underwater_env/water",
            side_channels=[config_channel, self.pos_info],
            worker_id=rank,
            base_port=5005
        )
        if not self.training:
            visibility = 3 * (13 ** random.uniform(0, 1)) if self.randomization else 3 * (13 ** visibility_constant)
            self.pos_info.assign_testpos_visibility(self.start_goal_pos + [visibility])
        config_channel.set_configuration_parameters(time_scale=10, capture_frame_rate=100)
        self.env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

    def _adjust_visibility(self):
        # Adjust the visibility of the environment
        if self.randomization:
            visibility_para = random.uniform(-1, 1)
            visibility = 3 * (13 ** ((visibility_para + 1) / 2))
            if self.adaptation:
                self.visibility_para_Gaussian = np.clip(np.random.normal(visibility_para, 0.02, 1), -1, 1)
            else:
                self.visibility_para_Gaussian = np.array([0])
        else:
            visibility = 3 * (13 ** visibility_constant)
            self.visibility_para_Gaussian = np.array([0])
        # Assign the visibility to the environment
        if self.training:
            self.pos_info.assign_testpos_visibility([0] * 9 + [visibility])
        else:
            self.pos_info.assign_testpos_visibility(self.start_goal_pos + [visibility])

    def _eval_save(self, obs_goal_depthfromwater):
        if not self.training:
            with open(os.path.join(assets_dir(), "learned_models/test_pos.txt"), "a") as f:
                f.write(f"{obs_goal_depthfromwater[4]} {obs_goal_depthfromwater[5]} {obs_goal_depthfromwater[6]}\n")

    def _get_obs(self, obs_img_ray):
        obs_ray = np.array([
            np.min([
                obs_img_ray[1][1],
                obs_img_ray[1][3],
                obs_img_ray[1][5],
                obs_img_ray[1][33],
                obs_img_ray[1][35]
            ]) * 8 * 0.5
        ])
        obstacle_distance = (
            np.min(
                [
                    obs_img_ray[1][1],
                    obs_img_ray[1][3],
                    obs_img_ray[1][5],
                    obs_img_ray[1][7],
                    obs_img_ray[1][9],
                    obs_img_ray[1][11],
                    obs_img_ray[1][13],
                    obs_img_ray[1][15],
                    obs_img_ray[1][17],
                ]
            ) * 8 * 0.5
        )
        obstacle_distance_vertical = (
            np.min(
                [
                    obs_img_ray[1][81],
                    obs_img_ray[1][79],
                    obs_img_ray[1][77],
                    obs_img_ray[1][75],
                    obs_img_ray[1][73],
                    obs_img_ray[1][71],
                ]
            ) * 8 * 0.5
        )
        return obs_ray,obstacle_distance,obstacle_distance_vertical