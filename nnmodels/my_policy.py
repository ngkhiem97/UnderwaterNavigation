import torch.nn as nn
import torch
import numpy as np
from util.mathpy import *
import torchvision
import cv2
import math


class MyPolicy(nn.Module):
    def __init__(self, pretrained):
        super(MyPolicy, self).__init__()
        self.is_disc_action = False
        """ layers for inputs of depth_images """
        self.conv1 = pretrained.conv1
        self.conv2 = pretrained.conv2
        self.conv3 = pretrained.conv3
        self.fc_img = pretrained.fc_img
        """ layers for inputs of goals and rays """
        self.fc_goal = pretrained.fc_goal
        self.fc_ray = pretrained.fc_ray
        self.fc_action = pretrained.fc_action
        """ layers for inputs concatenated information """
        self.img_goal_ray1 = pretrained.img_goal_ray1
        self.img_goal_ray2 = pretrained.img_goal_ray2
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.action_log_std = pretrained.action_log_std

    def forward(self, depth_img, goal, ray, hist_action):
        depth_img = self.relu(self.conv1(depth_img))
        depth_img = self.relu(self.conv2(depth_img))
        depth_img = self.relu(self.conv3(depth_img))
        depth_img = depth_img.view(depth_img.size(0), -1)
        depth_img = self.relu(self.fc_img(depth_img))
        goal = goal.view(goal.size(0), -1)
        goal = self.relu(self.fc_goal(goal))
        ray = ray.view(ray.size(0), -1)
        ray = self.relu(self.fc_ray(ray))
        hist_action = hist_action.view(hist_action.size(0), -1)
        hist_action = self.relu(self.fc_action(hist_action))
        img_goal_ray_aciton = torch.cat((depth_img, goal, ray, hist_action), 1)
        img_goal_ray_aciton = self.relu(self.img_goal_ray1(img_goal_ray_aciton))
        action_mean = self.tanh(self.img_goal_ray2(img_goal_ray_aciton))
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std

    def select_action(self, depth_img, goal, ray, hist_action):
        action_mean, _, action_std = self.forward(depth_img, goal, ray, hist_action)
        action = torch.clamp(torch.normal(action_mean, action_std), -1, 1)
        return action

    def get_log_prob(self, depth_img, goal, ray, hist_action, actions):
        action_mean, action_log_std, action_std = self.forward(depth_img, goal, ray, hist_action)
        return normal_log_density(actions, action_mean, action_log_std, action_std)
