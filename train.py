
import os
import sys
import time
import torch
from core.unity_underwater_env import UnderwaterNavigation
from core.agent import Agent
from util.args import parse_arguments
from util.zfilter import ZFilter
from util.tools import *

# Append system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Parse arguments
args = parse_arguments()

# Set default data type and device for torch
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cuda", index=args.gpu_index) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

# Initialize environments and agent
environments = [UnderwaterNavigation(args.depth_prediction_model, args.adaptation, args.randomization, i, args.hist_length) for i in range(args.num_threads)]
running_state = ZFilter(environments[0].observation_space_img_depth, environments[0].observation_space_goal, environments[0].observation_space_ray, clip=30)
agent = Agent(environments, running_state=running_state, num_threads=args.num_threads)

# Initialize networks
policy_network, value_network = initialize_networks(args, environments[0].action_space_size, dtype, device)
optimizer_policy = torch.optim.Adam(policy_network.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_network.parameters(), lr=args.learning_rate)

def main_loop():
    for iteration in range(args.max_iter_num):
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render)
        update_start_time = time.time()
        update_networks(batch, policy_network, value_network, optimizer_policy, optimizer_value, args, dtype, device, args.optim_epochs, args.optim_batch_size)
        update_end_time = time.time()
        _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)
        eval_end_time = time.time()
        print_and_save(args, iteration, log, update_start_time, update_end_time, log_eval, eval_end_time, policy_network, value_network, running_state, device)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn") # This is required for the code to work on Windows
    main_loop()