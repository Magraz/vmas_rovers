#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import os
import time
import argparse

import torch

from vmas import make_env
from vmas.simulator.utils import save_video
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.environment import Environment

from domain.rover_domain import RoverDomain, HeuristicPolicy
from domain.create_env import create_env


def use_vmas_env(
    name: str,
    env: Environment = None,
    render: bool = False,
    save_render: bool = False,
    n_envs: int = 1,
    n_steps: int = 600,
    device: str = "cpu",
    visualize_render: bool = True,
    **kwargs,
):
    """Example function to use a vmas environment

    Args:
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        save_render (bool):  Whether to save render of the scenario
        n_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action
        visualize_render (bool, optional): Whether to visualize the render. Defaults to ``True``.
        kwargs (dict, optional): Keyword arguments to pass to the scenario

    Returns:

    """
    assert not (save_render and not render), "To save the video you have to render it"

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    policy = HeuristicPolicy(continuous_action=True, device=device)
    G_total = torch.zeros((n_agents, n_envs)).to(device)
    D_total = torch.zeros((n_agents, n_envs)).to(device)
    G_list = []
    D_list = []

    for step in range(n_steps):
        step += 1

        actions = []

        for agent in env.agents:

            action = policy.compute_action(
                agent_position=agent.state.pos,
                observation=env.scenario.observation(agent),
                u_range=agent.action.u_range,
            )

            actions.append(action)

        obs, rews, dones, info = env.step(actions)
        temp = [g[:n_envs] for g in rews]

        G_list.append(torch.stack([g[:n_envs] for g in rews], dim=0)[0])
        D_list.append(torch.stack([g[n_envs : n_envs * 2] for g in rews], dim=0))

        G_total += torch.stack([g[:n_envs] for g in rews], dim=0)
        D_total += torch.stack([g[n_envs : n_envs * 2] for g in rews], dim=0)

        G = torch.stack([g[:n_envs] for g in rews], dim=0)
        D = torch.stack([g[n_envs : n_envs * 2] for g in rews], dim=0)

        if any(tensor.any() for tensor in rews):
            # print("G")
            # print(G)
            # print("D")
            # print(D)

            print("Total G")
            print(G_total)
            print("Total D")
            print(D_total)

        if render:
            frame = env.render(
                mode="rgb_array",
                agent_index_focus=None,  # Can give the camera an agent index to focus on
                visualize_when_rgb=visualize_render,
            )
            if save_render:
                frame_list.append(frame)

    print("G List Agg")
    G_agg = torch.sum(torch.stack(G_list), dim=0)
    print(G_agg)
    print("D List Agg")
    D_agg = torch.sum(torch.stack(D_list), dim=0)
    print(torch.transpose(D_agg, dim0=0, dim1=1))

    # print("G List")
    # print(G_list)
    # print("D List")
    # print(D_list)

    total_time = time.time() - init_time

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device} "
        f"for {name} scenario."
    )

    if render and save_render:
        save_video(name, frame_list, fps=10 / env.scenario.world.dt)


if __name__ == "__main__":
    # Arg parser variables
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpc", default=False, help="use hpc config files", action="store_true"
    )
    parser.add_argument(
        "--load_checkpoint", default=False, help="loads checkpoint", action="store_true"
    )
    parser.add_argument("--poi_type", default="static", help="static/decay", type=str)
    parser.add_argument("--model", default="mlp", help="mlp/gru/cnn/att", type=str)
    parser.add_argument(
        "--experiment_type",
        default="",
        help="standard/fitness_critic/teaming",
        type=str,
    )
    parser.add_argument("--trial_id", default=0, help="Sets trial ID", type=int)

    args = vars(parser.parse_args())

    # Set base_config path
    dir_path = os.path.dirname(os.path.realpath(__file__))

    config_dir = os.path.join(dir_path, "experiments", "yamls", args["experiment_type"])

    # Set configuration file
    yaml_filename = "_".join((args["poi_type"], args["model"])) + ".yaml"
    config_dir = os.path.join(config_dir, yaml_filename)

    n_agents = 3
    n_envs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_vmas_env(
        name=f"RoverDomain_{n_agents}a",
        env=create_env(config_dir=config_dir, n_envs=n_envs, device=device),
        render=False,
        save_render=False,
        continuous_action=True,
        device="cuda",
        n_envs=n_envs,
        # Environment specific
        n_agents=n_agents,
        n_steps=300,
    )
