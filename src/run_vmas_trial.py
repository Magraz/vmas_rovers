#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import random
import time

import torch

from vmas import make_env
from vmas.simulator.utils import save_video
from vmas.simulator.scenario import BaseScenario
from domain.rover_domain import RoverDomain, HeuristicPolicy


def use_vmas_env(
    name: str = "dummy",
    render: bool = False,
    save_render: bool = False,
    n_envs: int = 4,
    n_steps: int = 400,
    device: str = "cpu",
    scenario: BaseScenario = None,
    continuous_action: bool = True,
    visualize_render: bool = True,
    dict_spaces: bool = False,
    **kwargs,
):
    """Example function to use a vmas environment

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario (BaseScenario): Scenario Class
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        save_render (bool):  Whether to save render of the scenario
        n_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action
        visualize_render (bool, optional): Whether to visualize the render. Defaults to ``True``.
        dict_spaces (bool, optional): Weather to return obs, rewards, and infos as dictionaries with agent names.
            By default, they are lists of len # of agents
        kwargs (dict, optional): Keyword arguments to pass to the scenario

    Returns:

    """
    assert not (save_render and not render), "To save the video you have to render it"

    env = make_env(
        scenario=scenario,
        num_envs=n_envs,
        device=device,
        dict_spaces=dict_spaces,
        wrapper=None,
        seed=None,
        # Environment specific variables
        **kwargs,
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    policy = HeuristicPolicy(continuous_action=continuous_action)
    G = torch.zeros(n_envs)

    for _ in range(n_steps):
        step += 1

        actions = []

        for agent in env.agents:

            action = policy.compute_action(
                env.scenario.observation(agent), agent.action.u_range
            )

            actions.append(action)

        obs, rews, dones, info = env.step(actions)

        G += torch.stack(rews, dim=0)[0]

        if render:
            frame = env.render(
                mode="rgb_array",
                agent_index_focus=None,  # Can give the camera an agent index to focus on
                visualize_when_rgb=visualize_render,
            )
            if save_render:
                frame_list.append(frame)

    print(G)

    total_time = time.time() - init_time

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device} "
        f"for {name} scenario."
    )

    if render and save_render:
        save_video(name, frame_list, fps=10 / env.scenario.world.dt)


if __name__ == "__main__":
    scenario = RoverDomain()
    n_agents = 3
    use_vmas_env(
        name=f"RoverDomain_{n_agents}a",
        scenario=scenario,
        render=True,
        save_render=False,
        continuous_action=True,
        device="cpu",
        # Environment specific
        n_agents=n_agents,
        n_steps=100,
        x_semidim=2,
        y_semidim=2,
    )
