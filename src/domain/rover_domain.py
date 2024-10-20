#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import typing
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class RoverDomain(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):

        self.x_semidim = kwargs.pop("x_semidim", 1)
        self.y_semidim = kwargs.pop("y_semidim", 1)

        self.n_agents = kwargs.pop("n_agents", 5)
        self.n_targets = kwargs.pop("n_targets", 7)
        self.targets_positions = kwargs.pop("targets_positions", [])
        self.targets_values = torch.tensor(
            kwargs.pop("targets_values", []), device=device
        )
        self.agents_positions = kwargs.pop("agents_positions", [])

        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.2)
        self._lidar_range = kwargs.pop("lidar_range", 0.35)
        self._covering_range = kwargs.pop("covering_range", 0.25)

        self.n_lidar_rays_entities = kwargs.pop("n_lidar_rays_entities", 16)
        self.n_lidar_rays_agents = kwargs.pop("n_lidar_rays_agents", 16)

        self._agents_per_target = kwargs.pop("agents_per_target", 2)
        self.targets_respawn = kwargs.pop("targets_respawn", False)
        self.random_spawn = kwargs.pop("random_spawn", False)
        self.use_G = kwargs.pop("use_G", False)

        self.covering_rew_coeff = kwargs.pop("covering_rew_coeff", 1.0)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self._comms_range = self._lidar_range
        self.min_collision_distance = 0.005
        self.agent_radius = 0.05
        self.target_radius = self.agent_radius

        self.viewer_zoom = 1.2
        self.target_color = Color.GREEN
        self.device = device

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            collision_force=500,
            substeps=2,
            drag=0.25,
        )

        # Add agents
        entity_filter_agents: Callable[[Entity], bool] = lambda e: e.name.startswith(
            "agent"
        )
        entity_filter_targets: Callable[[Entity], bool] = lambda e: e.name.startswith(
            "target"
        )
        for i in range(self.n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                shape=Sphere(radius=self.agent_radius),
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=self.n_lidar_rays_entities,
                            max_range=self._lidar_range,
                            entity_filter=entity_filter_targets,
                            render_color=Color.GREEN,
                        )
                    ]
                    + [
                        Lidar(
                            world,
                            n_rays=self.n_lidar_rays_agents,
                            max_range=self._lidar_range,
                            entity_filter=entity_filter_agents,
                            render_color=Color.BLUE,
                        )
                    ]
                ),
            )
            agent.difference_rew = torch.zeros(batch_dim, device=device)
            world.add_agent(agent)

        self._targets = []
        for i in range(self.n_targets):
            target = Landmark(
                name=f"target_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=self.target_radius),
                color=self.target_color,
            )
            world.add_landmark(target)
            self._targets.append(target)

        self.covered_targets = torch.zeros(batch_dim, self.n_targets, device=device)
        self.global_rew = torch.zeros(batch_dim, device=device)

        return world

    def reset_world_at(self, env_index: int = None):
        placable_entities = self._targets[: self.n_targets] + self.world.agents

        if env_index is None:
            self.all_time_covered_targets = torch.full(
                (self.world.batch_dim, self.n_targets),
                False,
                device=self.world.device,
            )
        else:
            self.all_time_covered_targets[env_index] = False

        if self.random_spawn:
            ScenarioUtils.spawn_entities_randomly(
                entities=placable_entities,
                world=self.world,
                env_index=env_index,
                min_dist_between_entities=self._min_dist_between_entities,
                x_bounds=(-self.world.x_semidim, self.world.x_semidim),
                y_bounds=(-self.world.y_semidim, self.world.y_semidim),
            )

            for target in self._targets[self.n_targets :]:
                target.set_pos(self.get_outside_pos(env_index), batch_index=env_index)
        else:
            for idx, agent in enumerate(self.world.agents):
                pos = torch.ones(
                    (self.world.batch_dim, self.world.dim_p), device=self.world.device
                ) * torch.tensor(self.agents_positions[idx], device=self.world.device)
                agent.set_pos(
                    pos,
                    batch_index=env_index,
                )

            for idx, target in enumerate(self._targets):
                pos = torch.ones(
                    (self.world.batch_dim, self.world.dim_p), device=self.world.device
                ) * torch.tensor(self.targets_positions[idx], device=self.world.device)
                target.set_pos(
                    pos,
                    batch_index=env_index,
                )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:
            # Calculate G
            agents_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
            targets_pos = torch.stack([t.state.pos for t in self._targets], dim=1)
            agents_targets_dists = torch.cdist(agents_pos, targets_pos)
            agents_per_target = torch.sum(
                (agents_targets_dists < self._covering_range).type(torch.int),
                dim=1,
            )

            self.covered_targets = agents_per_target >= self._agents_per_target

            self.global_rew = torch.sum(
                self.covered_targets * self.targets_values, dim=1
            )

            # Calculate D
            global_rew_without_me = torch.zeros(
                self.world.batch_dim, device=self.world.device
            )
            for me in self.world.agents:
                global_rew_without_me[:] = 0
                me.difference_rew[:] = 0

                agents_without_me = [
                    agent for agent in self.world.agents if agent != me
                ]

                agents_pos_without_me = torch.stack(
                    [a.state.pos for a in agents_without_me], dim=1
                )

                agents_targets_dists_without_me = torch.cdist(
                    agents_pos_without_me, targets_pos
                )

                agents_per_target_without_me = torch.sum(
                    (agents_targets_dists_without_me < self._covering_range).type(
                        torch.int
                    ),
                    dim=1,
                )

                covered_targets_without_me = (
                    agents_per_target_without_me >= self._agents_per_target
                )

                global_rew_without_me = torch.sum(
                    covered_targets_without_me * self.targets_values, dim=1
                )

                me.difference_rew = self.global_rew - global_rew_without_me

        if is_last:
            self.all_time_covered_targets += self.covered_targets
            for i, target in enumerate(self._targets):
                target.state.pos[self.covered_targets[:, i]] = self.get_outside_pos(
                    None
                )[self.covered_targets[:, i]]

        covering_rew = agent.difference_rew if not self.use_G else self.global_rew

        return covering_rew

    def get_outside_pos(self, env_index):
        return torch.empty(
            (
                (1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p)
            ),
            device=self.world.device,
        ).uniform_(-1000 * self.world.x_semidim, -10 * self.world.x_semidim)

    def agent_reward(self, agent, agents_targets_dists):
        agent_index = self.world.agents.index(agent)

        covering_reward = torch.zeros(self.world.batch_dim, device=self.world.device)

        targets_covered_by_agent = (
            agents_targets_dists[:, agent_index] < self._covering_range
        )

        # dists_to_covered_targets = (targets_covered_by_agent * self.agents_targets_dists[:, agent_index])

        num_covered_targets_covered_by_agent = (
            targets_covered_by_agent * self.covered_targets
        ).sum(dim=-1)

        covering_reward += (
            num_covered_targets_covered_by_agent * self.covering_rew_coeff
        )

        return covering_reward

    def observation(self, agent: Agent):

        lidar_measures = torch.cat(
            (agent.sensors[0].measure(), agent.sensors[1].measure()), dim=-1
        ).unsqueeze(0)

        sectors = 4
        resolution = self.n_lidar_rays_agents // sectors

        # Minpool lidar measures to get a dense lidar measure per sector
        dense_lidar_measures = -F.max_pool1d(
            -lidar_measures, kernel_size=resolution, stride=resolution
        ).squeeze(0)

        dense_lidar_measures = F.tanh(
            torch.abs(dense_lidar_measures - self._lidar_range)
        )

        return dense_lidar_measures

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {
            "global_reward": (self.global_rew),
            "difference_reward": (agent.difference_rew),
            "targets_covered": self.covered_targets.sum(-1),
        }
        return info

    def done(self):
        return self.all_time_covered_targets.all(dim=-1)

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        # Target ranges
        for target in self._targets:
            range_circle = rendering.make_circle(self._covering_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*self.target_color.value)
            geoms.append(range_circle)
        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self._comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, continuous_action, device):

        super().__init__(continuous_action)

        self.device = device
        self.theta_max = 6 * torch.pi
        self.theta_range = torch.arange(
            0, self.theta_max, step=self.theta_max / 600
        ).to(device)

        self.targets = torch.tensor(
            [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], device=device
        )
        self.current_target = 0

    def compute_action(
        self,
        agent_position,
        observation: torch.Tensor,
        u_range: float,
    ) -> torch.Tensor:

        des_pos = torch.zeros((1, 2), device=observation.device).to(observation.device)

        if self.current_target >= len(self.targets):
            self.current_target = len(self.targets) - 1

        des_pos[:, 0] = self.targets[self.current_target, 0]
        des_pos[:, 1] = self.targets[self.current_target, 1]

        action = torch.clamp(
            (des_pos - agent_position),
            min=-u_range,
            max=u_range,
        )

        dist_to_target = torch.cdist(des_pos, agent_position) ** 2

        if dist_to_target[0, 0] < 0.02:
            self.current_target += 1

        return action


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
