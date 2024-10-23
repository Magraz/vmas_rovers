#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union

import torch

import vmas.simulator.core
from vmas.simulator.sensors import Sensor

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class SectorDensity(Sensor):
    def __init__(
        self,
        world: vmas.simulator.core.World,
        angle_start: float = 0.0,
        angle_end: float = 2 * torch.pi,
        n_rays: int = 4,
        max_range: float = 1.0,
        sectors: int = 4,
    ):
        super().__init__(world)
        if (angle_start - angle_end) % (torch.pi * 2) < 1e-5:
            angles = torch.linspace(
                angle_start, angle_end, n_rays + 1, device=self._world.device
            )[:n_rays]
        else:
            angles = torch.linspace(
                angle_start, angle_end, n_rays, device=self._world.device
            )

        self._angles = angles
        self._max_range = max_range
        self._last_measurement = None
        self._sectors = sectors

    def to(self, device: torch.device):
        self._angles = self._angles.to(device)

    def measure(self):

        measurement = torch.zeros((8)).to(self._world.device)

        all_other_agents = [
            agent for agent in self._world._agents if agent != self.agent
        ]

        dists_2_targets = torch.empty((0, self._world.batch_dim)).to(self._world.device)
        dists_2_t_angles = torch.empty((0, self._world.batch_dim)).to(
            self._world.device
        )
        dists_2_agents = torch.empty((0, self._world.batch_dim)).to(self._world.device)
        dists_2_a_angles = torch.empty((0, self._world.batch_dim)).to(
            self._world.device
        )

        for a in all_other_agents:
            dists_2_agents = torch.cat(
                (
                    dists_2_agents,
                    torch.pairwise_distance(
                        self.agent.state.pos, a.state.pos
                    ).unsqueeze(0),
                ),
                dim=0,
            )
            dif = self.agent.state.pos - a.state.pos
            dif_complex = torch.complex(dif[:, 0], dif[:, 1])
            dists_2_a_angles = torch.cat(
                (
                    dists_2_a_angles,
                    torch.remainder(
                        torch.angle(dif_complex).unsqueeze(0), 2 * torch.pi
                    ),
                ),
                dim=0,
            )

        for l in self._world._landmarks:
            dists_2_targets = torch.cat(
                (
                    dists_2_targets,
                    torch.pairwise_distance(
                        self.agent.state.pos, l.state.pos
                    ).unsqueeze(0),
                ),
                dim=0,
            )

            dif = self.agent.state.pos - l.state.pos
            dif_complex = torch.complex(dif[:, 0], dif[:, 1])

            dists_2_t_angles = torch.cat(
                (
                    dists_2_t_angles,
                    torch.remainder(
                        torch.angle(dif_complex).unsqueeze(0), 2 * torch.pi
                    ),
                ),
                dim=0,
            )

        for sector in range(self._sectors):
            # If we are in last sector
            if sector + 1 == self._sectors:
                targets_in_sector = (self._angles[sector] < dists_2_t_angles) & (
                    dists_2_t_angles < (2 * torch.pi)
                )
                agents_in_sector = (self._angles[sector] < dists_2_a_angles) & (
                    dists_2_a_angles < (2 * torch.pi)
                )
            else:
                targets_in_sector = (self._angles[sector] < dists_2_t_angles) & (
                    dists_2_t_angles < self._angles[sector + 1]
                )
                agents_in_sector = (self._angles[sector] < dists_2_a_angles) & (
                    dists_2_a_angles < self._angles[sector + 1]
                )

        self._last_measurement = measurement
        return measurement

    def render(self, env_index: int = 0) -> "List[Geom]":
        return []
