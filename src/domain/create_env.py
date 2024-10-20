import os
from pathlib import Path
import yaml

from vmas import make_env
from vmas.simulator.environment import Environment

from domain.rover_domain import RoverDomain


def create_env(config_dir, n_envs: int, device: str) -> Environment:
    config_dir = Path(os.path.expanduser(config_dir))

    with open(str(config_dir), "r") as file:
        config = yaml.safe_load(file)

    # Environment data
    map_size = config["env"]["map_size"]

    # Agent data
    n_agents = len(config["env"]["rovers"])
    agents_positions = [poi["position"]["fixed"] for poi in config["env"]["rovers"]]
    lidar_rays = [rover["lidar"]["rays"] for rover in config["env"]["rovers"]]
    lidar_range = [rover["lidar"]["range"] for rover in config["env"]["rovers"]]

    # POIs data
    n_pois = len(config["env"]["pois"])
    poi_positions = [poi["position"]["fixed"] for poi in config["env"]["pois"]]
    poi_values = [poi["value"] for poi in config["env"]["pois"]]
    coupling = [poi["coupling"] for poi in config["env"]["pois"]]
    obs_radius = [poi["observation_radius"] for poi in config["env"]["pois"]]

    # Set up the enviornment
    env = make_env(
        scenario=RoverDomain(),
        num_envs=n_envs,
        device=device,
        seed=None,
        # Environment specific variables
        n_agents=n_agents,
        n_targets=n_pois,
        agents_positions=agents_positions,
        targets_positions=poi_positions,
        targets_values=poi_values,
        x_semidim=map_size[0],
        y_semidim=map_size[1],
        agents_per_target=coupling[0],
        covering_range=obs_radius[0],
        n_lidar_rays_entities=lidar_rays[0],
        n_lidar_rays_agents=lidar_rays[0],
        lidar_range=lidar_range[0],
    )

    return env
