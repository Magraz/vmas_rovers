import os
from pathlib import Path
import yaml

from vmas import make_env
from vmas.simulator.environment import Environment

from domain.rover_domain import RoverDomain


def create_env(batch_dir, n_envs: int, device: str, **kwargs) -> Environment:

    env_file = os.path.join(batch_dir, "_env.yaml")

    with open(str(env_file), "r") as file:
        env_config = yaml.safe_load(file)

    # Environment data
    map_size = env_config["env"]["map_size"]

    # Agent data
    n_agents = len(env_config["env"]["rovers"])
    agents_positions = [poi["position"]["fixed"] for poi in env_config["env"]["rovers"]]
    lidar_range = [rover["observation_radius"] for rover in env_config["env"]["rovers"]]

    # POIs data
    n_pois = len(env_config["env"]["pois"])
    poi_positions = [poi["position"]["fixed"] for poi in env_config["env"]["pois"]]
    poi_values = [poi["value"] for poi in env_config["env"]["pois"]]
    poi_types = [poi["type"] for poi in env_config["env"]["pois"]]
    poi_orders = [poi["order"] for poi in env_config["env"]["pois"]]
    poi_colors = [poi["color"] for poi in env_config["env"]["pois"]]
    coupling = [poi["coupling"] for poi in env_config["env"]["pois"]]
    obs_radius = [poi["observation_radius"] for poi in env_config["env"]["pois"]]
    use_order = env_config["env"]["use_order"]

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
        targets_colors=poi_colors,
        x_semidim=map_size[0],
        y_semidim=map_size[1],
        agents_per_target=coupling[0],
        covering_range=obs_radius[0],
        lidar_range=lidar_range[0],
        targets_types=poi_types,
        targets_orders=poi_orders,
        use_order=use_order,
        viewer_zoom=kwargs.pop("viewer_zoom", 1)
    )

    return env
