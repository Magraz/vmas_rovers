from deap import base
from deap import creator
from deap import tools
import torch

import torch.multiprocessing as mp
import random

from vmas import make_env
from vmas.simulator.environment import Environment

from rover_domain import RoverDomain

from policies.mlp import MLP_Policy
from policies.gru import GRU_Policy
from policies.cnn import CNN_Policy

from fitness_critic.fitness_critic import FitnessCritic

from copy import deepcopy
import numpy as np
import os
from pathlib import Path
import yaml
import logging
import pickle
import csv

from itertools import combinations

# Create and configure logger
logging.basicConfig(format="%(asctime)s %(message)s")

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Agent(object):
    def __init__(self, idx: int, parameters: list):
        self.idx = idx
        self.parameters = parameters


class Team(object):
    def __init__(
        self, idx: int, individuals: list[Agent] = None, combination: list = None
    ):
        self.idx = idx
        self.individuals = individuals if individuals is not None else []
        self.combination = combination if combination is not None else []


class JointTrajectory(object):
    def __init__(
        self,
        joint_state_traj: list,
    ):
        self.states = joint_state_traj


class EvalInfo(object):
    def __init__(
        self,
        team: Team,
        team_fitness: float,
        joint_traj: list,
        agent_fitnesses: list[float] = None,
    ):
        self.team = team
        self.agent_fitnesses = agent_fitnesses
        self.team_fitness = team_fitness
        self.joint_traj = joint_traj


class CooperativeCoevolutionaryAlgorithm:
    def __init__(
        self, config_dir, experiment_name: str, trial_id: int, load_checkpoint: bool
    ):

        self.trial_id = trial_id
        self.config_dir = Path(os.path.expanduser(config_dir))
        self.trials_dir = os.path.join(
            self.config_dir.parents[2], "results", experiment_name
        )

        with open(str(self.config_dir), "r") as file:
            self.config = yaml.safe_load(file)

        # Experiment data
        self.trial_name = Path(self.config_dir).stem

        # Set map size
        self.map_size = self.config["env"]["map_size"]

        # Start by setting up variables for different agents
        self.n_agents = len(self.config["env"]["rovers"])
        self.lidar_rays = [
            rover["lidar"]["rays"] for rover in self.config["env"]["rovers"]
        ]
        self.lidar_range = [
            rover["lidar"]["range"] for rover in self.config["env"]["rovers"]
        ]
        self.use_teaming = self.config["teaming"]["use_teaming"]
        self.use_fc = self.config["fitness_critic"]["use_fit_crit"]
        self.fit_crit_loss_type = self.config["fitness_critic"]["loss_type"]

        self.n_eval_per_team = self.config["ccea"]["evaluation"]["n_evaluations"]
        self.team_size = (
            self.config["teaming"]["team_size"] if self.use_teaming else self.n_agents
        )
        self.team_combinations = [
            combo for combo in combinations(range(self.n_agents), self.team_size)
        ]

        self.n_eval_per_team_set = (
            len(self.team_combinations) * self.n_eval_per_team
            if self.use_teaming
            else self.n_eval_per_team
        )

        self.subpop_size = self.config["ccea"]["population"]["subpopulation_size"]

        self.policy_n_hidden = self.config["ccea"]["policy"]["hidden_layers"]
        self.policy_type = self.config["ccea"]["policy"]["type"]
        self.weight_initialization = self.config["ccea"]["weight_initialization"]

        self.fit_crit_type = self.config["fitness_critic"]["type"]
        self.fit_crit_n_hidden = self.config["fitness_critic"]["hidden_layers"]

        self.n_pois = len(self.config["env"]["pois"])
        self.coupling = [poi["coupling"] for poi in self.config["env"]["pois"]]
        self.values = [poi["value"] for poi in self.config["env"]["pois"]]
        self.obs_radius = [
            poi["observation_radius"] for poi in self.config["env"]["pois"]
        ]

        self.n_elites = round(
            self.config["ccea"]["selection"]["n_elites"] * self.subpop_size
        )
        self.n_mutants = self.subpop_size - self.n_elites

        self.fitness_method = self.config["ccea"]["evaluation"]["fitness_method"]

        self.n_steps = self.config["ccea"]["n_steps"]
        self.n_gens = self.config["ccea"]["n_gens"]

        self.nn_template = self.generateTemplateNN()
        self.rover_nn_size = self.nn_template.num_params

        self.use_multiprocessing = self.config["processing"]["use_multiprocessing"]
        self.n_threads = self.config["processing"]["n_threads"]

        # Data loading
        self.load_checkpoint = load_checkpoint

        # Data saving variables
        self.n_gens_between_save = self.config["data"]["n_gens_between_save"]

        # Create the type of fitness we're optimizing
        creator.create("FitnessMax", base.Fitness, weights=(1.0,), values=(0.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        # Now set up the toolbox
        self.toolbox = base.Toolbox()

        self.toolbox.register(
            "subpopulation",
            tools.initRepeat,
            list,
            self.createIndividual,
            n=self.config["ccea"]["population"]["subpopulation_size"],
        )

        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.subpopulation,
            n=self.n_agents,
        )

        # Set up multi processing methods
        if self.use_multiprocessing:
            self.pool = mp.Pool(processes=self.n_threads)
            self.map = self.pool.map_async
        else:
            self.toolbox.register("map", map)
            self.map = map

    def createIndividual(self):
        match (self.weight_initialization):
            case "kaiming":
                temp_model = self.generateTemplateNN()
                params = temp_model.get_params()
        return creator.Individual(params[:].cpu().numpy().astype(np.float32))

    def generateTemplateNN(self):
        match (self.policy_type):

            case "GRU":
                agent_nn = GRU_Policy(
                    input_size=36,
                    hidden_size=self.policy_n_hidden[0],
                    output_size=2,
                    n_layers=1,
                ).to(DEVICE)

            case "CNN":
                agent_nn = CNN_Policy(
                    img_size=self.image_size,
                ).to(DEVICE)

            case "MLP":
                agent_nn = MLP_Policy(
                    input_size=12,
                    hidden_layers=len(self.policy_n_hidden),
                    hidden_size=self.policy_n_hidden[0],
                    output_size=2,
                ).to(DEVICE)

        return agent_nn

    def getBestAgents(self, population) -> list[Agent]:
        best_agents = []

        # Get best agents
        for idx, subpop in enumerate(population):
            # Get the best N individuals
            best_ind = tools.selBest(subpop, 1)[0]
            best_agents.append(Agent(idx=idx, parameters=best_ind))

        return best_agents

    def evaluateBestTeam(self, population):
        # Create evaluation teams
        eval_teams = self.formTeams(population, for_evaluation=True)
        # Create one env
        env = self.create_env(1)

        return self.evaluateTeams(env, eval_teams)

    def formTeams(self, population, for_evaluation: bool = False) -> list[Team]:
        # Start a list of teams
        teams = []

        if for_evaluation:
            joint_policies = 1
        else:
            joint_policies = self.subpop_size

        # For each row in the population of subpops (grabs an individual from each row in the subpops)
        for i in range(joint_policies):

            if for_evaluation:
                agents = self.getBestAgents(population)
            else:
                # Get agents in this row of subpopulations
                agents = [
                    Agent(idx=idx, parameters=subpop[i])
                    for idx, subpop in enumerate(population)
                ]

            # Put the i'th individual on the team if it is inside our team combinations
            for combination in self.team_combinations:

                teams.extend(
                    [
                        Team(
                            idx=i,
                            individuals=[agents[idx] for idx in combination],
                            combination=combination,
                        )
                        for _ in range(self.n_eval_per_team)
                    ]
                )

        return teams

    def evaluateTeams(self, env: Environment, teams: list[Team], render: bool = False):
        # Set up models
        joint_policies = [
            [deepcopy(self.nn_template) for _ in range(self.team_size)] for _ in teams
        ]

        # Load in the weights
        for i, team in enumerate(teams):
            for agent_nn, individual in zip(joint_policies[i], team.individuals):
                agent_nn.set_params(torch.from_numpy(individual.parameters).to(DEVICE))

        # Get initial observations per agent
        observations = env.reset()

        # Store joint states per environment for the first state
        agent_positions = torch.stack([agent.state.pos for agent in env.agents], dim=0)
        joint_states_per_env = [torch.empty((0, 2)).to(DEVICE) for _ in teams]

        for i, j_states in enumerate(joint_states_per_env):
            joint_states_per_env[i] = torch.cat(
                (j_states, agent_positions[:, i, :]), dim=0
            )

        G_list = []

        # Start evaluation
        for _ in range(self.n_steps):

            stacked_obs = torch.stack(observations, -1)

            actions = [torch.empty((0, 2)).to(DEVICE) for _ in range(self.n_agents)]

            for observation, joint_policy in zip(stacked_obs, joint_policies):

                for i, policy in enumerate(joint_policy):
                    policy_output = policy.forward(observation[:, i]).unsqueeze(0)
                    actions[i] = torch.cat(
                        (
                            actions[i],
                            policy_output
                            * self.config["ccea"]["policy"]["rover_max_velocity"],
                        ),
                        dim=0,
                    )

            observations, rewards, dones, info = env.step(actions)

            # Store joint states per environment
            agent_positions = torch.stack(
                [agent.state.pos for agent in env.agents], dim=0
            )

            for i, j_states in enumerate(joint_states_per_env):
                joint_states_per_env[i] = torch.cat(
                    (j_states, agent_positions[:, i, :]), dim=0
                )

            G_list.append(torch.stack(rewards, dim=0)[0])

            if render:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=True,
                )

        # Compute team fitness
        match (self.fitness_method):

            case "aggregate":
                team_fitness_per_env = torch.sum(torch.stack(G_list), dim=0).tolist()

        # Generate evaluation infos
        eval_infos = [
            EvalInfo(
                team=team,
                team_fitness=team_fitness_per_env[i],
                joint_traj=JointTrajectory(
                    joint_states_per_env[i],
                ),
            )
            for i, team in enumerate(teams)
        ]

        return eval_infos

    def mutateIndividual(self, individual):

        individual *= np.random.normal(
            loc=self.config["ccea"]["mutation"]["mean"],
            scale=self.config["ccea"]["mutation"]["std_deviation"],
            size=np.shape(individual),
        )

    def mutate(self, population):
        # Don't mutate the elites
        for n_individual in range(self.n_mutants):

            mutant_idx = n_individual + self.n_elites

            for subpop in population:
                self.mutateIndividual(subpop[mutant_idx])
                subpop[mutant_idx].fitness.values = (np.float32(0.0),)

    def selectSubPopulation(self, subpopulation):
        # Get the best N individuals
        elites = tools.selBest(subpopulation, self.n_elites)

        non_elites = tools.selTournament(
            subpopulation, len(subpopulation) - self.n_elites, tournsize=2
        )

        offspring = elites + non_elites

        # Return a deepcopy so that modifying an individual that was selected does not modify every single individual
        # that came from the same selected individual
        return [deepcopy(individual) for individual in offspring]

    def select(self, population):
        # Perform a selection on that subpopulation and add it to the offspring population
        return [self.selectSubPopulation(subpop) for subpop in population]

    def shuffle(self, population):
        for subpop in population:
            random.shuffle(subpop)

    def trainFitnessCritics(
        self,
        fitness_critics: list[FitnessCritic],
        eval_infos: list[EvalInfo],
    ):
        fc_loss = []

        # Collect trajectories from eval_infos
        for eval_info in eval_infos:
            for idx in eval_info.team_formation:
                fitness_critics[idx].add(
                    eval_info.joint_traj.observations[:, idx, :],
                    np.float32(eval_info.team_fitness),
                )

        # Train fitness critics
        for fc in fitness_critics:
            accum_loss = fc.train(epochs=self.config["fitness_critic"]["epochs"])
            fc_loss.append(accum_loss)

        return fc_loss

    def assignFitnesses(
        self,
        fitness_critics,
        eval_infos: list[EvalInfo],
    ):
        for eval_info in eval_infos:
            for individual in eval_info.team.individuals:
                if individual.parameters.fitness.values[0] <= eval_info.team_fitness:
                    individual.parameters.fitness.values = (eval_info.team_fitness,)

    def setPopulation(self, population, offspring):
        for subpop, subpop_offspring in zip(population, offspring):
            subpop[:] = subpop_offspring

    def createEvalFitnessCSV(self, eval_fit_dir):
        header = ["gen", "team_fitness"]

        with open(eval_fit_dir, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows([header])

    def writeEvalFitnessCSV(self, eval_fit_dir, eval_infos):

        team_fitnesses = [eval_info.team_fitness for eval_info in eval_infos]

        # Now save it all to the csv
        with open(eval_fit_dir, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for eval_info in eval_infos:
                writer.writerow([self.gen, *team_fitnesses])

    def init_fitness_critics(self):
        # Initialize fitness critics
        fc = None

        if self.use_fc:

            loss_fn = 0

            match self.fit_crit_loss_type:
                case "MSE":
                    loss_fn = 0
                case "MAE":
                    loss_fn = 1
                case "MSE+MAE":
                    loss_fn = 2

            fc = [
                FitnessCritic(
                    device=DEVICE,
                    model_type=self.fit_crit_type,
                    loss_fn=loss_fn,
                    episode_size=self.n_steps,
                    hidden_size=self.fit_crit_n_hidden,
                    n_layers=len(self.fit_crit_n_hidden),
                )
                for _ in range(self.n_agents)
            ]

        return fc

    def create_env(self, n_envs: int) -> Environment:
        # Set up the enviornment
        env = make_env(
            scenario=RoverDomain(),
            num_envs=n_envs,
            device=DEVICE,
            seed=None,
            # Environment specific variables
            n_agents=self.n_agents,
            n_targets=self.n_pois,
            x_semidim=self.map_size[0],
            y_semidim=self.map_size[1],
            agents_per_target=self.coupling[0],
            covering_range=self.obs_radius[0],
            n_lidar_rays_entities=self.lidar_rays[0],
            n_lidar_rays_agents=self.lidar_rays[0],
            lidar_range=self.lidar_range[0],
            env_count=n_envs,
        )
        return env

    def run(self):

        # Set trial directory name
        trial_folder_name = "_".join(("trial", str(self.trial_id), self.trial_name))
        trial_dir = os.path.join(self.trials_dir, trial_folder_name)
        eval_fit_dir = f"{trial_dir}/fitness.csv"
        checkpoint_name = os.path.join(trial_dir, "checkpoint.pickle")

        # Create directory for saving data
        if not os.path.isdir(trial_dir):
            os.makedirs(trial_dir)

        # Load checkpoint
        if self.load_checkpoint:

            with open(checkpoint_name, "rb") as handle:
                checkpoint = pickle.load(handle)
                pop = checkpoint["population"]
                self.checkpoint_gen = checkpoint["gen"]
                fc_params = checkpoint["fitness_critics"]

            # Load fitness critics params
            if self.use_fc:
                fitness_critics = self.init_fitness_critics()
                for fc, params in zip(fitness_critics, fc_params):
                    fc.model.set_params(params)
            else:
                fitness_critics = None

            # Set fitness csv file to checkpoint
            new_fit_path = os.path.join(trial_dir, "fitness_edit.csv")
            with open(eval_fit_dir, "r") as inp, open(new_fit_path, "w") as out:
                writer = csv.writer(out)
                for row in csv.reader(inp):
                    if row[0].isdigit():
                        gen = int(row[0])
                        if gen <= self.checkpoint_gen:
                            writer.writerow(row)
                    else:
                        writer.writerow(row)

            # Remove old fitness file
            os.remove(eval_fit_dir)
            # Rename new fitness file
            os.rename(new_fit_path, eval_fit_dir)

        else:
            # Initialize the population
            pop = self.toolbox.population()

            # Create csv file for saving evaluation fitnesses
            self.createEvalFitnessCSV(eval_fit_dir)

            # Initialize fitness critics
            if self.use_fc:
                fitness_critics = self.init_fitness_critics()
            else:
                fitness_critics = None

        # Create environment
        env = self.create_env(self.subpop_size)

        for n_gen in range(self.n_gens + 1):

            # Set gen counter global var
            self.gen = n_gen

            # Get loading bar up to checkpoint
            if self.load_checkpoint and n_gen <= self.checkpoint_gen:
                continue

            # Perform selection
            offspring = self.select(pop)

            # Perform mutation
            self.mutate(offspring)

            # Shuffle subpopulations in offspring
            # to make teams random
            self.shuffle(offspring)

            # Form teams for evaluation
            teams = self.formTeams(offspring)

            # Evaluate each team
            eval_infos = self.evaluateTeams(env, teams)

            # Train Fitness Critics
            if self.use_fc:
                fc_loss = self.trainFitnessCritics(fitness_critics, eval_infos)
                self.writeFitCritLossCSV(trial_dir, fc_loss)

            # Now assign fitnesses to each individual
            self.assignFitnesses(fitness_critics, eval_infos)

            # Evaluate a team with the best individual from each subpopulation
            eval_infos = self.evaluateBestTeam(offspring)

            # Now populate the population with individuals from the offspring
            self.setPopulation(pop, offspring)

            # Save fitnesses
            self.writeEvalFitnessCSV(eval_fit_dir, eval_infos)

            # Save trajectories and checkpoint
            if n_gen % self.n_gens_between_save == 0:

                # Save checkpoint
                with open(os.path.join(trial_dir, "checkpoint.pickle"), "wb") as handle:
                    pickle.dump(
                        {
                            "best_team": eval_infos[0].team,
                            "population": pop,
                            "gen": n_gen,
                            "fitness_critics": (
                                [fc.params for fc in fitness_critics]
                                if self.use_fc
                                else None
                            ),
                        },
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

        if self.use_multiprocessing:
            self.pool.close()


def runCCEA(config_dir, experiment_name: str, trial_id: int, load_checkpoint: bool):
    ccea = CooperativeCoevolutionaryAlgorithm(
        config_dir, experiment_name, trial_id, load_checkpoint
    )
    return ccea.run()
