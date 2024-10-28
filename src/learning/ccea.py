from deap import base
from deap import creator
from deap import tools
import torch

import random

from vmas.simulator.environment import Environment
from vmas.simulator.utils import save_video

from policies.mlp import MLP_Policy
from policies.gru import GRU_Policy
from policies.cnn import CNN_Policy

from fitness_critic.fitness_critic import FitnessCritic
from domain.create_env import create_env

from copy import deepcopy
import numpy as np
import os
from pathlib import Path
import yaml
import logging
import pickle
import csv

from itertools import combinations
from operator import attrgetter

# Create and configure logger
logging.basicConfig(format="%(asctime)s %(message)s")

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


class Team(object):
    def __init__(
        self,
        idx: int,
        individuals: list = None,
        combination: list = None,
    ):
        self.idx = idx
        self.individuals = individuals if individuals is not None else []
        self.combination = combination if combination is not None else []


class JointTrajectory(object):
    def __init__(self, joint_state_traj: list, joint_obs_traj: list):
        self.states = joint_state_traj
        self.observations = joint_obs_traj


class EvalInfo(object):
    def __init__(
        self,
        team: Team,
        team_fitness: float,
        agent_fitnesses: list[float],
        joint_traj: list,
    ):
        self.team = team
        self.agent_fitnesses = agent_fitnesses
        self.team_fitness = team_fitness
        self.joint_traj = joint_traj


class CooperativeCoevolutionaryAlgorithm:
    def __init__(self, config_dir, experiment_name: str, trial_id: int):

        self.trial_id = trial_id
        self.video_name = f"{experiment_name}_{trial_id}"
        self.config_dir = Path(os.path.expanduser(config_dir))
        self.trials_dir = os.path.join(
            self.config_dir.parents[2], "results", experiment_name
        )

        with open(str(self.config_dir), "r") as file:
            self.config = yaml.safe_load(file)

        # Experiment data
        self.trial_name = Path(self.config_dir).stem

        # Environment data
        self.map_size = self.config["env"]["map_size"]
        self.n_steps = self.config["ccea"]["n_steps"]
        self.observation_size = 8
        self.action_size = 2

        # Agent data
        self.n_agents = len(self.config["env"]["rovers"])

        # POIs data
        self.n_pois = len(self.config["env"]["pois"])

        # Learning data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_teaming = self.config["teaming"]["use_teaming"]
        self.use_fc = self.config["fitness_critic"]["use_fit_crit"]
        self.fit_crit_loss_type = self.config["fitness_critic"]["loss_type"]

        self.team_size = (
            self.config["teaming"]["team_size"] if self.use_teaming else self.n_agents
        )
        self.team_combinations = [
            combo for combo in combinations(range(self.n_agents), self.team_size)
        ]

        self.fitness_shaping_method = self.config["ccea"]["fitness_shaping"]

        self.subpop_size = self.config["ccea"]["population"]["subpopulation_size"]

        self.policy_n_hidden = self.config["ccea"]["policy"]["hidden_layers"]
        self.policy_type = self.config["ccea"]["policy"]["type"]
        self.weight_initialization = self.config["ccea"]["weight_initialization"]

        self.fit_crit_type = self.config["fitness_critic"]["type"]
        self.fit_crit_n_hidden = self.config["fitness_critic"]["hidden_layers"]

        self.n_mutants = self.subpop_size // 2

        self.fitness_calculation = self.config["ccea"]["evaluation"][
            "fitness_calculation"
        ]

        self.n_gens = self.config["ccea"]["n_gens"]

        self.selection_method = self.config["ccea"]["selection"]

        self.std_dev_list = np.arange(
            start=self.config["ccea"]["mutation"]["max_std_deviation"],
            stop=self.config["ccea"]["mutation"]["min_std_deviation"],
            step=-(
                (
                    self.config["ccea"]["mutation"]["max_std_deviation"]
                    - self.config["ccea"]["mutation"]["min_std_deviation"]
                )
                / (self.n_gens + 1)
            ),
        )

        self.alpha = 0.0
        self.alpha_max = 0.5

        self.nn_template = self.generateTemplateNN()

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
                    input_size=self.observation_size,
                    hidden_size=self.policy_n_hidden[0],
                    output_size=self.action_size,
                    n_layers=1,
                ).to(self.device)

            case "CNN":
                agent_nn = CNN_Policy(
                    img_size=self.image_size,
                ).to(self.device)

            case "MLP":
                agent_nn = MLP_Policy(
                    input_size=self.observation_size,
                    hidden_layers=len(self.policy_n_hidden),
                    hidden_size=self.policy_n_hidden[0],
                    output_size=self.action_size,
                ).to(self.device)

        return agent_nn

    def getBestAgents(self, population) -> list:
        best_agents = []

        # Get best agents
        for subpop in population:
            # Get the best N individuals
            best_ind = tools.selBest(subpop, 1)[0]
            best_agents.append(best_ind)

        return best_agents

    def formTeams(self, population, joint_policies: int) -> list[Team]:
        # Start a list of teams
        teams = []

        # For each row in the population of subpops (grabs an individual from each row in the subpops)
        for i in range(joint_policies):

            # Get agents in this row of subpopulations
            agents = [subpop[i] for subpop in population]

            # Put the i'th individual on the team if it is inside our team combinations
            for combination in self.team_combinations:

                teams.append(
                    Team(
                        idx=i,
                        individuals=[agents[idx] for idx in combination],
                        combination=combination,
                    )
                )

        return teams

    def evaluateTeams(
        self,
        env: Environment,
        teams: list[Team],
        render: bool = False,
        save_render: bool = False,
    ):
        # Set up models
        joint_policies = [
            [deepcopy(self.nn_template) for _ in range(self.team_size)] for _ in teams
        ]

        # Load in the weights
        for i, team in enumerate(teams):
            for agent_nn, individual in zip(joint_policies[i], team.individuals):
                agent_nn.set_params(torch.from_numpy(individual).to(self.device))

        # Get initial observations per agent
        observations = env.reset()

        # Store joint states per environment for the first state
        agent_positions = torch.stack([agent.state.pos for agent in env.agents], dim=0)
        joint_states_per_env = [
            torch.empty((0, self.action_size)).to(self.device) for _ in teams
        ]

        tranposed_stacked_obs = (
            torch.stack(observations, -1).transpose(0, 1).transpose(0, -1)
        )
        joint_observations_per_env = [
            torch.empty((0, self.observation_size)).to(self.device) for _ in teams
        ]

        for i, (j_states, j_obs) in enumerate(
            zip(joint_states_per_env, joint_observations_per_env)
        ):
            joint_states_per_env[i] = torch.cat(
                (j_states, agent_positions[:, i, :]), dim=0
            )
            joint_observations_per_env[i] = torch.cat(
                (j_obs, tranposed_stacked_obs[:, i, :]), dim=0
            )

        G_list = []
        D_list = []
        frame_list = []

        # Start evaluation
        for step in range(self.n_steps):

            stacked_obs = torch.stack(observations, -1)

            actions = [
                torch.empty((0, self.action_size)).to(self.device)
                for _ in range(self.n_agents)
            ]

            for observation, joint_policy in zip(stacked_obs, joint_policies):

                for i, policy in enumerate(joint_policy):
                    policy_output = policy.forward(observation[:, i])
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

            for i, (j_states, j_obs) in enumerate(
                zip(joint_states_per_env, joint_observations_per_env)
            ):
                joint_states_per_env[i] = torch.cat(
                    (j_states, agent_positions[:, i, :]), dim=0
                )
                joint_observations_per_env[i] = torch.cat(
                    (j_obs, stacked_obs.transpose(0, 1).transpose(0, -1)[:, i, :]),
                    dim=0,
                )

            G_list.append(torch.stack([g[: len(teams)] for g in rewards], dim=0)[0])

            D_list.append(
                torch.stack([d[len(teams) : len(teams) * 2] for d in rewards], dim=0)
            )

            # Visualization
            if render:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=True,
                )
                if save_render:
                    frame_list.append(frame)

        # Save video
        if render and save_render:
            save_video(self.video_name, frame_list, fps=1 / env.scenario.world.dt)

        # Compute team fitness
        match (self.fitness_calculation):

            case "aggregate":
                g_per_env = torch.sum(torch.stack(G_list), dim=0).tolist()
                d_per_env = torch.transpose(
                    torch.sum(torch.stack(D_list), dim=0), dim0=0, dim1=1
                ).tolist()

            case "last_step":
                g_per_env = G_list[-1].tolist()
                d_per_env = torch.transpose(D_list[-1], dim0=0, dim1=1).tolist()

        # Generate evaluation infos
        eval_infos = [
            EvalInfo(
                team=team,
                team_fitness=g_per_env[i],
                agent_fitnesses=d_per_env[i],
                joint_traj=JointTrajectory(
                    joint_state_traj=joint_states_per_env[i].reshape(
                        self.n_agents, self.n_steps + 1, self.action_size
                    ),
                    joint_obs_traj=joint_observations_per_env[i].reshape(
                        self.n_agents, self.n_steps + 1, self.observation_size
                    ),
                ),
            )
            for i, team in enumerate(teams)
        ]

        return eval_infos

    def mutateIndividual(self, individual):

        individual += np.random.normal(
            loc=self.config["ccea"]["mutation"]["mean"],
            scale=self.std_dev_list[self.gen],
            size=np.shape(individual),
        )

        # for i, size in enumerate(self.nn_template.size_per_layer):
        #     if i == 0:
        #         individual[0:size] += np.random.normal(
        #             loc=self.config["ccea"]["mutation"]["mean"],
        #             scale=self.std_dev_list[self.gen],
        #             size=np.shape(individual),
        #         )
        #     else:
        #         individual[
        #             self.nn_template.size_per_layer[
        #                 i - 1
        #             ] : self.nn_template.size_per_layer[i - 1]
        #             + size
        #         ] += np.random.normal(
        #             loc=self.config["ccea"]["mutation"]["mean"],
        #             scale=self.std_dev_list[self.gen],
        #             size=np.shape(individual),
        #         )

    def mutate(self, population):
        # Don't mutate the elites
        for n_individual in range(self.n_mutants):

            mutant_idx = n_individual + self.n_mutants

            for subpop in population:
                self.mutateIndividual(subpop[mutant_idx])
                subpop[mutant_idx].fitness.values = (np.float32(0.0),)

    def binarySelection(self, individuals, tournsize: int, fit_attr: str = "fitness"):

        # Shuffle the list randomly
        random.shuffle(individuals)

        # Create list of random pairs without repetition
        pairs_of_candidates = [
            (individuals[i], individuals[i + 1])
            for i in range(0, len(individuals) - 1, tournsize)
        ]

        chosen_ones = [
            max(candidates, key=attrgetter(fit_attr))
            for candidates in pairs_of_candidates
        ]

        return chosen_ones

    def epsilonGreedySelection(
        self,
        individuals,
        k: int,
        epsilon: float,
        fit_attr: str = "fitness",
    ):

        chosen_ones = []

        individuals_copy = individuals[:]

        for _ in range(k):
            if np.random.choice([True, False], 1, p=[1 - epsilon, epsilon]):
                chosen_one = max(individuals_copy, key=attrgetter(fit_attr))
                individuals_copy.remove(chosen_one)
            else:
                chosen_one = random.choice(individuals)

            chosen_ones.append(chosen_one)

        return sorted(chosen_ones, key=attrgetter(fit_attr), reverse=True)

    def selectSubPopulation(self, subpopulation):

        match (self.selection_method):
            case "binary":
                chosen_ones = self.binarySelection(subpopulation, tournsize=2)
            case "epsilon":
                chosen_ones = self.epsilonGreedySelection(
                    subpopulation, self.subpop_size // 2, epsilon=0.1
                )
            case "tournament":
                chosen_ones = tools.selTournament(
                    subpopulation, self.subpop_size // 2, 2
                )

        offspring = chosen_ones + chosen_ones

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
            for idx in eval_info.team.combination:
                fitness_critics[idx].add(
                    eval_info.joint_traj.observations[idx, :, :],
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
        hof_eval_info: EvalInfo,
    ):

        match (self.fitness_shaping_method):
            case "global":
                for eval_info in eval_infos:
                    for individual in eval_info.team.individuals:
                        individual.fitness.values = (eval_info.team_fitness,)
            case "hof_difference":

                env = create_env(
                    self.config_dir,
                    n_envs=self.subpop_size * self.n_agents,
                    device=self.device,
                )

                mod_hof_teams = []
                mod_idx_hof_teams = []

                for eval_info in eval_infos:
                    for individual, combo_idx in zip(
                        eval_info.team.individuals,
                        eval_info.team.combination,
                    ):
                        hof_eval_info_copy = deepcopy(hof_eval_info)

                        hof_eval_info_copy.team.individuals[combo_idx] = individual

                        mod_hof_teams.append(hof_eval_info_copy.team)
                        mod_idx_hof_teams.append(combo_idx)

                mod_hof_eval_infos = self.evaluateTeams(
                    env,
                    mod_hof_teams,
                )

                for mod_idx, mod_hof_eval_info in zip(
                    mod_idx_hof_teams, mod_hof_eval_infos
                ):

                    d_hof = mod_hof_eval_info.team_fitness - hof_eval_info.team_fitness

                    mod_hof_eval_info.team.individuals[mod_idx].fitness.values = (
                        (1 - self.alpha) * eval_info.agent_fitnesses[combo_idx]
                        + self.alpha * d_hof,
                    )
            case "difference":
                for eval_info in eval_infos:
                    for individual, combo_idx in zip(
                        eval_info.team.individuals,
                        eval_info.team.combination,
                    ):
                        individual.fitness.values = (
                            eval_info.agent_fitnesses[combo_idx],
                        )

    def setPopulation(self, population, offspring):
        for subpop, subpop_offspring in zip(population, offspring):
            subpop[:] = subpop_offspring

    def createEvalFitnessCSV(self, eval_fit_dir):
        header = ["gen", "avg_team_fitness", "best_team_fitness"]

        with open(eval_fit_dir, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows([header])

    def writeEvalFitnessCSV(self, eval_fit_dir, avg_fitness, best_fitness):

        # Now save it all to the csv
        with open(eval_fit_dir, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.gen, avg_fitness, best_fitness])

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
                    device=self.device,
                    model_type=self.fit_crit_type,
                    loss_fn=loss_fn,
                    episode_size=self.n_steps,
                    hidden_size=self.fit_crit_n_hidden,
                    n_layers=len(self.fit_crit_n_hidden),
                )
                for _ in range(self.n_agents)
            ]

        return fc

    def select_hof_team(self, eval_infos: list[EvalInfo], hof_eval_info: EvalInfo):

        best_eval_info = max(eval_infos, key=lambda item: item.team_fitness)

        if best_eval_info.team_fitness > hof_eval_info.team_fitness:
            new_hol_eval_info = best_eval_info
        else:
            new_hol_eval_info = hof_eval_info

        return new_hol_eval_info

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
        checkpoint_exists = Path(checkpoint_name).is_file()
        if checkpoint_exists:
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

        # Create environment for hof team
        env = create_env(self.config_dir, n_envs=1, device=self.device)

        hof_team = self.formTeams(pop, joint_policies=1)

        hof_eval_info = self.evaluateTeams(env, hof_team)[0]

        # Create environment
        env = create_env(self.config_dir, n_envs=self.subpop_size, device=self.device)

        for n_gen in range(self.n_gens + 1):

            # Set gen counter global var
            self.gen = n_gen

            # Get loading bar up to checkpoint
            if checkpoint_exists and n_gen <= self.checkpoint_gen:
                continue

            # Perform selection
            offspring = self.select(pop)

            # Perform mutation
            self.mutate(offspring)

            # Shuffle subpopulations in offspring
            # to make teams random
            self.shuffle(offspring)

            # Form teams for evaluation
            teams = self.formTeams(offspring, joint_policies=self.subpop_size)

            # Evaluate each team
            eval_infos = self.evaluateTeams(env, teams)

            # Train Fitness Critics
            if self.use_fc:
                fc_loss = self.trainFitnessCritics(fitness_critics, eval_infos)
                # self.writeFitCritLossCSV(trial_dir, fc_loss)

            # Now assign fitnesses to each individual
            self.assignFitnesses(fitness_critics, eval_infos, hof_eval_info)

            # Select new hof team
            hof_eval_info = self.select_hof_team(eval_infos, hof_eval_info)

            # Update alpha
            self.alpha += self.alpha_max / (0.8 * self.n_gens)

            # Evaluate best team of generation
            avg_team_fitness = (
                sum([eval_info.team_fitness for eval_info in eval_infos])
                / self.subpop_size
            )
            best_team_eval_info = max(eval_infos, key=lambda item: item.team_fitness)

            # Now populate the population with individuals from the offspring
            self.setPopulation(pop, offspring)

            # Save fitnesses
            self.writeEvalFitnessCSV(
                eval_fit_dir, avg_team_fitness, best_team_eval_info.team_fitness
            )

            # Save trajectories and checkpoint
            if n_gen % self.n_gens_between_save == 0:

                # Save checkpoint
                with open(os.path.join(trial_dir, "checkpoint.pickle"), "wb") as handle:
                    pickle.dump(
                        {
                            "best_team": best_team_eval_info.team,
                            "hof_team": hof_eval_info.team,
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


def runCCEA(config_dir, experiment_name: str, trial_id: int):
    ccea = CooperativeCoevolutionaryAlgorithm(config_dir, experiment_name, trial_id)
    return ccea.run()
