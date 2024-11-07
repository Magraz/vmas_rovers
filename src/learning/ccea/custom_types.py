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
