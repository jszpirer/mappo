import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        # set any world properties first
        world.world_length = args.episode_length
        world.dim_c = 10
        world.limit = 4
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.num_agents = 2  # 2
        world.grid_resolution = args.grid_resolution
        world.nb_additional_data = args.nb_additional_data
        assert world.num_agents == 2, (
            "only 2 agents is supported, check the config.py.")
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.15
            agent.u_noise = 1
            agent.max_speed = 0.51
            # agent.u_noise = 1e-1
            # agent.c_noise = 1e-1
        # add landmarks
        world.num_landmarks = 3  # 3
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want other agent to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        world.agents[1].goal_a = world.agents[0]
        world.agents[1].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[1].color = np.array([0.25, 0.75, 0.25])
        world.landmarks[2].color = np.array([0.25, 0.25, 0.75])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color
        world.agents[1].goal_a.color = world.agents[1].goal_b.color
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-3.85, +3.85, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-3.85, +3.85, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(
            np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2  # np.exp(-dist2)

    def observation(self, agent, world):
        range_observation = 3
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color
        # get positions of all entities in this agent's reference frame
        entity_pos = np.zeros((world.grid_resolution, world.grid_resolution))
        for entity in world.landmarks:  # world.entities:
            if np.linalg.norm(entity.state.p_pos - agent.state.p_pos) <= range_observation:
                distance = entity.state.p_pos - agent.state.p_pos
                coef = world.grid_resolution/(world.limit*4)
                scale = (world.grid_resolution//2) - 1
                entity_pos[round(coef*distance[0]) + scale][round(coef*distance[1]) + scale] = 1
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent:
                continue
            if np.linalg.norm(other.state.p_pos - agent.state.p_pos) <= range_observation:
                comm.append(other.state.c)
            else:
                comm.append(np.zeros(10))
        vel = [np.pad(agent.state.p_vel, (0, world.grid_resolution-2), 'constant', constant_values = 0)]
        goal_color = [np.pad(goal_color[1], (0, world.grid_resolution-3), 'constant', constant_values = 0)]
        comm_padded = [np.pad(comm[0], (0, world.grid_resolution-10), 'constant', constant_values = 0)]
        observations = np.concatenate((np.array(vel), np.array(goal_color), np.array(comm_padded), np.array(entity_pos)), axis=0)
        return observations
