import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 3
        world.num_landmarks = args.num_landmarks  # 3
        world.collaborative = True
        world.limit = 4
        world.grid_resolution = args.grid_resolution
        # add agents
        world.num_agents = args.num_agents  # 2
        assert world.num_agents == 2, (
            "only 2 agents is supported, check the config.py.")
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.15
            # agent.u_noise = 1
            agent.max_speed = 0.51
        # speaker
        world.agents[0].movable = False
        # listener
        world.agents[1].silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65, 0.15, 0.15])
        world.landmarks[1].color = np.array([0.15, 0.65, 0.15])
        world.landmarks[2].color = np.array([0.15, 0.15, 0.65])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + \
            np.array([0.45, 0.45, 0.45])
        # set random initial states
        for agent in world.agents:
            if agent.movable:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            else:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return reward(agent, reward)

    def reward(self, agent, world):
        # squared distance from listener to landmark
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # goal color
        goal_color = [np.zeros((world.grid_resolution, world.grid_resolution)) for _ in range(world.num_landmarks)]
        if agent.goal_b is not None:
            for i in range(len(agent.goal_b.color)):
              goal_color[i][0][0] = agent.goal_b.color[i]

        # Initialize entity positions
        entity_positions = [np.zeros((world.grid_resolution, world.grid_resolution)) for _ in range(world.num_landmarks)]

        # Calculate positions of all entities in this agent's reference frame
        for i, entity in enumerate(world.landmarks):
            distance = entity.state.p_pos - agent.state.p_pos
            coef = world.grid_resolution / (world.limit * 4)
            scale = (world.grid_resolution // 2) - 1 + world.grid_resolution%2
            x = round(coef * distance[0]) + scale
            y = round(coef * distance[1]) + scale
            entity_positions[i][x][y] = 1

        # communication of all other agents
        comm = [np.zeros((world.grid_resolution, world.grid_resolution)) for _ in range(world.dim_c)]
        
        for other in world.agents:
            if other is agent or (other.state.c is None):
                continue
            indices = [index for index, value in enumerate(other.state.c) if value != 1]
            for index in indices:
                comm[index][0][0] = 1

        # speaker
        if not agent.movable:
            observations = np.concatenate((np.array([np.zeros(world.grid_resolution)]), np.vstack(goal_color),  np.vstack(np.zeros((3, world.grid_resolution, world.grid_resolution)))), axis=0)
            return observations
        # listener
        if agent.silent:
            vel = [np.pad(agent.state.p_vel, (0, world.grid_resolution-2), 'constant', constant_values = 0)]
            observations = np.concatenate((np.array(vel), np.vstack(comm), np.vstack(entity_positions)), axis=0)
            return observations
