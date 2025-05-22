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
        world.num_agents = args.num_agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.15
            # agent.u_noise = 1
            agent.max_speed = 0.51
        for i in range(world.num_agents):
            if i%2 == 0:
                # speaker
                world.agents[i].movable = False
            else:
                # listener
                world.agents[i].silent = True
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
        color_goal = np.random.choice(world.landmarks)
        for i in range(world.num_agents):
            if i%2 == 0:
                world.agents[i].goal_a = world.agents[i+1]
                world.agents[i].goal_b = color_goal
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65, 0.15, 0.15])
        world.landmarks[1].color = np.array([0.15, 0.65, 0.15])
        world.landmarks[2].color = np.array([0.15, 0.15, 0.65])
        # special colors for goals
        for i in range(world.num_agents):
            if i%2 == 0:
                world.agents[i].goal_a.color = world.agents[i].goal_b.color + \
                    np.array([0.45, 0.45, 0.45])
        # set random initial states
        for agent in world.agents:
            if agent.movable:
                agent.state.p_pos = np.random.uniform(-1.5, +1.5, world.dim_p)
            else:
                agent.state.p_pos = np.random.uniform(-1.5, +1.5, world.dim_p)
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
        dist = 0
        for i in range(world.num_agents):
            if i%2 == 0:
                a = world.agents[i]
                dist += np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        return -dist

    def observation(self, agent, world):
        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # entity positions as many channels as there are different colors for the landmarks
        entity_pos = np.zeros((3, 2, world.num_landmarks))
        
        # Calculate positions of all entities in this agent's reference frame
        for i, entity in enumerate(world.landmarks):
            distance = entity.state.p_pos - agent.state.p_pos
            coef = world.grid_resolution / (world.limit * 4)
            scale = (world.grid_resolution // 2) - 1 + world.grid_resolution % 2
            x = round(coef * distance[0]) + scale
            y = round(coef * distance[1]) + scale            
            
            # Determine color based on index
            color_index = i % 3
            entity_pos[color_index][0][i] = x
            entity_pos[color_index][1][i] = y

        # communication of all other agents
        comm = np.zeros(world.dim_c)
        for other in world.agents:
            if other is agent or (other.state.c is None):
                continue
            comm = other.state.c


        # speaker
        if not agent.movable:
            observations = np.empty([1], dtype=object)
            observations[:] = [goal_color]
            return observations
        # listener
        if agent.silent:
            observations = np.empty([5], dtype=object)
            observations[:] = [agent.state.p_vel, entity_pos[0], entity_pos[1], entity_pos[2], comm]
            return observations
