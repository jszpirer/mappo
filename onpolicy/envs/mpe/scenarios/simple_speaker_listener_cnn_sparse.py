import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario
import torch


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
        # Initialize lists for sparse tensor indices and values for goal_color
        indices_goal_color = []
        values_goal_color = []
        
        # Check if agent has a goal
        if agent.goal_b is not None:
            for i in range(len(agent.goal_b.color)):
                # Set the position for each color in goal_color
                indices_goal_color.append(i)
                values_goal_color.append(agent.goal_b.color[i])
        
        # Create one-dimensional sparse tensor for goal_color
        sparse_tensor_goal_color = torch.sparse_coo_tensor(torch.tensor([indices_goal_color]), torch.tensor(values_goal_color), size=(3,)).coalesce()

        # Initialize lists for sparse tensor indices and values
        indices_red = [[], []]
        indices_blue = [[], []]
        indices_green = [[], []]
        values_red = []
        values_blue = []
        values_green = []
        
        # Calculate positions of all entities in this agent's reference frame
        for i, entity in enumerate(world.landmarks):
            distance = entity.state.p_pos - agent.state.p_pos
            coef = world.grid_resolution / (world.limit * 4)
            scale = (world.grid_resolution // 2) - 1 + world.grid_resolution % 2
            x = round(coef * distance[0]) + scale
            y = round(coef * distance[1]) + scale
            
            # Determine color based on index
            color_index = i % 3
            if color_index == 0:
                indices_red[0].append(x)
                indices_red[1].append(y)
                values_red.append(1)
            elif color_index == 1:
                indices_blue[0].append(x)
                indices_blue[1].append(y)
                values_blue.append(1)
            elif color_index == 2:
                indices_green[0].append(x)
                indices_green[1].append(y)
                values_green.append(1)
        
        # Create sparse tensors
        sparse_tensor_red = torch.sparse_coo_tensor(torch.tensor(indices_red), torch.tensor(values_red), size=(world.grid_resolution, world.grid_resolution)).coalesce()
        sparse_tensor_blue = torch.sparse_coo_tensor(torch.tensor(indices_blue), torch.tensor(values_blue), size=(world.grid_resolution, world.grid_resolution)).coalesce()
        sparse_tensor_green = torch.sparse_coo_tensor(torch.tensor(indices_green), torch.tensor(values_green), size=(world.grid_resolution, world.grid_resolution)).coalesce()

        # Initialize lists for sparse tensor indices and values for communication
        indices_comm = []
        values_comm = []
        
        # Calculate communication states of all other agents
        for other in world.agents:
            if other is agent or (other.state.c is None):
                continue
            indices = [index for index, value in enumerate(other.state.c) if value != 1]
            for index in indices:
                indices_comm.append(index)
                values_comm.append(1)
        
        # Create one-dimensional sparse tensor for communication
        sparse_tensor_comm = torch.sparse_coo_tensor(torch.tensor([indices_comm]), torch.tensor(values_comm), size=(world.dim_c,)).coalesce()


        # speaker
        if not agent.movable:
            observations = [sparse_tensor_goal_color]
            return observations
        # listener
        if agent.silent:
            sparse_tensor_vel = torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.tensor(agent.state.p_vel), size=(2,)).coalesce()
            observations = [sparse_tensor_vel, sparse_tensor_red, sparse_tensor_blue, sparse_tensor_green, sparse_tensor_comm]
            return observations
