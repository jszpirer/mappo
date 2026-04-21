import numpy as np
from math import ceil, cos, sin, pi
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.limit = 3.75
        world.num_agents = args.num_agents
        world.collaborative = True
        world.grid_resolution = args.grid_resolution
        world.grid_resolution_critic = args.grid_resolution_critic
        world.nb_additional_data = args.nb_additional_data
        world.omniscient_critic = True
        world.use_directions = args.use_directions
        world.discrete_actions = args.discrete_action
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            # agent.u_noise = 1
            agent.max_speed = 0.51
            if not world.discrete_actions:
                agent.u_range = agent.max_speed
            if world.use_directions:
                agent.direction = np.random.uniform(0, 2 * np.pi, 1)
                agent.direction = np.mod(agent.direction, 2 * np.pi)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-3.6, +3.6, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        dists = []
        for a in world.agents:
            if a is agent:
                continue
            dists.append(np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))))
        rew = min(dists)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        other_pos = np.zeros((2, world.num_agents))
        i = 0
        j = 0
        for other in world.agents:
            if other is agent:
                continue
            if np.linalg.norm(other.state.p_pos - agent.state.p_pos) <= 3:
                distance = other.state.p_pos - agent.state.p_pos
                if agent.direction is not None:
                    new_distance = np.zeros(2)
                    new_distance[0] = cos(agent.direction) * distance[0] - sin(agent.direction) * distance[1]
                    new_distance[1] = sin(agent.direction) * distance[0] + cos(agent.direction) * distance[1]
                    distance = new_distance
                coef = world.grid_resolution/(world.limit*4)
                scale = (world.grid_resolution//2) - 1
                other_pos[0][i] = round(coef*distance[0]) + scale
                other_pos[1][i] = round(coef*distance[1]) + scale
                i += 1
            else:
                j += 1
        if j > 0:
            other_pos = other_pos[:, :-j]
        observations = np.empty([2], dtype=object)
        if agent.action.u is None:
            observations[:] = [np.zeros(2), other_pos]
        else:
            observations[:] = [agent.action.u, other_pos]
        return observations
    
    def critic_observation(self, world):
        # For velocities, need to know in which liste the indices of the grid are
        agents_vel_x = np.zeros((world.num_agents + 1))
        agents_vel_x[0] = 2
        agents_vel_y = np.zeros((world.num_agents + 1))
        agents_vel_y[0] = 2
        other_pos = np.zeros((2, world.num_agents))
        i = 0
        for other in world.agents:
            agents_vel_x[i + 1] = other.state.p_vel[0]
            agents_vel_y[i + 1] = other.state.p_vel[1]
            distance = other.state.p_pos
            coef = int(ceil(world.grid_resolution/2)/(world.limit*2))
            scale = int((ceil(world.grid_resolution/2)//2)) - 1
            other_pos[0][i] = round(coef*distance[0]) + scale
            other_pos[1][i] = round(coef*distance[1]) + scale
            i += 1
        observations = np.empty([3], dtype=object)
        observations[:] = [agents_vel_x, agents_vel_y, other_pos]
        return observations
    
    
