import numpy as np
from math import ceil
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
        world.nb_additional_data = args.nb_additional_data
        world.omniscient_critic = args.omniscient_critic
        world.use_directions = args.use_directions
        world.sensivity = 5.0
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            # agent.u_noise = 1
            agent.max_speed = 0.51
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
            if world.use_directions:
                agent.direction = np.random.uniform(0, 2 * np.pi, 1)
                agent.direction = np.mod(agent.direction, 2 * np.pi)
                agent.direction_init = agent.direction

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
        rew = -max(dists)
        return rew

    def observation(self, agent, world):
        agent_pos = agent.state.p_pos
        agent_dir = np.array([np.sin(agent.direction), np.cos(agent.direction)])
        other_pos = np.zeros((2, world.num_agents))
        i = 0
        j = 0
        for other in world.agents:
            if other is agent:
                continue
            if np.linalg.norm(other.state.p_pos - agent.state.p_pos) <= 2.14:
                if np.random.binomial(n=1, p=0.85) == 0:
                    distance = other.state.p_pos - agent.state.p_pos
                    if world.use_directions:
                        old_distance = distance
                        distance[0] = np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1]
                        distance[1] = np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1]
                    noise = np.random.normal(0, 0.0644, size=distance.shape)
                    distance = distance + noise
                    coef = world.grid_resolution/(world.limit*4)
                    scale = (world.grid_resolution//2) - 1
                    other_pos[0][i] = round(coef*distance[0]) + scale
                    other_pos[1][i] = round(coef*distance[1]) + scale
                    i += 1
                else:
                    j += 1
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
        # Critic's observations are the same not matter which robot is used
        other_pos = np.zeros((2, world.num_agents))
        i = 0
        for other in world.agents:
            distance = other.state.p_pos
            coef = int(ceil(world.grid_resolution/2)/(world.limit*2))
            scale = int((ceil(world.grid_resolution/2)//2)) - 1
            other_pos[0][i] = round(coef*distance[0]) + scale
            other_pos[1][i] = round(coef*distance[1]) + scale
            i += 1
        observations = np.empty([1], dtype=object)
        observations[:] = [other_pos]
        return observations

