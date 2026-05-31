import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario
 
sigma_table = [[0, 0.3570609], [2, 0.3192310], [5, 0.1926492], [10, 0.1529397], [15, 0.1092330], [30, 0.1216533], [45, 0.1531546],
                [60, 0.1418425], [80, 0.1418425]]
 
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
        self.velocities_critic = args.velocities_critic
        self.sigma = args.sigma
        self.loss_probability = args.loss_probability
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
 
    def interpolation(self, x, table):
        # Using sigma an mu to interpolate from distance
        for i in range(len(table) - 1):
            x0 = table[i][0]
            y0 = table[i][1]
            x1 = table[i+1][0]
            y1 = table[i+1][1]
            if x0 <= x and x <= x1:
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return table[-1][1]

 
    def observation(self, agent, world):
            other_pos = np.zeros((world.num_agents - 1, 2))
            i = 0
            j = 0
            for other in world.agents:
                if other is agent:
                    continue
                distance = np.linalg.norm(other.state.p_pos - agent.state.p_pos)
                if distance <= 2.14:
                    if np.random.binomial(n=1, p=self.loss_probability) == 0:
                        dist = other.state.p_pos - agent.state.p_pos
                        # noise = np.random.normal(0, self.sigma, size=dist.shape)
                        # Need to convert to cm first (before adding noise)
                        dcm = distance * (0.035/0.15) * 100
                        # mu = self.interpolation(dcm, mu_table)
                        sigma = self.interpolation(dcm, sigma_table)
                        noise_factor = np.random.lognormal(mean=1, sigma=sigma)
                        d_noisy_m = (dcm / 100) * noise_factor
                        distance = d_noisy_m / (0.035/0.15)
                        direction = dist / distance
                        dist = direction * distance
                        if world.use_directions:
                            old_distance = dist
                            dist[0] = np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1]
                            dist[1] = np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1]
                            bearing = np.arctan2(dist[0], dist[1])
                            other_pos[i][0] = distance
                            other_pos[i][1] = bearing
                        else:
                            other_pos[i][0] = dist[0]
                            other_pos[i][1] = dist[1]
                        i += 1
                    else:
                        j += 1
                else:
                    j += 1
            if j > 0:
                other_pos = other_pos[:-j, :]
            observations = np.empty([2], dtype=object)
            if agent.action.u is None:
                observations[:] = [np.zeros(2), other_pos]
            else:
                observations[:] = [agent.action.u, other_pos]
            return observations

    def critic_observation(self, world):
            if not self.velocities_critic:
                agents_pos = np.zeros((world.num_agents, 2))
                for i, a in enumerate(world.agents):
                    agents_pos[i][0] = a.state.p_pos[0]
                    agents_pos[i][1] = a.state.p_pos[1]
                observations = np.empty([1], dtype=object)
                observations[:] = [agents_pos]
            else:
                agents = np.zeros((world.num_agents, 4))
                for i, a in enumerate(world.agents):
                    agents[i][0] = a.state.p_pos[0]
                    agents[i][1] = a.state.p_pos[1]
                    agents[i][2] = a.state.p_vel[0]
                    agents[i][3] = a.state.p_vel[1]
                observations = np.empty([1], dtype=object)
                observations[:] = [agents]
            return observations