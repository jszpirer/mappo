import numpy as np
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

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisiononpolicy/envs/mpe/scenarios/simple_coverage_cnn_sparse_localj.py(np.square(delta_pos)))
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

        other_pos = np.zeros((2, world.num_agents + 3 * 2)) # Adding the possibility that a robot can be against a wall
        i = 0
        j = 0
        for other in world.agents:
            if other is agent:
                continue
            if np.linalg.norm(other.state.p_pos - agent.state.p_pos) <= 3:
                distance = other.state.p_pos - agent.state.p_pos
                coef = world.grid_resolution/(world.limit*4)
                scale = (world.grid_resolution//2) - 1
                other_pos[0][i] = round(coef*distance[0]) + scale
                other_pos[1][i] = round(coef*distance[1]) + scale
                i += 1
            else:
                j += 1
        
        if agent.state.p_pos[0] >= 3.56:
            other_pos[:, i:i+3] = [[39, 39, 39], [37, 38, 39]]
            i += 3
        elif agent.state.p_pos[0] <= -3.56:
            other_pos[:, i:i+3] = [[37, 37, 37], [37, 38, 39]]
            i += 3
        else:
            j += 3
        if agent.state.p_pos[1] >= 3.56:
            other_pos[:, i:i+3] = [[37, 38, 39], [37, 37, 37]]
            i += 3
        elif agent.state.p_pos[1] <= -3.56:
            other_pos[:, i:i+3] = [[37, 38, 39], [39, 39, 39]]
            i += 3
        else:
            j += 3
        
        if j > 0:
            other_pos = other_pos[:, :-j]
        observations = np.empty([2], dtype=object)
        observations[:] = [agent.state.p_vel, other_pos]
        return observations
    
    
