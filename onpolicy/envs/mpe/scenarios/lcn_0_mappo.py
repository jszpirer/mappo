import numpy as np
from onpolicy.envs.mpe.core import World, Agent
from onpolicy.envs.mpe.scenario import BaseScenario
from collections import defaultdict
import math, random

class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.limit = 4
        num_agents = args.num_agents
        num_landmarks = args.num_landmarks
        world.collaborative = True
        world.one_reward = True
        # set scenario properties
        self.max_comm_dist = 1.5
        self.sensing_range = 1.07
        self.arena_center = [0, 0]
        self.number_points = 100000
        self.distribution_radius = 3.85
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.u_noise = 1
            agent.max_speed = 0.51
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-3.85, +3.85, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def benchmark_data(self, agent, world):
        # Not using it?
        rew = 0
        collisions = 0
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

    def distance(self, a, b):
        return np.sqrt(np.sum(np.square(a.state.p_pos - b.state.p_pos)))

    def add_neighs(self, agents, agent):
        for other_agent in agents:
            if other_agent.cluster_id == 0 and self.distance(agent, other_agent) <= self.max_comm_dist:
                other_agent.cluster_id = agent.cluster_id
                self.add_neighs(agents, other_agent)


    def reward(self, agent, world):
        collide = 0
        max_used_id = 0
        for agent in world.agents:
            if agent.cluster_id != 0:
                continue
            max_used_id += 1
            agent.cluster_id = max_used_id
            self.add_neighs(world.agents, agent)

        group_sizes = defaultdict(int)
        for agent in world.agents:
            group_sizes[agent.cluster_id] += 1

        biggest_group_id = max(group_sizes, key=group_sizes.get)
        biggest_group = [agent for agent in world.agents if agent.cluster_id == biggest_group_id]

        min_x = min(agent.state.p_pos[0] for agent in biggest_group)
        min_y = min(agent.state.p_pos[1] for agent in biggest_group)
        max_x = max(agent.state.p_pos[0] for agent in biggest_group)
        max_y = max(agent.state.p_pos[1] for agent in biggest_group)

        width = [min_x - self.sensing_range, max_x + self.sensing_range]
        height = [min_y - self.sensing_range, max_y + self.sensing_range]

        width_span = width[1] - width[0]
        height_span = height[1] - height[0]

        avg = 0
        for _ in range(1000):
            rx = np.random.uniform(width[0], width[1])
            ry = np.random.uniform(height[0], height[1])
            rnd_point = [rx, ry]

            for pos in biggest_group:
                dist = np.sqrt(np.sum(np.square(rnd_point - pos.state.p_pos)))
                dist2 = np.linalg.norm(np.array(rnd_point) - np.array(self.arena_center))
                if dist <= self.sensing_range and dist2 < self.distribution_radius:
                    avg += 1
                    break
        
        # avg /= np.ceil(width_span * height_span * self.number_points)
        avg /= np.ceil(1000)
        performance = width_span * height_span * avg

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and a is not agent:
                    # rew -= 1
                    collide += 1

        return performance

    def observation(self, agent, world):
        # Definition of the range of detection for the agents
        detection_range = 3
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        indices = np.array([item for item in range(0, len(world.landmarks))])
        duplicate = False
        duplicates = []
        for i, entity in enumerate(world.landmarks):  # world.entities:
            if math.dist(entity.state.p_pos, agent.state.p_pos) < detection_range and random.uniform(0,1) > 0.85:
                index = np.argwhere(indices==i)
                indices = np.delete(indices, index)
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                duplicates.append(entity.state.p_pos - agent.state.p_pos)
                duplicate = True
            else:
                entity_pos.append(np.array([0, 0]))
        if len(indices) > 0:
            for index in indices:
                if duplicate:
                    entity_pos[index] = random.choice(duplicates)
                else:
                    x = random.uniform(max(-world.limit, agent.state.p_pos[0] - detection_range), min(world.limit, agent.state.p_pos[0] + detection_range))
                    y = random.uniform(max(-world.limit, agent.state.p_pos[1] - detection_range), min(world.limit, agent.state.p_pos[1] + detection_range))
                    entity_pos[index] = agent.state.p_pos - np.array([x, y])  

        other_pos = []
        duplicate = False
        duplicates = []
        indices = []
        i = 0
        for other in world.agents:
            if other is agent: continue
            if math.dist(other.state.p_pos, agent.state.p_pos) < detection_range and random.uniform(0,1) > 0.85:
                noise = np.random.normal(0, 1, other.state.p_pos.shape)
                position = other.state.p_pos + noise
                other_pos.append(position - agent.state.p_pos)
                duplicates.append(position - agent.state.p_pos)
                duplicate = True
            else:
                indices.append(i)
                other_pos.append(np.array([0, 0]))
            i += 1
        if len(indices) > 0:
            for index in indices:
                if duplicate:
                    other_pos[index] = random.choice(duplicates)
                else:
                    x = random.uniform(max(-world.limit, agent.state.p_pos[0] - detection_range), min(world.limit, agent.state.p_pos[0] + detection_range))
                    y = random.uniform(max(-world.limit, agent.state.p_pos[1] - detection_range), min(world.limit, agent.state.p_pos[1] + detection_range)) 
                    other_pos[index] = agent.state.p_pos - np.array([x, y])                  

        return np.concatenate([agent.state.p_vel] + entity_pos + other_pos)
