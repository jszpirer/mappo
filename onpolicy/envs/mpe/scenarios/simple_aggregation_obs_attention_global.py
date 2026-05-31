import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark, Wall, Entity
from onpolicy.envs.mpe.scenario import BaseScenario
import random


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.limit = 3.75
        world.num_agents = args.num_agents
        world.num_obstacles = args.num_obstacles
        print("Num obstacles")
        print(world.num_obstacles)
        world.collaborative = True
        world.grid_resolution = args.grid_resolution
        world.nb_additional_data = args.nb_additional_data
        world.omniscient_critic = args.omniscient_critic
        world.use_directions = args.use_directions
        world.sensivity = 5.0
        self.velocities_critic = args.velocities_critic
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            # agent.u_noise = 1
            agent.max_speed = 0.51
        # add walls
        vertices = [(-4, 4), (4, 4), (4, -4), (-4, -4)]
        for i in range(len(vertices)):
            if i == len(vertices) - 1:
                end = 0
            else:
                end = i + 1
            wall = Wall(startpoint=vertices[i], endpoint=vertices[end], width=0.1, hard=True)
            world.walls.append(wall)
        # add obstacles
        self.possible_positions = [(0, 0), (0, 2), (0, -2), (2, 0), (2, 2), (2, -2), (-2, 0), (-2, 2), (-2, -2)]
        world.obstacles = [Entity() for i in range(world.num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = 'obstacle %d' % i
            obstacle.size = 0.5
            obstacle.color = (0.75, 0.25, 0.25)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()
        # set random initial states
        indices = random.sample(range(len(self.possible_positions)), world.num_obstacles)
        for i, obstacle in enumerate(world.obstacles):
            obstacle.state.p_pos = np.array(self.possible_positions[indices[i]])
        # Need more than one try per agent because it can interfere with the obstacles
        for agent in world.agents:
            for _ in range(1000):
                pos = np.random.uniform(-3.6, +3.6, world.dim_p)
                valid = True
                for obstacle in world.obstacles:
                    obs_pos = obstacle.state.p_pos
                    if np.linalg.norm(pos - obs_pos) <= 0.65:
                        valid = False
                        break
                if valid:
                    continue
            agent.state.p_pos = pos
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
        other_pos = np.zeros((world.num_agents - 1, 2))
        i = 0
        j = 0
        for other in world.agents:
            if other is agent:
                continue
            distance = np.linalg.norm(other.state.p_pos - agent.state.p_pos)
            if distance <= 15:
                dist = other.state.p_pos - agent.state.p_pos
                if world.use_directions:
                    old_distance = dist
                    dist[0] = np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1]
                    dist[1] = np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1]
                bearing = np.arctan2(dist[0], dist[1])
                other_pos[i][0] = distance
                other_pos[i][1] = bearing
                i += 1
            else:
                j += 1
        if j > 0:
            other_pos = other_pos[:-j, :]
        obs_pos = np.zeros((world.num_obstacles + 4, 2))
        i = 0
        j = 0
        for obs in world.obstacles:
            distance = np.linalg.norm(obs.state.p_pos - agent.state.p_pos)
            if distance <= 15:
                dist = obs.state.p_pos - agent.state.p_pos
                if world.use_directions:
                    old_distance = dist
                    dist[0] = np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1]
                    dist[1] = np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1]
                bearing = np.arctan2(dist[0], dist[1])
                obs_pos[i][0] = distance
                obs_pos[i][1] = bearing
                i += 1
            else:
                j += 1
        if j > 0:
            obs_pos = obs_pos[:-j, :]
        distance = 4 - agent.state.p_pos[0]
        dist = [4 - agent.state.p_pos[0], 0]
        if world.use_directions:
            old_distance = dist
            dist[0] = np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1]
            dist[1] = np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1]
        bearing = np.arctan2(dist[0], dist[1])
        obs_pos[i][0] = distance
        obs_pos[i][1] = bearing
        i += 1
        distance = 4 + agent.state.p_pos[0]
        dist = [-4 - agent.state.p_pos[0], 0]
        if world.use_directions:
            old_distance = dist
            dist[0] = np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1]
            dist[1] = np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1]
        bearing = np.arctan2(dist[0], dist[1])
        obs_pos[i][0] = distance
        obs_pos[i][1] = bearing
        i += 1
        distance = 4 - agent.state.p_pos[1]
        dist = [0, 4 - agent.state.p_pos[1]]
        if world.use_directions:
            old_distance = dist
            dist[0] = np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1]
            dist[1] = np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1]
        bearing = np.arctan2(dist[0], dist[1])
        obs_pos[i][0] = distance
        obs_pos[i][1] = bearing
        i += 1
        distance = 4 + agent.state.p_pos[1]
        dist = [0, -4 - agent.state.p_pos[1]]
        if world.use_directions:
            old_distance = dist
            dist[0] = np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1]
            dist[1] = np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1]
        bearing = np.arctan2(dist[0], dist[1])
        obs_pos[i][0] = distance
        obs_pos[i][1] = bearing
        i += 1
        observations = np.empty([4], dtype=object)
        observations[:] = [agent.state.p_vel, agent.state.p_pos, other_pos, obs_pos]
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
        obs_pos = np.zeros((world.num_obstacles, 2))
        for i, a in enumerate(world.obstacles):
            obs_pos[i][0] = a.state.p_pos[0]
            obs_pos[i][1] = a.state.p_pos[1]
        observations = np.empty([2], dtype=object)
        observations[:] = [agents, obs_pos]
        return observations

