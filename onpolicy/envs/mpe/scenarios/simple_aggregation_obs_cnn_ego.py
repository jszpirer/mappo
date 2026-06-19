import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark, Wall, Entity
from onpolicy.envs.mpe.scenario import BaseScenario
import random
from math import sin, cos, sqrt, ceil

proximity_sensors = [-2.6179, -1.5708, -0.785398, -0.261799, 0.261799, 0.785398, 1.5708, 2.6179]
nb_prox = 8

class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.limit = 3.75
        world.num_agents = args.num_agents
        world.num_obstacles = args.num_obstacles
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
    
    def get_closest_indice(self, bearing):
        return min(range(len(proximity_sensors)), key=lambda i: abs(proximity_sensors[i] - bearing))

    def intersection_droite_cercle(self, x0, y0, dx, dy, xc, yc, r):
        # Coefficients du polynôme
        a = dx**2 + dy**2
        b = 2 * ((x0 - xc)*dx + (y0 - yc)*dy)
        c = (x0 - xc)**2 + (y0 - yc)**2 - r**2

        delta = b**2 - 4*a*c

        if delta < 0:
            return []  # pas d'intersection

        elif delta == 0:
            t = -b / (2*a)
            x = x0 + t * dx
            y = y0 + t * dy
            return [[x, y]]

        else:
            sqrt_delta = sqrt(delta)

            t1 = (-b - sqrt_delta) / (2*a)
            t2 = (-b + sqrt_delta) / (2*a)

            p1 = [x0 + t1 * dx, y0 + t1 * dy]
            p2 = [x0 + t2 * dx, y0 + t2 * dy]

            return [p1, p2]
    def intersection_droites(self, x1, y1, dx1, dy1, x2, y2, dx2, dy2):
        D = dx1 * dy2 - dy1 * dx2

        if D == 0:
            return None  # parallèles ou confondues

        t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / D

        x = x1 + t * dx1
        y = y1 + t * dy1

        return [x, y]

    def observation(self, agent, world):
        coef = int(ceil(world.grid_resolution/2)/(world.limit*2))
        scale = int((ceil(world.grid_resolution/2)//2)) - 1
        obs_pos = []
        other_pos = np.zeros((2, world.num_agents - 1))
        i = 0
        j = 0
        for other in world.agents:
            if other is agent:
                continue
            distance = np.linalg.norm(other.state.p_pos - agent.state.p_pos)
            if distance <= 3:
                dist = other.state.p_pos - agent.state.p_pos
                if world.use_directions:
                    old_distance = dist
                    dist[0] = (np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1])[0]
                    dist[1] = (np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1])[0]
                bearing = np.arctan2(dist[0], dist[1])
                other_pos[0][i] = round(coef*dist[0]) + scale
                other_pos[1][i] = round(coef*dist[1]) + scale
                if distance <= 0.47:
                    prox = self.get_closest_indice(bearing)
                    for k in range(0, nb_prox):
                        plus = (prox + k) % nb_prox
                        new_bearing = proximity_sensors[plus]
                        inter = self.intersection_droite_cercle(0, 0, sin(new_bearing), cos(new_bearing), dist[0], dist[1], 0.15)
                        if len(inter) == 0:
                            break
                        elif len(inter) == 1:
                            distance = sqrt(pow(inter[0][0], 2) + pow(inter[0][1], 2))
                            if distance <= 0.32:
                                new_point = [round(coef*(inter[0][0] - sin(new_bearing) * 0.15)) + scale, round(coef*(inter[0][1] - cos(new_bearing) * 0.15)) + scale]
                                if new_point not in obs_pos:
                                    obs_pos.append(new_point)
                            else:
                                break
                        else:
                            distance_1 = sqrt(pow(inter[0][0], 2) + pow(inter[0][1], 2))
                            distance_2 = sqrt(pow(inter[1][0], 2) + pow(inter[1][1], 2))
                            if distance_1 <= 0.32 and distance_2 >= distance_1:
                                new_point = [round(coef*(inter[0][0] - sin(new_bearing) * 0.15)) + scale, round(coef*(inter[0][1] - cos(new_bearing) * 0.15)) + scale]
                                if new_point not in obs_pos:
                                    obs_pos.append(new_point)
                            elif distance <= 0.32:
                                new_point = [round(coef*(inter[1][0] - sin(new_bearing) * 0.15)) + scale, round(coef*(inter[1][1] - cos(new_bearing) * 0.15)) + scale]
                                if new_point not in obs_pos:
                                    obs_pos.append(new_point)
                            else:
                                break
                    for k in range(1, nb_prox):
                        plus = (prox - k) % nb_prox
                        new_bearing = proximity_sensors[plus]
                        inter = self.intersection_droite_cercle(0, 0, sin(new_bearing), cos(new_bearing), dist[0], dist[1], 0.15)
                        if len(inter) == 0:
                            break
                        elif len(inter) == 1:
                            distance = sqrt(pow(inter[0][0], 2) + pow(inter[0][1], 2))
                            if distance <= 0.32:
                                new_point = [round(coef*(inter[0][0] - sin(new_bearing) * 0.15)) + scale, round(coef*(inter[0][1] - cos(new_bearing) * 0.15)) + scale]
                                if new_point not in obs_pos:
                                    obs_pos.append(new_point)
                            else:
                                break
                        else:
                            distance_1 = sqrt(pow(inter[0][0], 2) + pow(inter[0][1], 2))
                            distance_2 = sqrt(pow(inter[1][0], 2) + pow(inter[1][1], 2))
                            if distance_1 <= 0.32 and distance_2 >= distance_1:
                                new_point = [round(coef*(inter[0][0] - sin(new_bearing) * 0.15)) + scale, round(coef*(inter[0][1] - cos(new_bearing) * 0.15)) + scale]
                                if new_point not in obs_pos:
                                    obs_pos.append(new_point)
                            elif distance <= 0.32:
                                new_point = [round(coef*(inter[1][0] - sin(new_bearing) * 0.15)) + scale, round(coef*(inter[1][1] - cos(new_bearing) * 0.15)) + scale]
                                if new_point not in obs_pos:
                                    obs_pos.append(new_point)
                            else:
                                break
                i += 1
            else:
                j += 1
        if j > 0:
            other_pos = other_pos[:, :-j]
        for obs in world.obstacles:
            distance = np.linalg.norm(obs.state.p_pos - agent.state.p_pos)
            if distance <= 0.82:
                dist = obs.state.p_pos - agent.state.p_pos
                if world.use_directions:
                    old_distance = dist
                    dist[0] = (np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1])[0]
                    dist[1] = (np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1])[0]
                bearing = np.arctan2(dist[0], dist[1])
                prox = self.get_closest_indice(bearing)
                for k in range(0, nb_prox):
                    plus = (prox + k) % nb_prox
                    new_bearing = proximity_sensors[plus]
                    inter = self.intersection_droite_cercle(0, 0, sin(new_bearing), cos(new_bearing), dist[0], dist[1], 0.5)
                    if len(inter) == 0:
                        break
                    elif len(inter) == 1:
                        distance = sqrt(pow(inter[0][0], 2) + pow(inter[0][1], 2))
                        if distance <= 0.32:
                            new_point = [round(coef*(inter[0][0] - sin(new_bearing) * 0.15)) + scale, round(coef*(inter[0][1] - cos(new_bearing) * 0.15)) + scale]
                            if new_point not in obs_pos:
                                obs_pos.append(new_point)
                        else:
                            break
                    else:
                        distance_1 = sqrt(pow(inter[0][0], 2) + pow(inter[0][1], 2))
                        distance_2 = sqrt(pow(inter[1][0], 2) + pow(inter[1][1], 2))
                        if distance_1 <= 0.32 and distance_2 >= distance_1:
                            new_point = [round(coef*(inter[0][0] - sin(new_bearing) * 0.15)) + scale, round(coef*(inter[0][1] - cos(new_bearing) * 0.15)) + scale]
                            if new_point not in obs_pos:
                                obs_pos.append(new_point)
                        elif distance <= 0.32:
                            new_point = [round(coef*(inter[1][0] - sin(new_bearing) * 0.15)) + scale, round(coef*(inter[1][1] - cos(new_bearing) * 0.15)) + scale]
                            if new_point not in obs_pos:
                                obs_pos.append(new_point)
                        else:
                            break
                for k in range(1, nb_prox):
                    plus = (prox - k) % nb_prox
                    new_bearing = proximity_sensors[plus]
                    inter = self.intersection_droite_cercle(0, 0, sin(new_bearing), cos(new_bearing), dist[0], dist[1], 0.5)
                    if len(inter) == 0:
                        break
                    elif len(inter) == 1:
                        distance = sqrt(pow(inter[0][0], 2) + pow(inter[0][1], 2))
                        if distance <= 0.32:
                            new_point = [round(coef*(inter[0][0] - sin(new_bearing) * 0.15)) + scale, round(coef*(inter[0][1] - cos(new_bearing) * 0.15)) + scale]
                            if new_point not in obs_pos:
                                obs_pos.append(new_point)
                        else:
                            break
                    else:
                        distance_1 = sqrt(pow(inter[0][0], 2) + pow(inter[0][1], 2))
                        distance_2 = sqrt(pow(inter[1][0], 2) + pow(inter[1][1], 2))
                        if distance_1 <= 0.32 and distance_2 >= distance_1:
                            new_point = [round(coef*(inter[0][0] - sin(new_bearing) * 0.15)) + scale, round(coef*(inter[0][1] - cos(new_bearing) * 0.15)) + scale]
                            if new_point not in obs_pos:
                                obs_pos.append(new_point)
                        elif distance <= 0.32:
                            new_point = [round(coef*(inter[1][0] - sin(new_bearing) * 0.15)) + scale, round(coef*(inter[1][1] - cos(new_bearing) * 0.15)) + scale]
                            if new_point not in obs_pos:
                                obs_pos.append(new_point)
                        else:
                            break
        if agent.state.p_pos[0] >= 3.63:
            distance = 4 - agent.state.p_pos[0]
            dist = [4 - agent.state.p_pos[0], 0]
            if world.use_directions:
                old_distance = dist
                dist[0] = (np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1])[0]
                dist[1] = (np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1])[0]
            bearing = np.arctan2(dist[0], dist[1])
            prox = self.get_closest_indice(bearing)
            for k in range(0, nb_prox):
                plus = (prox + k) % nb_prox
                new_bearing = proximity_sensors[plus]
                inter = self.intersection_droites(0, 0, sin(new_bearing), cos(new_bearing), 4, 0, 4, 1)
                if inter is None:
                    break
                else:
                    distance = sqrt(pow(inter[0], 2) + pow(inter[1], 2))
                    if distance <= 0.37:
                        a_prime = [sin(new_bearing) * 0.15, cos(new_bearing) * 0.15]
                        b_prime = [inter[0] - sin(new_bearing) * 0.05, inter[1] - cos(new_bearing) * 0.05]
                        new_point = [round(coef*(b_prime[0] - a_prime[0]) + scale), round(coef*(b_prime[1] - a_prime[1]) + scale)]
                        if new_point not in obs_pos:
                            obs_pos.append(new_point)
                    else:
                        break
            for k in range(1, nb_prox):
                plus = (prox - k) % nb_prox
                new_bearing = proximity_sensors[plus]
                inter = self.intersection_droites(0, 0, sin(new_bearing), cos(new_bearing), 4, 0, 4, 1)
                if inter is None:
                    break
                else:
                    distance = sqrt(pow(inter[0], 2) + pow(inter[1], 2))
                    if distance <= 0.37:
                        a_prime = [sin(new_bearing) * 0.15, cos(new_bearing) * 0.15]
                        b_prime = [inter[0] - sin(new_bearing) * 0.05, inter[1] - cos(new_bearing) * 0.05]
                        new_point = [round(coef*(b_prime[0] - a_prime[0]) + scale), round(coef*(b_prime[1] - a_prime[1]) + scale)]
                        if new_point not in obs_pos:
                            obs_pos.append(new_point)
                    else:
                        break
        elif agent.state.p_pos[0] <= -3.63:
            distance = 4 + agent.state.p_pos[0]
            dist = [-4 - agent.state.p_pos[0], 0]
            if world.use_directions:
                old_distance = dist
                dist[0] = (np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1])[0]
                dist[1] = (np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1])[0]
            bearing = np.arctan2(dist[0], dist[1])
            prox = self.get_closest_indice(bearing)
            for k in range(0, nb_prox):
                plus = (prox + k) % nb_prox
                new_bearing = proximity_sensors[plus]
                inter = self.intersection_droites(0, 0, sin(new_bearing), cos(new_bearing), -4, 0, -4, 1)
                if inter is None:
                    break
                else:
                    distance = sqrt(pow(inter[0], 2) + pow(inter[1], 2))
                    if distance <= 0.37:
                        a_prime = [sin(new_bearing) * 0.15, cos(new_bearing) * 0.15]
                        b_prime = [inter[0] - sin(new_bearing) * 0.05, inter[1] - cos(new_bearing) * 0.05]
                        new_point = [round(coef*(b_prime[0] - a_prime[0]) + scale), round(coef*(b_prime[1] - a_prime[1]) + scale)]
                        if new_point not in obs_pos:
                            obs_pos.append(new_point)
                    else:
                        break
            for k in range(1, nb_prox):
                plus = (prox - k) % nb_prox
                new_bearing = proximity_sensors[plus]
                inter = self.intersection_droites(0, 0, sin(new_bearing), cos(new_bearing), -4, 0, -4, 1)
                if inter is None:
                    break
                else:
                    distance = sqrt(pow(inter[0], 2) + pow(inter[1], 2))
                    if distance <= 0.37:
                        a_prime = [sin(new_bearing) * 0.15, cos(new_bearing) * 0.15]
                        b_prime = [inter[0] - sin(new_bearing) * 0.05, inter[1] - cos(new_bearing) * 0.05]
                        new_point = [round(coef*(b_prime[0] - a_prime[0]) + scale), round(coef*(b_prime[1] - a_prime[1]) + scale)]
                        if new_point not in obs_pos:
                            obs_pos.append(new_point)
                    else:
                        break
        if agent.state.p_pos[1] >= 3.63:
            distance = 4 - agent.state.p_pos[1]
            dist = [0, 4 - agent.state.p_pos[1]]
            if world.use_directions:
                old_distance = dist
                dist[0] = (np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1])[0]
                dist[1] = (np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1])[0]
            bearing = np.arctan2(dist[0], dist[1])
            prox = self.get_closest_indice(bearing)
            for k in range(0, nb_prox):
                plus = (prox + k) % nb_prox
                new_bearing = proximity_sensors[plus]
                inter = self.intersection_droites(0, 0, sin(new_bearing), cos(new_bearing), 0, 4, 1, 4)
                if inter is None:
                    break
                else:
                    distance = sqrt(pow(inter[0], 2) + pow(inter[1], 2))
                    if distance <= 0.37:
                        a_prime = [sin(new_bearing) * 0.15, cos(new_bearing) * 0.15]
                        b_prime = [inter[0] - sin(new_bearing) * 0.05, inter[1] - cos(new_bearing) * 0.05]
                        new_point = [round(coef*(b_prime[0] - a_prime[0]) + scale), round(coef*(b_prime[1] - a_prime[1]) + scale)]
                        if new_point not in obs_pos:
                            obs_pos.append(new_point)
                    else:
                        break
            for k in range(1, nb_prox):
                plus = (prox - k) % nb_prox
                new_bearing = proximity_sensors[plus]
                inter = self.intersection_droites(0, 0, sin(new_bearing), cos(new_bearing), 0, 4, 1, 4)
                if inter is None:
                    break
                else:
                    distance = sqrt(pow(inter[0], 2) + pow(inter[1], 2))
                    if distance <= 0.37:
                        a_prime = [sin(new_bearing) * 0.15, cos(new_bearing) * 0.15]
                        b_prime = [inter[0] - sin(new_bearing) * 0.05, inter[1] - cos(new_bearing) * 0.05]
                        new_point = [round(coef*(b_prime[0] - a_prime[0]) + scale), round(coef*(b_prime[1] - a_prime[1]) + scale)]
                        if new_point not in obs_pos:
                            obs_pos.append(new_point)
                    else:
                        break
        elif agent.state.p_pos[1] <= -3.63:
            distance = 4 + agent.state.p_pos[1]
            dist = [0, -4 - agent.state.p_pos[1]]
            if world.use_directions:
                old_distance = dist
                dist[0] = (np.cos(agent.direction)*old_distance[0] - np.sin(agent.direction)*old_distance[1])[0]
                dist[1] = (np.sin(agent.direction)*old_distance[0] + np.cos(agent.direction)*old_distance[1])[0]
            bearing = np.arctan2(dist[0], dist[1])
            prox = self.get_closest_indice(bearing)
            for k in range(0, nb_prox):
                plus = (prox + k) % nb_prox
                new_bearing = proximity_sensors[plus]
                inter = self.intersection_droites(0, 0, sin(new_bearing), cos(new_bearing), 0, -4, 1, -4)
                if inter is None:
                    break
                else:
                    distance = sqrt(pow(inter[0], 2) + pow(inter[1], 2))
                    if distance <= 0.37:
                        a_prime = [sin(new_bearing) * 0.15, cos(new_bearing) * 0.15]
                        b_prime = [inter[0] - sin(new_bearing) * 0.05, inter[1] - cos(new_bearing) * 0.05]
                        new_point = [round(coef*(b_prime[0] - a_prime[0]) + scale), round(coef*(b_prime[1] - a_prime[1]) + scale)]
                        if new_point not in obs_pos:
                            obs_pos.append(new_point)
                    else:
                        break
            for k in range(1, nb_prox):
                plus = (prox - k) % nb_prox
                new_bearing = proximity_sensors[plus]
                inter = self.intersection_droites(0, 0, sin(new_bearing), cos(new_bearing), 0, -4, 1, -4)
                if inter is None:
                    break
                else:
                    distance = sqrt(pow(inter[0], 2) + pow(inter[1], 2))
                    if distance <= 0.37:
                        a_prime = [sin(new_bearing) * 0.15, cos(new_bearing) * 0.15]
                        b_prime = [inter[0] - sin(new_bearing) * 0.05, inter[1] - cos(new_bearing) * 0.05]
                        new_point = [round(coef*(b_prime[0] - a_prime[0]) + scale), round(coef*(b_prime[1] - a_prime[1]) + scale)]
                        if new_point not in obs_pos:
                            obs_pos.append(new_point)
                    else:
                        break
        if len(obs_pos) == 0:
            obs_pos = np.empty((2,0))
        else:
            obs_pos = np.array(obs_pos).T
        observations = np.empty([3], dtype=object)
        if agent.action.u is None:
            observations[:] = [np.zeros(2), other_pos, obs_pos]
        else:
            observations[:] = [agent.action.u, other_pos, obs_pos]
        return observations
    
    def critic_observation(self, world):
        # Critic's observations are the same not matter which robot is used
        # For velocities, need to know in which liste the indices of the grid are
        coef = int(ceil(world.grid_resolution/2)/(world.limit*2))
        scale = int((ceil(world.grid_resolution/2)//2)) - 1
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
            other_pos[0][i] = round(coef*distance[0]) + scale
            other_pos[1][i] = round(coef*distance[1]) + scale
            i += 1
        obs_pos = np.zeros((2, world.num_obstacles))
        i = 0
        for obs in world.obstacles:
            distance = obs.state.p_pos
            obs_pos[0][i] = round(coef*distance[0]) + scale
            obs_pos[1][i] = round(coef*distance[1]) + scale
            i += 1
        observations = np.empty([4], dtype=object)
        observations[:] = [agents_vel_x, agents_vel_y, other_pos, obs_pos]
        return observations


