import numpy as np
from math import ceil, cos, sin, pi
from onpolicy.envs.mpe.core import World, Agent, Landmark, Wall
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.limit = 4.33
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
        # Walls
        side_length = 2.2414
        self.num_points = int(side_length/((4 * world.limit)/world.grid_resolution)) + 1
        n_sides = 12
 
        # Rayon du cercle circonscrit
        radius = side_length / (2 * np.sin(np.pi / n_sides))
 
        # Génération des sommets
        vertices = []
        for i in range(n_sides):
            angle = 2 * np.pi * i / n_sides
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append((x, y)) 
 
        # Création des murs
        for i in range(n_sides):
            start = vertices[i]
            end = vertices[(i + 1) % n_sides]
            wall = Wall(startpoint=start, endpoint=end, width=0.1, hard=True)
            world.walls.append(wall)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()

        # set random initial states
        for agent in world.agents:
            # The agents should be initialized inside the dodecagone
            r  = 4.0325 * np.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2 * np.pi)
            agent.state.p_pos = np.array([r * np.cos(theta), r * np.sin(theta)])
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

    def is_onpatch(self, agent, world):
        # Checks if the agent is on one of the patches of the aren
        for patch in world.landmarks:
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - patch.state.p_pos)))
            if dist <= patch.size:
                return True
        return False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        dists = []
        for a in world.agents:
            if a is agent:
                continue
            if self.is_onpatch(a, world):
                rew -= 1
                continue
            dists.append(np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))))
        rew = min(dists)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew
    
    def discretize_wall(self, wall, grid_resolution):
        start, end = wall.start, wall.end
        length = np.linalg.norm(end - start)
        num_points = int(length / grid_resolution)
        points = [start + (end - start) * i / num_points for i in range(num_points + 1)]
        return points

    def observation(self, agent, world):
        cam_fov = np.deg2rad(130)
        cam_min, cam_max = 0.393, 5.89
        lidar_min, lidar_max = 0.118, 2.77

        grid_res = world.grid_resolution
        scale = (grid_res // 2) - 1
        coef = grid_res / (world.limit * 4)

        agent_pos = agent.state.p_pos
        agent_dir = np.array([sin(agent.direction), cos(agent.direction)])

        camera_coords = [1]
        lidar_coords = [1]
        num_points = self.num_points * 12
        all_pos = np.zeros((2, world.num_agents + num_points))

        i = 0
        j = 0
        for other in world.agents:
            camera = False
            lidar = False
            if other is agent:
                j += 1
                continue

            rel_pos = other.state.p_pos - agent_pos
            dist = np.linalg.norm(rel_pos)
            rel_dir = rel_pos / (dist + 1e-6)
            angle_to_other = np.arccos(np.clip(np.dot(agent_dir, rel_dir), -1, 1))

            # Verfification if the agent can see the other or not
            occluded = False
            for blocker in world.agents:
                if blocker is agent or blocker is other:
                    continue
                blocker_vec = blocker.state.p_pos - agent_pos
                if np.linalg.norm(blocker_vec) < dist and np.dot(blocker_vec, rel_pos) > 0:
                    if np.linalg.norm(np.cross(rel_pos, blocker_vec)) / dist < blocker.size:
                        occluded = True
                        break

            if occluded:
                j += 1
                continue

            grid_x = int(round(coef * rel_pos[0]) + scale)
            grid_y = int(round(coef * rel_pos[1]) + scale)
 
            if lidar_min <= dist <= lidar_max:
                lidar = True
 
            if cam_min <= dist <= cam_max and angle_to_other <= cam_fov / 2:
                camera = True

            if camera or lidar:
                all_pos[0][i] = grid_x
                all_pos[1][i] = grid_y
                if camera:
                    camera_coords.append(1)
                else:
                    camera_coords.append(0)
                if lidar :
                    lidar_coords.append(1)
                else:
                    lidar_coords.append(0)
                i+= 1
            else:
                j += 1

        for wall in world.walls:
            wall_points = self.discretize_wall(wall, 1 / coef)  # résolution adaptée à l'échelle
            for point in wall_points:
                rel_pos = point - agent_pos
                dist = np.linalg.norm(rel_pos)
                if not (lidar_min <= dist <= 7.86):
                    j += 1
                    continue

                # Vérification d'occlusion par les agents
                occluded = False
                for blocker in world.agents:
                    if blocker is agent:
                        continue
                    blocker_vec = blocker.state.p_pos - agent_pos
                    if np.linalg.norm(blocker_vec) < dist and np.dot(blocker_vec, rel_pos) > 0:
                        if np.linalg.norm(np.cross(rel_pos, blocker_vec)) / dist < blocker.size:
                            occluded = True
                            break

                if occluded:
                    j += 1
                    continue

                grid_x = int(round(coef * rel_pos[0]) + scale)
                grid_y = int(round(coef * rel_pos[1]) + scale)

                all_pos[0][i] = grid_x
                all_pos[1][i] = grid_y
                i += 1
                camera_coords.append(0)
                lidar_coords.append(1)
                
        if j > 0:
            all_pos = all_pos[:, :-j]

        
 
        camera_array = np.array(camera_coords, dtype=int)
        lidar_array = np.array(lidar_coords, dtype=int)


        landmarks_array = []
        # Array for the landmarks (for now I only have one channel for now)
        for patch in world.landmarks:
            distance = entity.state.p_pos - agent.state.p_pos
        
        for i, entity in enumerate(world.landmarks):  # world.entities:
            distance = entity.state.p_pos - agent.state.p_pos
            coef = world.grid_resolution/(world.limit*4)
            scale = (world.grid_resolution//2) - 1
            entity_pos[0][i] = round(coef*distance[0]) + scale
            entity_pos[1][i] = round(coef*distance[1]) + scale
        
        observations = np.empty([4], dtype=object)
        observations[:] = [agent.state.p_vel, all_pos, camera_array, lidar_array]
        return observations
    
    
    def critic_observation(self, world):
        # Critic's observations are the same not matter which robot is used
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
    
    
