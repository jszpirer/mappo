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
        world.num_landmarks = 1
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
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 1
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

        # set random initial positions for the landmarks
        for landmark in world.landmarks:
            r  = (4.0325 - 1) * np.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2 * np.pi)
            landmark.state.p_pos = np.array([r * np.cos(theta), r * np.sin(theta)])
            landmark.state.p_vel = np.zeros(world.dim_p)

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
        for a in world.agents:
            if self.is_onpatch(a, world):
                rew += 1
        return rew

    def discretize_wall(self, wall, grid_resolution):
        start, end = wall.start, wall.end
        length = np.linalg.norm(end - start)
        num_points = int(length / grid_resolution)
        points = [start + (end - start) * i / num_points for i in range(num_points + 1)]
        return points


    def _prepare_blockers(self, agent, world, max_target_dist2):
        blockers_pos = []
        blockers_size = []
        blockers_objs = []
        for b in world.agents:
            if b is agent:
                continue
            blockers_pos.append(b.state.p_pos)
            blockers_size.append(b.size)
            blockers_objs.append(b)

        if len(blockers_pos) == 0:
            bvec = np.empty((0, 2), dtype=np.float64)
            bdist2 = np.empty((0,), dtype=np.float64)
            bsize = np.empty((0,), dtype=np.float64)
            blocker_index_map = {}
            return bvec, bdist2, bsize, blocker_index_map

        bpos = np.asarray(blockers_pos, dtype=np.float64)
        bsize = np.asarray(blockers_size, dtype=np.float64)

        bvec_all = bpos - agent.state.p_pos
        bdist2_all = np.einsum('ij,ij->i', bvec_all, bvec_all)

        mask = (bdist2_all <= max_target_dist2)

        bvec = bvec_all[mask]
        bdist2 = bdist2_all[mask]
        bsize = bsize[mask]

        orig_indices = np.nonzero(mask)[0]
        blocker_index_map = {id(blockers_objs[i]): j for j, i in enumerate(orig_indices)}

        return bvec, bdist2, bsize, blocker_index_map

    @staticmethod
    def _is_occluded(rel_pos, bvec, bdist2, bsize, skip_index=None):
        if bvec.shape[0] == 0:
            return False

        ax, ay = float(rel_pos[0]), float(rel_pos[1])
        dist2 = ax*ax + ay*ay
        if dist2 == 0.0:
            return False

        cross = ax * bvec[:, 1] - ay * bvec[:, 0]
        near_line = (cross * cross) < (bsize * bsize) * dist2
        closer = bdist2 < dist2
        ahead = (bvec[:, 0] * ax + bvec[:, 1] * ay) > 0.0

        mask = near_line & closer & ahead
        if skip_index is not None and 0 <= skip_index < mask.size:
            mask[skip_index] = False

        return bool(np.any(mask))


    def observation(self, agent, world):
        cam_fov = np.deg2rad(130)
        cam_min2, cam_max2 = 0.393 ** 2, 5.89 ** 2
        lidar_min2, lidar_max2 = 0.118 ** 2, 2.77 ** 2
        walls_lidar_min2, walls_lidar_max2 = 0.118 ** 2, 7.86 ** 2
        cos_fov_half2 = np.cos(cam_fov/2) ** 2

        grid_res = world.grid_resolution
        scale = (grid_res // 2) - 1
        coef = grid_res / (world.limit * 4)

        agent_pos = agent.state.p_pos
        agent_dir = np.array([sin(agent.direction), cos(agent.direction)])

        camera_coords = [1]
        lidar_coords = [1]
        all_pos_x = []
        all_pos_y = []


        max_target_dist2 = max(cam_max2, walls_lidar_max2)
        bvec, bdist2, bsize, blocker_index_map = self._prepare_blockers(agent, world, max_target_dist2)

        for other in world.agents:
            if other is agent:
                continue
            rel_pos = other.state.p_pos - agent_pos
            dist2 = rel_pos[0]*rel_pos[0] + rel_pos[1]*rel_pos[1]
            if dist2 == 0.0:
                continue
            dot_ar = agent_dir[0]*rel_pos[0] + agent_dir[1]*rel_pos[1]
            lidar = (lidar_min2 <= dist2 <= lidar_max2)
            camera = (cam_min2 <= dist2 <= cam_max2) and (dot_ar >= 0.0) and ((dot_ar*dot_ar) >= dist2 * cos_fov_half2)
            if not (camera or lidar):
                continue

            skip_idx = blocker_index_map.get(id(other), None)
            if self._is_occluded(rel_pos, bvec, bdist2, bsize, skip_index=skip_idx):
                continue

            grid_x = int(np.rint(coef * rel_pos[0]) + scale)
            grid_y = int(np.rint(coef * rel_pos[1]) + scale)

            all_pos_x.append(grid_x)
            all_pos_y.append(grid_y)
            camera_coords.append(1 if camera else 0)
            lidar_coords.append(1 if lidar else 0)

        if not hasattr(self, "_wall_points_cache"):
            self._wall_points_cache = {}

        for wall in world.walls:
            wid = id(wall)
            if wid not in self._wall_points_cache:
                self._wall_points_cache[wid] = self.discretize_wall(wall, 1.0 / coef)
            wall_points = self._wall_points_cache[wid]
            for point in wall_points:
                rel_pos = point - agent_pos
                dist2 = rel_pos[0]*rel_pos[0] + rel_pos[1]*rel_pos[1]
                if not (lidar_min2 <= dist2 <= walls_lidar_max2):
                    continue
                if self._is_occluded(rel_pos, bvec, bdist2, bsize):
                    continue
                grid_x = int(np.rint(coef * rel_pos[0]) + scale)
                grid_y = int(np.rint(coef * rel_pos[1]) + scale)

                all_pos_x.append(grid_x)
                all_pos_y.append(grid_y)
                camera_coords.append(0)
                lidar_coords.append(1)
        all_pos = np.array([all_pos_x, all_pos_y], dtype=np.int32)
        camera_array = np.asarray(camera_coords, dtype=np.int32)
        lidar_array  = np.asarray(lidar_coords,  dtype=np.int32)

        landmarks_x = []
        landmarks_y = []

        for landmark in world.landmarks:
            rel_pos = landmark.state.p_pos - agent_pos
            dist2 = np.dot(rel_pos, rel_pos)

            # Vérification d'occlusion par les agents
            occluded = False
            for blocker in world.agents:
                if blocker is agent:
                    continue
                blocker_vec = blocker.state.p_pos - agent_pos
                blocker_dist2 = np.dot(blocker_vec, blocker_vec)
                if blocker_dist2 < dist2 and np.dot(blocker_vec, rel_pos) > 0:
                    c = rel_pos[0] * blocker_vec[1] - rel_pos[1] * blocker_vec[0]
                    if c* c < (blocker.size * blocker.size) * dist2:
                        occluded = True
                        break

            if occluded:
                continue

            # Si visible par la caméra
            if cam_min2 <= dist2 <= cam_max2 :
                dot_ar = agent_dir[0]*rel_pos[0] + agent_dir[1]*rel_pos[1]
                if (dot_ar >= 0.0) and ((dot_ar*dot_ar) >= dist2 * cos_fov_half2) :
                    # Discrétisation dans la grille
                    grid_x = int(round(coef * rel_pos[0]) + scale)
                    grid_y = int(round(coef * rel_pos[1]) + scale)

                    # Marquer la zone du landmark (cercle)
                    radius_cells = int(round(landmark.size * coef))
                    for dx in range(-radius_cells, radius_cells + 1):
                        for dy in range(-radius_cells, radius_cells + 1):
                            if dx**2 + dy**2 <= radius_cells**2:
                                gx, gy = grid_x + dx, grid_y + dy
                                if 0 <= gx < grid_res and 0 <= gy < grid_res:
                                    landmarks_x.append(gx)
                                    landmarks_y.append(gy)

            # Si le robot est dessus (capteur sol)
            if dist2 <= landmark.size * landmark.size:
                grid_x = int(round(coef * rel_pos[0]) + scale)
                grid_y = int(round(coef * rel_pos[1]) + scale)
                landmarks_x.append(grid_x)
                landmarks_y.append(grid_y)

        landmarks = [landmarks_x, landmarks_y]
        landmarks_array = np.array(landmarks)

        observations = np.empty([5], dtype=object)
        observations[:] = [agent.state.p_vel, all_pos, camera_array, lidar_array, landmarks_array]
        return observations


    def critic_observation(self, world):
        # Critic's observations are the same not matter which robot is used
        # For velocities, need to know in which liste the indices of the grid are
        agents_vel_x = np.zeros((world.num_agents + 1))
        agents_vel_x[0] = 2
        agents_vel_y = np.zeros((world.num_agents + 1))
        agents_vel_y[0] = 2
        other_pos = np.zeros((2, world.num_agents))
        landmarks_x = []
        landmarks_y = []
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
        for patch in world.landmarks:
            pos = patch.state.p_pos
            grid_x = int(round(coef * pos[0]) + scale)
            grid_y = int(round(coef * pos[1]) + scale)

            # Marquer la zone du landmark (cercle)
            radius_cells = int(round(patch.size * coef))
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    if dx**2 + dy**2 <= radius_cells**2:
                        gx, gy = grid_x + dx, grid_y + dy
                        if 0 <= gx < world.grid_resolution and 0 <= gy < world.grid_resolution:
                            landmarks_x.append(gx)
                            landmarks_y.append(gy)
        landmarks = [landmarks_x, landmarks_y]
        landmarks_array = np.array(landmarks)

        observations = np.empty([4], dtype=object)
        observations[:] = [agents_vel_x, agents_vel_y, other_pos, landmarks_array]
        return observations


