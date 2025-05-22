import numpy as np
import random
import math
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.limit = 4
        world.num_agents = args.num_agents
        world.num_landmarks = args.num_landmarks
        world.collaborative = True
        world.one_reward = True
        self.distribution_radius = 3.85
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.u_noise = 1
            agent.max_speed = 0.51
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 1.2857
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-3.85, +3.85, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-2.7143, +2.7143, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

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


    def get_random_point_in_arena(self):
        return [random.uniform(-self.distribution_radius, self.distribution_radius) for _ in range(2)]


    def is_on_landmark(self, point, world):
       dist_1 = np.sqrt(np.sum(np.square(point - world.landmarks[0].state.p_pos)))
       dist_2 = np.sqrt(np.sum(np.square(point - world.landmarks[1].state.p_pos)))
       dist_3 = np.sqrt(np.sum(np.square(point - world.landmarks[2].state.p_pos)))

       if dist_1 <= world.landmarks[0].size or dist_2 <= world.landmarks[1].size or dist_3 <= world.landmarks[2].size:
           return True
       return False


    def get_closest_agent(self, point, agents):
        min_distance = 2*self.distribution_radius
        for agent in agents:
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - point)))
            if dist < min_distance:
                min_distance = dist
        return min_distance


    def get_expected_distance(self, world):
        # The total distance between each randomly selected point in the arena and the closest robot.
        fTotalDistance = 0
        
        # The expected distance between a randomly selected point and any robot
        fExpectedDistance = 0

        nb_trials = 1000

        agents_to_consider = []

        for agent in world.agents:
            if not self.is_on_landmark(agent.state.p_pos, world):
                agents_to_consider.append(agent)

        if len(agents_to_consider) > 0:
            for i in range(nb_trials):
                random_point = self.get_random_point_in_arena()
                fTotalDistance += self.get_closest_agent(random_point, agents_to_consider) 

            fExpectedDistance = fTotalDistance / nb_trials

        else:
            fExpectedDistance = 2 * self.distribution_radius

        return fExpectedDistance

    
    
    def reward(self, agent, world):      
        performance = ((2 * self.distribution_radius) - self.get_expected_distance(world)) * 100
        if performance < 0:
            performance = 0
        return performance

    def observation(self, agent, world):
        #  get positions of all entities in this agent's reference frame
        entity_pos = np.zeros((2, world.num_landmarks))
        j = 0
        for i, entity in enumerate(world.landmarks):  # world.entities:
            if np.linalg.norm(entity.state.p_pos - agent.state.p_pos) <= 3:
                distance = entity.state.p_pos - agent.state.p_pos
                coef = world.grid_resolution/(world.limit*4)
                scale = (world.grid_resolution//2) - 1
                entity_pos[0][i] = round(coef*distance[0]) + scale
                entity_pos[1][i] = round(coef*distance[1]) + scale
            else:
                j += 1
        if j > 0:
            entity_pos = entity_pos[:, :-j]

        other_pos = np.zeros((2, world.num_agents))
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
        if j > 0:
            other_pos = other_pos[:, :-j]
        
        observations = np.empty([3], dtype=object)
        observations[:] = [agent.state.p_vel, entity_pos, other_pos]
        return observations
