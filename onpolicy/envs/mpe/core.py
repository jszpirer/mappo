import numpy as np
import seaborn as sns
from math import cos, sin

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties of wall entities
class Wall(object):
    def __init__(self, startpoint=(0, 0), endpoint=(0, 0), width=0.1,
                 hard=True):
        # startpoint of wall
        self.start = np.array(startpoint)
        # endpoint of wall 
        self.end = np.array(endpoint)
        # width of wall
        self.width = width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state: including internal/mental state p_pos, p_vel
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # commu channel
        self.channel = None

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agent are adversary
        self.adversary = False
        # agent are dummy
        self.dummy = False
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state: including communication state(communication utterance) c and internal/mental state p_pos, p_vel
        self.state = AgentState()
        # action: physical action u & communication action c
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # zoe 20200420
        self.goal = None
        # agents can have a direction
        self.direction = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.walls = []
        self.limit = 100
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping（阻尼）
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None
        # zoe 20200420
        self.world_length = 25
        self.world_step = 0
        self.num_agents = 0
        self.num_landmarks = 0
        self.one_reward = False
        self.grid_resolution = 0
        self.grid_resolution_critic = 0
        self.nb_additional_data = 0
        self.omniscient_critic = False
        self.discrete_actions = True

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities （size相加�?
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)

        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        # r g b
        dummy_colors = [(0.25, 0.75, 0.25)] * n_dummies
        # sns.color_palette("OrRd_d", n_adversaries)
        adv_colors = [(0.75, 0.25, 0.25)] * n_adversaries
        # sns.color_palette("GnBu_d", n_good_agents)
        good_colors = [(0.25, 0.25, 0.75)] * n_good_agents
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

    # update state of the world
    def step(self):
        # zoe 20200420
        self.world_step += 1
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        # calculate and store distances between all entities
        if self.cache_dists:
            self.calculate_distances()

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(
                    *agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                # force = mass * a * action + n
                p_force[i] = (
                    agent.mass * agent.accel if agent.accel is not None else agent.mass) * agent.action.u + noise
                # if the agent has a specific direction, the force should be in the robot system of axis
                if self.use_directions:
                    x = p_force[i][0]
                    y = p_force[i][1]
                    p_force[i][0] = cos(agent.direction) * x + sin(agent.direction) * y
                    p_force[i][1] = -sin(agent.direction) * x + cos(agent.direction) * y
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if(b <= a):
                    continue
                [f_a, f_b] = self.get_entity_collision_force(a, b)
                if(f_a is not None):
                    if(p_force[a] is None):
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if(f_b is not None):
                    if(p_force[b] is None):
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            if entity_a.movable:
                for wall in self.walls:
                    wf = self.get_wall_collision_force(entity_a, wall)
                    if wf is not None:
                        if p_force[a] is None:
                            p_force[a] = 0.0
                        p_force[a] = p_force[a] + wf
        return p_force

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt
            if abs(entity.state.p_pos[0]) > (self.limit - entity.size):
                if entity.state.p_pos[0] > (self.limit - entity.size):
                    entity.state.p_pos[0] = self.limit - entity.size
                else:
                    entity.state.p_pos[0] = - (self.limit - entity.size)
            if abs(entity.state.p_pos[1]) > (self.limit - entity.size):
                if entity.state.p_pos[1] > (self.limit - entity.size):
                    entity.state.p_pos[1] = self.limit - entity.size
                else:
                    entity.state.p_pos[1] = - (self.limit - entity.size)
            if self.use_directions:
                # New direction after the moving step (need to take into account that the robot while follow a circle)
                # entity.direction = np.arctan2(entity.state.p_vel[0], entity.state.p_vel[1])
                # entity.direction = np.mod(entity.direction, 2 * np.pi)
                # Need the new position in the robot system
                x_B = cos(entity.direction) * entity.state.p_pos[0] + sin(entity.direction) * entity.state.p_pos[1]
                y_B = -sin(entity.direction) * entity.state.p_pos[0] + cos(entity.direction) * entity.state.p_pos[1]
                alpha = arctan2(y_B, x_B)
                theta = np.pi - 2 * alpha
                entity.direction += theta
                entity.direction = np.mod(entity.direction, 2 * np.pi)

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * \
                agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_entity_collision_force(self, ia, ib):
        entity_a = self.entities[ia]
        entity_b = self.entities[ib]
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None]  # neither entity moves
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        if self.cache_dists:
            delta_pos = self.cached_dist_vect[ia, ib]
            dist = self.cached_dist_mag[ia, ib]
            dist_min = self.min_dists[ia, ib]
        else:
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        if dist == 0:
            dist = 0.01
        force = self.contact_force * delta_pos / dist * penetration
        if entity_a.movable and entity_b.movable:
            # consider mass in collisions
            force_ratio = entity_b.mass / entity_a.mass
            force_a = force_ratio * force
            force_b = -(1 / force_ratio) * force
        else:
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # Calculate the force from a wall to an entity
    def get_wall_collision_force(self, entity, wall):
        if entity.ghost and not wall.hard:
            return None

        ent_pos = entity.state.p_pos
        wall_vec = wall.end - wall.start
        wall_len = np.linalg.norm(wall_vec)
        wall_dir = wall_vec / wall_len

        # Vector from wall start to entity
        to_entity = ent_pos - wall.start

        # Projection of entity position onto wall
        proj_length = np.dot(to_entity, wall_dir)
        proj_point = wall.start + proj_length * wall_dir

        # Clamp projection to wall segment
        proj_length_clamped = np.clip(proj_length, 0, wall_len)
        closest_point = wall.start + proj_length_clamped * wall_dir

        # Distance from entity to wall
        delta = ent_pos - closest_point
        dist = np.linalg.norm(delta)

        # Compute penetration
        k = self.contact_margin
        dist_min = entity.size + 0.5 * wall.width
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k

        if dist == 0:
            force_dir = np.random.randn(2)
            force_dir /= np.linalg.norm(force_dir)
        else:
            force_dir = delta / dist

        force = self.contact_force * force_dir * penetration
        return force
