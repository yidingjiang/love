import abc
import collections
import enum

import gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Action(enum.IntEnum):
    left = 0
    up = 1
    right = 2
    down = 3
    pickup = 4


class GridObject(abc.ABC):
    """An object that can be placed in the GridEnv.

    Subclasses should register themselves in the list of GridObject above."""

    def __init__(self, color, size=0.4):
        """Constructs.

        Args:
            color (str): valid PIL color.
        """
        self._color = color
        self._size = size

    @property
    def color(self):
        return self._color

    @property
    def size(self):
        return self._size


class Wall(GridObject):
    """An object that cannot be passed through."""

    def __init__(self):
        super().__init__("brown", 1)

    @property
    def type(self):
        return "Wall"


class ComPILEObject(GridObject):
    """Set of 10 objects for ComPILE grid world.

    Unique objects are defined by their color.
    """

    COLORS = ["red", "cornsilk", "blue", "purple", "black", "orange", "yellow",
              "green", "gray", "brown"]

    @classmethod
    def num_types(cls):
        return len(cls.COLORS)

    @classmethod
    def random_object(cls, random):
        return cls(random.choice(cls.COLORS))

    @property
    def type(self):
        return self.COLORS.index(self.color)


class GridEnv(gym.Env):
    """A grid world to move around in.

    The observations are np.ndarrays of shape (height, width, num_objects + 2),
    where obs[y, x, obj_val] = 1 denotes that obj_val is present at (x, y).

    obj_val:
        - value 0, ..., num_objects - 1 denote the objects
        - value num_objects denotes the agent
        - value num_objects + 1 denotes a wall
    """

    def __init__(self, max_steps=20, width=10, height=10):
        """Constructs the environment with dynamics according to env_id.

        Args:
            env_id (int): a valid env_id in TransportationGridEnv.env_ids()
            wrapper (function): see create_env.
            max_steps (int): maximum horizon of a single episode.
        """
        super().__init__()
        self._max_steps = max_steps
        self._grid = [[None for _ in range(height)] for _ in range(width)]
        self._width = width
        self._height = height

    @property
    def observation_space(self):
        low = np.zeros(
                (self._height, self._width, ComPILEObject.num_types() + 2))
        high = np.ones(
                (self._height, self._width, ComPILEObject.num_types() + 2))
        return gym.spaces.Box(low, high, dtype=np.int)

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(Action))

    @property
    def width(self):
        """Returns number of columns in the grid (int)."""
        return self._width

    @property
    def height(self):
        """Returns number of rows in the grid (int)."""
        return self._height

    @property
    def agent_pos(self):
        """Returns location of the agent (np.array)."""
        return self._agent_pos

    @property
    def steps_remaining(self):
        """Returns the number of timesteps remaining in the episode (int)."""
        return self._max_steps - self._steps

    def text_description(self):
        return "grid"

    def get(self, position):
        """Returns the object in the grid at the given position.

        Args:
            position (np.array): (x, y) coordinates.

        Returns:
            object | None
        """
        return self._grid[position[0]][position[1]]

    def place(self, obj, position, exist_ok=False):
        """Places an object in the grid at the given position.

        Args:
            obj (GridObj): object to place into the grid.
            position (np.array): (x, y) coordinates.
        """
        existing_obj = self.get(position)
        if existing_obj is not None and not exist_ok:
            raise ValueError(
                    "Object {} already exists at {}.".format(existing_obj, position))
        self._grid[position[0]][position[1]] = obj

    def _place_objects(self):
        """After grid size is determined, place any objects into the grid."""
        self._agent_pos = np.array([1, 1])

    def _gen_obs(self):
        """Returns an observation (np.array)."""
        obs = np.zeros(
                (self._height, self._width,
                 ComPILEObject.num_types() + 2)).astype(np.float32)
        for x in range(self._width):
            for y in range(self._height):
                obj = self.get((x, y))
                if obj is not None:
                    if isinstance(obj, Wall):
                        obs[y][x][ComPILEObject.num_types() + 1] = 1
                    else:
                        assert isinstance(obj, ComPILEObject)
                        obs[y][x][obj.type] = 1

        # Add agent
        obs[self._agent_pos[1]][self._agent_pos[0]][
                ComPILEObject.num_types()] = 1
        return obs

    def reset(self):
        self._steps = 0
        self._grid = [[None for _ in range(self.height)]
                      for _ in range(self.width)]
        self._place_objects()
        self._history = [np.array(self._agent_pos)]
        return self._gen_obs()

    def step(self, action):
        self._steps += 1

        original_pos = np.array(self._agent_pos)
        if action == Action.left:
            self._agent_pos[0] -= 1
        elif action == Action.up:
            self._agent_pos[1] += 1
        elif action == Action.right:
            self._agent_pos[0] += 1
        elif action == Action.down:
            self._agent_pos[1] -= 1
        elif action == Action.pickup:
            self._grid[self._agent_pos[0]][self._agent_pos[1]] = None

        # Can't walk through wall
        obj = self.get(self._agent_pos)
        if obj is not None and isinstance(obj, Wall):
            self._agent_pos = original_pos

        self._agent_pos[0] = max(min(self._agent_pos[0], self.width - 1), 0)
        self._agent_pos[1] = max(min(self._agent_pos[1], self.height - 1), 0)

        done = self._steps == self._max_steps
        self._history.append(np.array(self._agent_pos))
        return self._gen_obs(), 0, done, {}

    def render(self, mode="human"):
        image = GridRender(self.width, self.height)

        image.draw_rectangle(self.agent_pos, 0.6, "red")
        for x, col in enumerate(self._grid):
            for y, obj in enumerate(col):
                if obj is not None:
                    image.draw_rectangle(np.array((x, y)), obj.size, obj.color)

        for pos in self._history:
            image.draw_rectangle(pos, 0.2, "orange")

        image.write_text(self.text_description())
        image.write_text("Current state: {}".format(self._gen_obs()))
        image.write_text("Env ID: {}".format(self.env_id))
        return image


class GridRender(object):
    """Human-readable rendering of a GridEnv state."""

    def __init__(self, width, height):
        """Creates a grid visualization with a banner with the text.

        Args:
            width (int): number of rows in the SimpleGridEnv state space.
            height (int): number of columns in the SimpleGridEnv state space.
        """
        self._PIXELS_PER_GRID = 100
        self._width = width
        self._height = height

        self._banner = Image.new(
                mode="RGBA",
                size=(width * self._PIXELS_PER_GRID,
                      int(self._PIXELS_PER_GRID * 2)),
                color="white")
        self._text = []
        self._inventory = Image.new(
                mode="RGBA",
                size=(width * self._PIXELS_PER_GRID, self._PIXELS_PER_GRID),
                color="white")
        self._inventory_draw = ImageDraw.Draw(self._inventory)
        self._image = Image.new(
                mode="RGBA",
                size=(width * self._PIXELS_PER_GRID,
                            height * self._PIXELS_PER_GRID),
                color="white")
        self._draw = ImageDraw.Draw(self._image)
        for col in range(width):
            x = col * self._PIXELS_PER_GRID
            line = ((x, 0), (x, height * self._PIXELS_PER_GRID))
            self._draw.line(line, fill="black")

        for row in range(height):
            y = row * self._PIXELS_PER_GRID
            line = ((0, y), (width * self._PIXELS_PER_GRID, y))
            self._draw.line(line, fill="black")

    def write_text(self, text):
        """Adds a banner with the given text. Appends to any previous text.

        Args:
            text (str): text to display on top of rendering.
        """
        self._text.append(text)

    def draw_rectangle(self, position, size, color, outline=False):
        """Draws a rectangle at the specified position with the specified size.

        Args:
            position (np.array): (x, y) position corresponding to a valid state.
            size (float): between 0 and 1 corresponding to how large the rectangle
                should be.
            color (Object): valid PIL color for the rectangle.
        """
        start = position * self._PIXELS_PER_GRID + (0.5 - size / 2) * np.array(
                [self._PIXELS_PER_GRID, self._PIXELS_PER_GRID])
        end = position * self._PIXELS_PER_GRID + (0.5 + size / 2) * np.array(
                [self._PIXELS_PER_GRID, self._PIXELS_PER_GRID])
        self._draw.rectangle((tuple(start), tuple(end)), fill=color)
        if outline:
            self._draw.rectangle((tuple(start), tuple(end)), outline="black", width=3)

    def __deepcopy__(self, memo):
        cls = self.__class__
        deepcopy = cls.__new__(cls)
        memo[id(self)] = deepcopy

        # PIL doesn't support deepcopy directly
        image_copy = self.__dict__["_image"].copy()
        draw_copy = ImageDraw.Draw(image_copy)
        banner_copy = self.__dict__["_banner"].copy()
        setattr(deepcopy, "_image", image_copy)
        setattr(deepcopy, "_draw", draw_copy)
        setattr(deepcopy, "_banner", banner_copy)

        for k, v in self.__dict__.items():
            if k not in ["_image", "_draw", "_banner"]:
                setattr(deepcopy, k, copy.deepcopy(v, memo))
        return deepcopy

    def image(self):
        """Returns PIL Image representing this render."""
        image = Image.new(
                mode="RGBA",
                size=(
                        self._image.width, self._image.height + self._banner.height +
                        self._inventory.height))
        draw = ImageDraw.Draw(self._banner)
        font = ImageFont.truetype("asset/fonts/arial.ttf", 20)
        draw.text((0, 0), "\n".join(self._text), (0, 0, 0), font=font)
        image.paste(self._banner, (0, 0))
        image.paste(self._inventory, (0, self._banner.height))
        image.paste(self._image, (0, self._banner.height + self._inventory.height))
        return image


class ComPILEEnv(GridEnv):
    """Implements the grid world defined in the ComPILE paper.

    Each instance of the environment is associated with a random initial state
    and a sequence of items to pick up. These are fixed for all episodes in the
    same environment, and are identified by the seed argument in the
    constructor.

    The random initial state is generated by uniformly sampling 6 ComPILE
    objects and placing them at random positions in the grid, and generating a
    random initial agent position.

    The sequence of items is just a list of 3 random ComPILE object types,
    selected without replacement from the 6 randomly generated.
    """

    def __init__(self, seed, max_steps=50, width=10, height=10, num_objects=6,
                 visit_length=5, sparse_reward=False):
        super().__init__(max_steps=max_steps, width=width, height=height)

        self._sparse_reward = sparse_reward

        self._random = np.random.RandomState(seed)
        initial_objects = self._random.choice(
                list(range(ComPILEObject.num_types())), size=num_objects,
                replace=False)
        self._initial_objects = [ComPILEObject.random_object(self._random)
                                 for _ in range(num_objects)]

        self._object_positions = []
        for obj in self._initial_objects:
            while True:
                x = self._random.randint(1, width - 1)
                y = self._random.randint(1, height - 1)
                allowed = (x, y) not in self._object_positions
                if allowed:
                    self._object_positions.append((x, y))
                    break

        num_walls = self._random.randint(5)
        while True:
            self._wall_positions = []
            for _ in range(num_walls):
                x = self._random.randint(1, width - 1)
                y = self._random.randint(1, height - 1)
                pos = (x, y)
                length = self._random.randint(1, 4)
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                direction = directions[self._random.randint(len(directions))]
                for i in range(length):
                    wall_pos = tuple(np.array(pos) + np.array(direction) * i)
                    allowed = self._in_bounds(wall_pos)
                    allowed = allowed and wall_pos not in self._wall_positions
                    allowed = allowed and wall_pos not in self._object_positions
                    if allowed:
                        self._wall_positions.append(wall_pos)

            if self._all_states_reachable(self._object_positions[0]):
                break

        object_types = [obj.type for obj in self._initial_objects]
        self._items_to_visit = self._random.choice(
                object_types, size=visit_length, replace=False)

        # index into _items_to_visit that the agent is currently on
        self._visited_index = 0

    def _in_bounds(self, pos):
        in_bounds = 1 <= pos[0] < self._width - 1
        return in_bounds and 1 <= pos[1] < self._height - 1

    def _place_objects(self):
        """After grid size is determined, place any objects into the grid."""
        self._agent_pos = np.array(self._initial_agent_pos)

        for obj, position in zip(self._initial_objects, self._object_positions):
            self.place(obj, position)

        # Surrounding walls
        for x in range(self._width):
            self.place(Wall(), (x, 0), exist_ok=True)
            self.place(Wall(), (x, self._height - 1), exist_ok=True)

        for y in range(self._height):
            self.place(Wall(), (0, y), exist_ok=True)
            self.place(Wall(), (self._width - 1, y), exist_ok=True)

        for wall_pos in self._wall_positions:
            self.place(Wall(), wall_pos)

    def reset(self):
        while True:
            self._initial_agent_pos = (
                self._random.randint(1, self._width - 1),
                self._random.randint(1, self._height - 1))
            obj = self.get(self._initial_agent_pos)
            allowed = obj is None or isinstance(obj, Wall)
            allowed = (
                    allowed and
                    self._initial_agent_pos not in self._wall_positions)
            if allowed:
                break

        self._visited_index = 0
        return super().reset()

    def step(self, action):
        done = False
        reward = 0
        pickup = False
        if action == Action.pickup:
            obj = self.get(self._agent_pos)
            item_to_visit = self._items_to_visit[self._visited_index]
            if obj is not None and obj.type == item_to_visit:
                self._visited_index += 1
                reward = int(not self._sparse_reward)
                pickup = True

        # Always give reward for completing task
        if self._visited_index == len(self._items_to_visit):
            reward = 1
            done = True

        state, _, max_steps, info = super().step(action)
        done = max_steps or done

        return state, reward, done, info

    def _all_states_reachable(self, start):
        bfs_queue = collections.deque([start])
        visited = set([start])
        while len(bfs_queue) > 0:
            pos = bfs_queue.popleft()

            actions = [Action.up, Action.down, Action.left, Action.right]
            deltas = [(0, 1), (0, -1), (-1, 0), (1, 0)]
            for action, delta in zip(actions, deltas):
                neighbor = (pos[0] + delta[0], pos[1] + delta[1])
                if not self._in_bounds(neighbor):
                    continue

                if neighbor not in visited and neighbor not in self._wall_positions:
                    visited.add(neighbor)
                    bfs_queue.append(neighbor)
        return len(visited) == (self._width - 2) * (self._height - 2) - len(self._wall_positions)

    def solve(self):
        """Returns list of actions (list[int]) that optimally solves env."""

        def shortest_path(start, finish):
            bfs_queue = collections.deque([(start, [])])
            visited = set([start])
            while len(bfs_queue) > 0:
                pos, path = bfs_queue.popleft()
                if pos == finish:
                    return path

                actions = [Action.up, Action.down, Action.left, Action.right]
                deltas = [(0, 1), (0, -1), (-1, 0), (1, 0)]
                for action, delta in zip(actions, deltas):
                    neighbor = (pos[0] + delta[0], pos[1] + delta[1])
                    if not self._in_bounds(neighbor):
                        continue

                    obj = self.get(neighbor)
                    not_wall = obj is None or not isinstance(obj, Wall)
                    if neighbor not in visited and not_wall:
                        visited.add(neighbor)
                        neighbor_path = path + [action]
                        bfs_queue.append((neighbor, neighbor_path))
            return None


        def shortest_path_to_obj_types(start, obj_types, visited_objs):
            if len(obj_types) == 0:
                return []

            indices = [i for i, obj in enumerate(self._initial_objects)
                       if obj.type == obj_types[0] and i not in visited_objs]
            potential_positions = [self._object_positions[i] for i in indices]

            paths = []
            for obj_index, pos in zip(indices, potential_positions):
                prefix = shortest_path(start, pos) + [Action.pickup]
                suffix = shortest_path_to_obj_types(
                        pos, obj_types[1:], visited_objs | {obj_index})
                paths.append(prefix + suffix)
            return min(paths, key=lambda path: len(path))

        return shortest_path_to_obj_types(
                self._initial_agent_pos, self._items_to_visit, set())

    def _gen_obs(self):
        state_obs = super()._gen_obs()
        #return state_obs, self._items_to_visit[self._visited_index]
        item = 0
        if self._visited_index < len(self._items_to_visit):
            item = self._items_to_visit[self._visited_index]
        return state_obs, item

    def render(self, mode="human"):
        image = GridRender(self.width, self.height)

        image.draw_rectangle(self.agent_pos, 0.9, "cyan")
        for x, col in enumerate(self._grid):
            for y, obj in enumerate(col):
                if obj is not None:
                    image.draw_rectangle(np.array((x, y)), 0.4, obj.color)

        for pos in self._history:
            image.draw_rectangle(pos, 0.2, "orange")

        image.write_text(" ".join(
            ComPILEObject.COLORS[obj] for obj in self._items_to_visit))
        return image
