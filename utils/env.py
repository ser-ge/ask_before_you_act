from gym_minigrid.minigrid import Grid, Goal
from gym_minigrid import envs
import re
import gym
import gym_minigrid

def make_env(env_name):
    empty_room_match = re.match(r"MiniGrid-Empty-Random-([0-9]+)x[0-9]+", env_name)
    if empty_room_match:
        env = EmptyRandomEnv(int(empty_room_match.group(1)))
    else:
        env = gym.make(env_name)

    return env

class EmptyRandomEnv(envs.EmptyEnv):
    def __init__(self, size=20):
        super().__init__(size=size, agent_start_pos=None)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal randomly
        self.place_obj(Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"