from gym_minigrid.minigrid import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
import numpy as np
import gym


class Oracle:

    def __init__(self, parser, tree_to_grid, require_all=True):

        self.parse = parser.parse
        self.to_state_premise = tree_to_grid().transform
        self.require_all = require_all

    def answer(self, question: str, grid: np.array):
        """
        question: question / premise
        grid: (w h c)
        c: object, type color, state
        """

        try:
            tree = self.parse(question)
        except:
            raise ValueError("invalid syntax")  # TODO appropriate return value, perhaps exceptions

        state_premise = self.to_state_premise(tree)

        if self.require_all and None in state_premise:
            raise ValueError('missing tokens')

        states = grid[..., 2].ravel()
        matched = self.find_objects(state_premise, grid)

        if len(matched) == 0:
            raise ValueError("no such object")
        elif len(matched) > 1:
            raise ValueError("too many objects")
        else:
            return states[matched[0]] == state_premise.state_id

    def find_objects(self, premise, grid):
        objects = grid[..., 0].ravel()
        colors = grid[..., 1].ravel()
        matched = np.where((premise.object_id == objects) & (premise.color_id == colors))[0]
        return matched


class OracleWrapper(gym.core.Wrapper):

    def __init__(self, env, oracle):

        super().__init__(env)

        self.oracle = oracle

    def _answer(self, question):

        full_grid = self.env.grid.encode()

        try:
            ans = self.oracle.answer(question, full_grid)
            return ans

        except ValueError as e:
            print(e)
