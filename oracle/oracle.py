# from gym_minigrid.minigrid import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
import random

import numpy as np
import gym

from oracle.lang import StatePremise, DirectionPremise, parser, TreeToGrid


class Oracle:

    def __init__(self, parser, tree_to_grid, env, require_all=True):

        self.env = env

        self.parse = parser.parse
        self.to_premise = tree_to_grid().transform
        self.require_all = require_all

    def answer(self, question: str, grid=None):
        """
        question: question / premise
        grid: (w h c)
        c: object, type color, state
        """

        try:
            tree = self.parse(question)
        except:
            raise MySyntaxError("invalid syntax")  # TODO appropriate return value, perhaps exceptions

        premise = self.to_premise(tree)

        # if self.require_all and None in state_premise:
        #     raise  ValueError('missing tokens')

        if grid is None:
            grid = self.env.grid.encode()

        if isinstance(premise, DirectionPremise):
            return self.answer_direction(premise, grid)

        elif isinstance(premise, StatePremise):
            return self.answer_state(premise, grid)

        else:
            raise MyValueError("no such premise type")

    def answer_state(self, premise, grid):
        states = grid[..., 2].ravel()
        matched = self.find_objects(premise, grid)
        matched = self.validate_matched(matched)

        return states[matched[0]] == premise.state_id

    def answer_direction(self, premise, grid):
        height, width, _ = grid.shape
        matched = self.find_objects(premise, grid)
        matched = self.validate_matched(matched)

        x = matched[0] % width
        y = matched[0] // height

        direction = premise.direction
        agent_x, agent_y = self.env.agent_pos

        if direction == 'north':
            return y < agent_y
        elif direction == 'south':
            return y > agent_y
        elif direction == 'west':
            return x < agent_x
        elif direction == 'east':
            return x > agent_x
        else:
            raise MyValueError()

    def find_objects(self, premise, grid):
        objects = grid[..., 0].ravel()
        colors = grid[..., 1].ravel()
        matched = np.where((premise.object_id == objects) & (premise.color_id == colors))[0]
        return matched

    def validate_matched(self, matched):

        if len(matched) == 0:
            raise MyValueError("no such object")
        elif len(matched) > 1:
            raise MyValueError("too many objects")
        else:
            return matched


from enum import Enum

class Answer(Enum):
    TRUTH = 1
    FALSE = 2
    UNDEFINED = 3
    BAD_SYNTAX = 4

    def decode(self):
        return {
            "TRUTH": np.array([1, 1]),
            "FALSE": np.array([0, 0]),
            "UNDEFINED": np.array([1, 0]),
            "BAD_SYNTAX": np.array([0, 1])
        }[self.name]

    def __str__(self):
        return {
            "TRUTH": 'True',
            "FALSE": 'False',
            "UNDEFINED": 'Undefined',
            "BAD_SYNTAX": 'Bad syntax',
        }[self.name]

class OracleWrapper(gym.core.Wrapper):

    def __init__(self, env, syntax_error_reward=-0.1, undefined_error_reward=-0.1, ans_random=False):

        super().__init__(env)

        self.oracle = Oracle(parser=parser, tree_to_grid=TreeToGrid, env=env)
        self.ans_random = ans_random

        self.syntax_error_reward = syntax_error_reward
        self.undefined_error_reward = undefined_error_reward

    def answer(self, question):

        full_grid = self.env.grid.encode()

        try:

            if self.ans_random:  # TODO still penalize if syntax is incorrect?
                ans = random.choice(list(Answer))
            else:
                ans = self.oracle.answer(question, full_grid)
                ans = Answer.TRUTH if ans else Answer.FALSE

            return ans, 0

        except MyValueError:
            return (Answer.UNDEFINED, self.undefined_error_reward)

        except MySyntaxError:
            return (Answer.BAD_SYNTAX, self.syntax_error_reward)


class MySyntaxError(Exception):
    pass


class MyValueError(Exception):
    pass
