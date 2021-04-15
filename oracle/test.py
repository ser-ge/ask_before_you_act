from generator import gen_phrases
from lark import Lark
from gym_minigrid.minigrid import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
import numpy as np
import pytest
import gym

from lang import grammar, TreeToGrid, parser
from oracle import Oracle, OracleWrapper




def test_gen():
    errors = try_gen()
    assert len(errors) == 0

def try_gen():
    errors = []
    parser = Lark(grammar, start='sentence')
    for phrase in gen_phrases():
        try:
            parser.parse(phrase)
        except:
            print(phrase)
            errors.append(phrase)
    return errors


def test_parser():
    parser = Lark(grammar, start='sentence')
    sentences = ['red door is closed', 'red door is north']

    for sent in sentences:

        assert parser.parse(sent)


def test_transformer():

    parser = Lark(grammar, start='sentence')

    transformer = TreeToGrid()
    sent = "red door is north"
    tree = parser.parse(sent)

    premise = transformer.transform(tree)

    assert premise == (4, 0, "north")











# %%
# %% [markdown]
"""
Each tile is encoded as a 3 dimensional tuple:
(OBJECT_IDX, COLOR_IDX, STATE)

I've assumed that the grid is returned as 3 channel np array: TODO check
"""
# %%

door = OBJECT_TO_IDX["door"]
red = COLOR_TO_IDX["red"]
closed = STATE_TO_IDX["closed"]
opn = STATE_TO_IDX["open"]
blue = COLOR_TO_IDX["blue"]

objects = np.array([door, None, None, door]).reshape(2,2)
colors = np.array([red, None , None, blue ]).reshape(2,2)
states = np.array([closed, None, None, opn ]).reshape(2,2)

example_grid = np.stack([objects, colors, states], 2)
# %%


def test_oracle():

    parser = Lark(grammar, start='sentence')
    p_true = 'red door is closed'
    p_false = 'blue door is closed'
    p_none = 'green door is open'

    env = gym.make("MiniGrid-MultiRoom-N4-S5-v0")
    oracle = Oracle(parser, TreeToGrid, env)

    assert oracle.answer(p_true, example_grid)
    assert not oracle.answer(p_false, example_grid)

    from oracle import MyValueError
    with pytest.raises(MyValueError):
        oracle.answer(p_none, example_grid)


def test_oracle_direction():

    from oracle import TRUTH, UNDEFINED, FALSE

    env = gym.make("MiniGrid-Empty-8x8-v0")
    env = OracleWrapper(env)


    p_true = "green goal is south"
    # p_false = "green door is open"
    # p_none = "red door is closed"
    # p_syntax_err = "red door is closedsdafsdaf"

    ans = env._answer(p_true)
    assert (ans[0] == TRUTH).all()
    assert ans[1] == 0
    # assert env._answer(p_false) == (FALSE, 0)
    # assert env._answer(p_none) == (UNDEFINED, 0)
    # assert env._answer(p_syntax_err) == (UNDEFINED, env.syntax_error_reward)



def test_wrapper():

    from oracle import TRUTH, UNDEFINED, FALSE

    env = gym.make("MiniGrid-MultiRoom-N4-S5-v0")
    env = OracleWrapper(env)

    p_true = "green door is closed"
    p_false = "green door is open"
    p_none = "red door is closed"
    p_syntax_err = "red door is closedsdafsdaf"


    assert env._answer(p_true) == (TRUTH, 0)
    assert env._answer(p_false) == (FALSE, 0)
    assert env._answer(p_none) == (UNDEFINED, 0)
    assert env._answer(p_syntax_err) == (UNDEFINED, env.syntax_error_reward)




