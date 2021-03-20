from generator import gen_phrases
from lark import Lark
from gym_minigrid.minigrid import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
import numpy as np

from lang import grammar, TreeToGrid
from oracle import Oracle


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
    sentence = 'red door is closed'
    assert parser.parse(sentence)

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

    oracle = Oracle(parser, TreeToGrid)

    assert oracle.answer(p_true, example_grid) == True
    assert oracle.answer(p_false, example_grid) == False
    assert oracle.answer(p_none, example_grid) is None

