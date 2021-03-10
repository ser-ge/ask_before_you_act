from lark import Lark, tree, Transformer
from gym_minigrid.minigrid import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

from collections import namedtuple


grammar = """
    sentence: noun verb STATE -> state_premise
              | noun verb DIRECTION -> direction_premise
    noun: adj? NOUN
    verb: VERB
    adj: ADJ
    state: STATE
    NOUN: "unseen"| "empty" | "wall"  | "floor" | "door"  |  "key" | "ball "| "box" | "goal"  | "lava"  | "agent"
    STATE : "open" | "closed" | "locked"
    VERB: "is"
    ADJ : "red" | "green" | "blue"| "purple"| "yellow" | "grey"
    DIRECTION : "north" | "south" | "west" | "east"
    %import common.WS
    %ignore WS
"""


# %%

class TreeToGrid(Transformer):

    def NOUN(self, noun):
        return OBJECT_TO_IDX[noun]

    def adj(self, adj):
        return COLOR_TO_IDX[adj[0]]

    def direction(self, direction):
        pass

    def STATE(self, state):
        return STATE_TO_IDX[state]

    noun = lambda self, noun : list(reversed(noun))
    verb = lambda self, _ : None

    def state_premise(self, sent):
        noun_phrase, _, state = sent

        p = namedtuple("state_premise", ["object_id", "color_id", "state_id"])

        if len(noun_phrase) == 1:
            return p(noun_phrase[0], None , state)
        else:
            return p(noun_phrase[0], noun_phrase[1], state)

# %%
