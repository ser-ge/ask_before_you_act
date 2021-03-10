from gym_minigrid.minigrid import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX


class Oracle:

    def __init__(self, parser, tree_to_grid, require_all=True):

        self.parse = parser.parse
        self.to_grid_slice = tree_to_grid().transform
        self.require_all = require_all


    def answer(self, question, example_grid):
        """
        Assumed grid shape: (Channels, height, width) TODO check grid world
        """

        try:
            tree = self.parse(question)
        except:
            print("invalid syntac, I dont understand")
            return None  #TODO appropriate return value, perhaps exceptions

        state_premise = self.to_grid_slice(tree)

        if self.require_all and None in state_premise:
            print("Mising Values")
            return None


        objects = example_grid[0].ravel()
        colors = example_grid[1].ravel()
        states = example_grid[2].ravel()

        matched = []

        #TODO Vectorise!
        for pos, (obj, clr) in enumerate(zip(objects, colors)):
            if state_premise.object_id == obj and state_premise.color_id == clr:
                matched.append(pos)

        if len(matched) == 0:
            print("No such object")
            return None
        elif len(matched) > 1:
            print("Multiple Objects, I dont understand")
            return None
        else:
            return states[matched[0]] == state_premise.state_id
















