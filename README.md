



## Working:

- Parser for state premises: "red door is open"
- AST for state premise to state premise named tuples (see TreeToGrid)
- Orcale for state premises given grid (toy example)
- generator of all possible valid phrases

## To do:
- lots
- check that grid channel order matchtes assumed, and test against enviroment
- currentlyu (CHANNEL, HEIGHT, WIDTH)
- direction sentenses (parsing easy to finish)
- direction sentece oracle


Parsing libraries:

- Lark (main parsing and to grid conversion)
- NanGram for generating only, ideally get rid of this if possible.
