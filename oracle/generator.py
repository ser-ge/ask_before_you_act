import nangram

grammar = nangram.Grammar.load_bnf('grammar.bnf')

grammar.max_products = 100000
grammar.max_recursions = 100000
grammar.max_repetitions = 100000

gen_phrases = grammar.generate








