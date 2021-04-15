import nangram

#import os
#THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
#my_file = os.path.join(THIS_FOLDER, 'grammar.bnf')
#grammar = nangram.Grammar.load_bnf(my_file)

grammar = nangram.Grammar.load_bnf('grammar.bnf')

grammar.max_products = 100000
grammar.max_recursions = 100000
grammar.max_repetitions = 100000

gen_phrases = grammar.generate
