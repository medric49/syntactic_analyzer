from nltk.corpus import treebank
from nltk import Production
from nltk import treetransforms
from nltk import induce_pcfg
from nltk.parse import pchart
from nltk import Nonterminal
from nltk.parse import ViterbiParser


def get_additional_productions():
    productions = []
    unk_non_terminals = ['MD', 'RBR', 'RP', 'PRP', 'DT', 'JJS', 'JJR', 'IN', 'NNPS', 'VBP', 'VBZ', 'RB', '-NONE-',
                         'VBD', 'VBG', 'VBN', 'VB', 'CD', 'NNS', 'JJ', 'NNP', 'NN']

    unk_num_non_terminals = ['CD', 'NNS', 'NNPS', 'NN', 'NNP', 'JJ']

    for n_ter in unk_non_terminals:
        left = Nonterminal(n_ter)
        right = ('UNK_',)
        productions.append(Production(left, right))

    for n_ter in unk_num_non_terminals:
        left = Nonterminal(n_ter)
        right = ('UNK_NUM',)
        productions.append(Production(left, right))

    return productions


def get_treebank_productions():
    productions = []
    lengths = []

    for item in treebank.fileids():
        for tree in treebank.parsed_sents(item):
            lengths.append(len(tree.leaves()))
            tree.collapse_unary(collapsePOS=False)

            for prod in tree.productions():
                if prod.is_lexical():
                    non_terminal = prod.lhs()

                    token = prod.rhs()[0].lower()
                    prod = Production(non_terminal, (token,))

                productions.append(prod)

    return productions, lengths


def get_treebank_grammar(productions, dismiss_length=0):
    s = Nonterminal('S')

    grammar = induce_pcfg(s, productions)

    prob_productions = grammar.productions()
    sorted_prob_productions = sorted(prob_productions, key=lambda prod: prod.prob())

    sorted_prob_productions_prob_productions = sorted_prob_productions[dismiss_length:]

    voc = set([prod.rhs()[0] for prod in sorted_prob_productions_prob_productions if prod.is_lexical()])

    sorted_prob_productions_prob_productions += get_additional_productions()

    grammar = induce_pcfg(s, sorted_prob_productions_prob_productions)

    return grammar, voc

