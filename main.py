import string
import time

import pandas as pd
from nltk import ViterbiParser

from utils import get_treebank_productions, get_treebank_grammar


def manage_unknown_words(sentences, voc):
    new_sentences = []
    digits = set(string.digits)
    for sentence in sentences:
        sentence = sentence.split(' ')
        for i, token in enumerate(sentence):
            if token not in voc:
                if len(set(token).intersection(digits)) != 0:
                    sentence[i] = 'UNK_NUM'
                else:
                    sentence[i] = 'UNK_'
        new_sentences.append(sentence)
    return new_sentences


if __name__ == '__main__':
    dev_data = pd.read_csv('data/cola_public/tokenized/in_domain_dev.tsv', header=None, delimiter='\t')
    dev_data = dev_data[~dev_data[2].isna()].reset_index(drop=True)
    dev_sentences = dev_data[3].to_list()[:75]

    productions, lengths = get_treebank_productions()
    for dismiss_length in range(20000, 10000, -4000):
        grammar, voc = get_treebank_grammar(productions, dismiss_length)
        parser = ViterbiParser(grammar)

        sentences = manage_unknown_words(dev_sentences, voc)
        sentences = sorted(sentences, key=lambda sentence: len(sentence))

        observations = []
        for sentence in sentences:
            length = len(sentence)

            tic = time.time()
            parses = parser.parse_all(sentence)
            tac = time.time() - tic

            observation = [str(length), str(tac), str(len(parses))]
            observations.append(observation)
            print(observation)

        log_file = f'logs/{dismiss_length}_{len(grammar.productions())}.log'
        log_file = open(log_file, 'w')
        for observation in observations:
            log_file.write(' '.join(observation) + '\n')
        log_file.close()
