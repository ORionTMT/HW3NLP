import sys
import copy
import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:
            # Extract features for the current state
            features = self.extractor.get_input_representation(words, pos, state)
            features_tensor = torch.tensor(features, dtype=torch.long).unsqueeze(0)

            # Use the model to predict the scores for each transition
            scores = self.model(features_tensor)
            
            # Get the list of possible transitions
            possible_transitions = []
            for transition_idx, transition in self.output_labels.items():
                if transition[0] == "shift":
                    if len(state.buffer) > 1 or len(state.stack) == 0:
                        possible_transitions.append((transition_idx, transition))
                elif transition[0] == "left_arc":
                    if len(state.stack) > 0 and state.stack[-1] != 0:
                        possible_transitions.append((transition_idx, transition))
                elif transition[0] == "right_arc":
                    if len(state.stack) > 0:
                        possible_transitions.append((transition_idx, transition))

            # Sort the possible transitions by their scores in descending order
            possible_transitions.sort(key=lambda x: scores[0][x[0]], reverse=True)

            # Select the highest-scoring permitted transition
            for transition_idx, transition in possible_transitions:
                if transition[0] == "shift":
                    state.shift()
                    break
                elif transition[0] == "left_arc":
                    state.left_arc(transition[1])
                    break
                elif transition[0] == "right_arc":
                    state.right_arc(transition[1])
                    break

        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        return result

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
