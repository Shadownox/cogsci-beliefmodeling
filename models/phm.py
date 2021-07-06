import numpy as np
import ccobra

import helpers.phm as phmhelper

class PHM():
    def __init__(self):
        # Initialize parameters
        self.p_entailment = False
        self.max_confidence = {'A': 1, 'I': 0, 'E': 0, 'O': 0}

        self.phm_inst = phmhelper.PHM()

    def fit(self, train_data):
        best = None
        best_score = -1

        # Iterate over parameters
        max_confidence_grid = [
            {'A': 1, 'I': 1, 'E': 1, 'O': 1},
            {'A': 1, 'I': 1, 'E': 1, 'O': 0},
            {'A': 1, 'I': 1, 'E': 0, 'O': 1},
            {'A': 1, 'I': 1, 'E': 0, 'O': 0},
            {'A': 1, 'I': 0, 'E': 0, 'O': 0},
            {'A': 0, 'I': 0, 'E': 0, 'O': 0}
        ]

        # Prepare the optimization loop
        for p_ent in [1, 0]:
            for max_conf in max_confidence_grid:
                self.p_entailment = p_ent
                self.max_confidence = max_conf

                score = 0
                for key, val in train_data.items():
                    syl, concl = key
                    possible, necessary = self.evaluate_conclusion(concl, syl)
                    score += val[int(necessary)]

                score /= len(train_data)

                if  score > best_score:
                    best_score = score
                    best = (p_ent, max_conf)

        self.p_entailment, self.max_confidence = best

    def evaluate_conclusion(self, conclusion, syllogism):
        preds = self.phm_inst.generate_conclusions(syllogism, self.p_entailment)

        if conclusion in preds:
            if not self.phm_inst.max_heuristic(syllogism, *[self.max_confidence[x] for x in ['A', 'I', 'E', 'O']]):
                return True, False
            else:
                return True, True
        else:
            return False, False
