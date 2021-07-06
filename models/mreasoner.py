import numpy as np
import ccobra


CACHE_NECESSARY = 'caches/necessary.npy'
CACHE_POSSIBLE = 'caches/possible.npy'

class MReasoner():
    def __init__(self):
        # Load mReasoner cache
        self.cache_necessary = np.load(CACHE_NECESSARY)
        self.cache_possible = np.load(CACHE_POSSIBLE)

        # Initialize parameters
        self.idx_epsilon = 0
        self.idx_lambda = 0
        self.idx_omega = 0
        self.idx_sigma = 0

        self.n_epsilon, self.n_lambda, self.n_omega, self.n_sigma = self.cache_necessary.shape[:-2]

    def fit(self, train_data):
        best_parameters = [0, 0, 0, 0]
        best_value = -1

        for idx_epsilon in range(self.n_epsilon):
            for idx_lambda in range(self.n_lambda):
                for idx_omega in range(self.n_omega):
                    for idx_sigma in range(self.n_epsilon):
                        # Obtain values
                        is_necessary = self.cache_necessary[
                            idx_epsilon, idx_lambda, idx_omega, idx_sigma]

                        # Evaluate stuff
                        score = 0
                        for key, val in train_data.items():
                            syl, concl = key
                            idx_syl = ccobra.syllogistic.SYLLOGISMS.index(syl)
                            idx_concl = ccobra.syllogistic.RESPONSES.index(concl)
                            necc = is_necessary[idx_syl, idx_concl]

                            score += val[int(necc < 0.5)] / np.sum(val)
                        score /= len(train_data)

                        if score > best_value:
                            best_parameters = [idx_epsilon, idx_lambda, idx_omega, idx_sigma]
                            best_value = score

        self.idx_epsilon, self.idx_lambda, self.idx_omega, self.idx_sigma = best_parameters

    def evaluate_conclusion(self, conclusion, syllogism):
        # Obtain indices
        idx_syl = ccobra.syllogistic.SYLLOGISMS.index(syllogism)
        idx_concl = ccobra.syllogistic.RESPONSES.index(conclusion)

        is_possible = self.cache_possible[
            self.idx_epsilon, self.idx_lambda, self.idx_omega, self.idx_sigma, idx_syl, idx_concl
        ]

        is_necessary = self.cache_necessary[
            self.idx_epsilon, self.idx_lambda, self.idx_omega, self.idx_sigma, idx_syl, idx_concl
        ]

        return is_possible >= 0.5, is_necessary >= 0.5
