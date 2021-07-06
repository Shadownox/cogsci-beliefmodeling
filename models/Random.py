import ccobra
import numpy as np

class RandomModel(ccobra.CCobraModel):
    def __init__(self, name='Random'):
        super(RandomModel, self).__init__(name, ['syllogistic-belief'], ['verify'])

    def predict(self, item, **kwargs):
        return np.random.choice([True, False])

    def predict_rating(self, item, **kwargs):
        return int(np.random.randint(1, 7))