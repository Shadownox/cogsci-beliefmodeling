import logging
import time

import ccobra
import numpy as np

import mreasoner
import fol
import phm

logger = logging.getLogger(__name__)

class NoBeliefModel(ccobra.CCobraModel):
    def __init__(self, name='NoBelief', method='mReasoner'):
        super(NoBeliefModel, self).__init__(name, ['syllogistic-belief'], ['verify'])

        # Determine method
        self.method = None
        if method == 'mReasoner':
            self.method = mreasoner.MReasoner()
        elif method == 'fol':
            self.method = fol.FOL()
        elif method == 'phm':
            self.method = phm.PHM()

        # Prepare members
        self.time_start = None

    def start_participant(self, **kwargs):
        self.time_start = time.time()

    def end_participant(self, identifier, model_log, **kwargs):
        logger.info('End participant %d (%.2fs)', identifier, time.time() - self.time_start)

    def pre_train(self, dataset, **kwargs):
        # Train method parameters based on verification response
        train_data_response = {}
        for dude_data in dataset:
            for task_data in dude_data:
                syl = ccobra.syllogistic.Syllogism(task_data['item'])
                enc_syllogism = syl.encoded_task
                enc_conclusion = syl.encode_response(task_data['item'].choices[0])

                key = (enc_syllogism, enc_conclusion)
                if key not in train_data_response:
                    train_data_response[key] = [0, 0]
                train_data_response[key][int(task_data['response'])] += 1

        self.method.fit(train_data_response)

    def pre_train_person(self, dataset, **kwargs):
        #pass
        self.pre_train([dataset])

    def predict(self, item, **kwargs):
        syl = ccobra.syllogistic.Syllogism(item)
        enc_syllogism = syl.encoded_task
        enc_conclusion = syl.encode_response(item.choices[0])
        return self.method.evaluate_conclusion(enc_conclusion, enc_syllogism)[1]

    def predict_rating(self, item, **kwargs):
        syl = ccobra.syllogistic.Syllogism(item)
        enc_syllogism = syl.encoded_task
        enc_conclusion = syl.encode_response(item.choices[0])

        possible, necessary = self.method.evaluate_conclusion(enc_conclusion, enc_syllogism)
        if necessary:
            return 5
        return 2
