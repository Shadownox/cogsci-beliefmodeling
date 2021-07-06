import logging
import time

import ccobra
import numpy as np

import mreasoner
import fol
import phm

import SelectiveScrutinyModel as ss
import MisinterpretedNecessity as mn
import NoBeliefModel as nobel

logger = logging.getLogger(__name__)

class BeliefPortfolio(ccobra.CCobraModel):
    def __init__(self, name='BeliefPortfolio', method='mReasoner', optimize_rating=False):
        super(BeliefPortfolio, self).__init__(name, ['syllogistic-belief'], ['verify'])

        self.selectivescrutiny = ss.SelectiveScrutinyRestr(method=method)
        self.misinterpretednecessity = mn.MisinterpretedNecessityRestr(method=method)
        self.nobelief = nobel.NoBeliefModel(method=method)
        self.optimal_belief_model = "nobel"
        self.optimize_rating = optimize_rating

        # Prepare members
        self.time_start = None

    def start_participant(self, **kwargs):
            self.time_start = time.time()

    def end_participant(self, identifier, model_log, **kwargs):
        logger.info('End participant %d (%.2fs)', identifier, time.time() - self.time_start)
        model_log["belief_model"] = self.optimal_belief_model 
        
    def pre_train(self, dataset, **kwargs):
        self.selectivescrutiny.pre_train(dataset, **kwargs)
        self.misinterpretednecessity.pre_train(dataset, **kwargs)
        self.nobelief.pre_train(dataset, **kwargs)

        # find the best belief x syl model combination
        best_score = 0
        best_error = 9999
        best = "nobel"
        for bel_model in ["nobel", "mn", "ss"]:
            bm = self.get_bel_model(bel_model)
           
            if self.optimize_rating:
                error = 0                
                for dude_data in dataset:
                    for task_data in dude_data:
                        pred = bm.predict_rating(task_data['item'], **task_data["full"])
                        truth = task_data['rating']
                        error += abs(pred - truth)
                if error < best_error:
                    best_error = error
                    best = bel_model
            else:
                score = 0                
                for dude_data in dataset:
                    for task_data in dude_data:
                        pred = bm.predict(task_data['item'], **task_data["full"])
                        truth = task_data['response']
                        score += int(pred == truth)
                if score > best_score:
                    best_score = score
                    best = bel_model

        self.optimal_belief_model = best

    def get_bel_model(self, bel_model):
        if bel_model == "nobel":
            return self.nobelief
        elif bel_model == "ss":
            return self.selectivescrutiny
        else:
            return self.misinterpretednecessity

    def pre_train_person(self, dataset, **kwargs):
        self.pre_train([dataset])
        
    def predict(self, item, **kwargs):
        return self.predict_rating(item, **kwargs) > 3

    def predict_rating(self, item, **kwargs):
        bm = self.get_bel_model(self.optimal_belief_model)
        return bm.predict_rating(item, **kwargs)
