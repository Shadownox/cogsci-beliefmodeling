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

class Portfolio(ccobra.CCobraModel):
    def __init__(self, name='Portfolio', optimize_rating=False):
        super(Portfolio, self).__init__(name, ['syllogistic-belief'], ['verify'])

        self.selectivescrutiny = ss.SelectiveScrutinyRestr()
        self.misinterpretednecessity = mn.MisinterpretedNecessityRestr()
        self.nobelief = nobel.NoBeliefModel()

        self.mreasoner = mreasoner.MReasoner()
        self.phm = phm.PHM()
        self.fol = fol.FOL()
        
        self.optimize_rating = optimize_rating

        # Prepare members
        self.time_start = None
        
        self.optimal_syl_model = "fol"
        self.optimal_belief_model = "nobel"

    def start_participant(self, **kwargs):
            self.time_start = time.time()

    def end_participant(self, identifier, model_log, **kwargs):
        logger.info('End participant %d (%.2fs)', identifier, time.time() - self.time_start)
        model_log["syl_model"] = self.optimal_syl_model
        model_log["belief_model"] = self.optimal_belief_model 
        
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

        self.mreasoner.fit(train_data_response)
        self.phm.fit(train_data_response)
        self.fol.fit(train_data_response)
        
        # find the best belief x syl model combination
        best_score = 0
        best_error = 9999
        best_comb = ["fol", "nobel"]
        for syl_model in ["fol", "mreasoner", "phm"]:
            sm = self.get_syl_model(syl_model)

            for bel_model in ["nobel", "mn", "ss"]:
                bm = self.get_bel_model(bel_model)
                bm.method = sm
               
                if self.optimize_rating:
                    error = 0                
                    for dude_data in dataset:
                        for task_data in dude_data:
                            pred = bm.predict_rating(task_data['item'], **task_data["full"])
                            truth = task_data['rating']
                            error += abs(pred - truth)
                    if error < best_error:
                        best_error = error
                        best_comb = (syl_model, bel_model)
                else:
                    score = 0                
                    for dude_data in dataset:
                        for task_data in dude_data:
                            pred = bm.predict(task_data['item'], **task_data["full"])
                            truth = task_data['response']
                            score += int(pred == truth)
                    if score > best_score:
                        best_score = score
                        best_comb = (syl_model, bel_model)
        self.optimal_syl_model = best_comb[0]
        self.optimal_belief_model = best_comb[1]

    def get_syl_model(self, syl_model):
        if syl_model == "fol":
            return self.fol
        elif syl_model == "mreasoner":
            return self.mreasoner
        else:
            return self.phm

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
        sm = self.get_syl_model(self.optimal_syl_model)
        bm = self.get_bel_model(self.optimal_belief_model)
        bm.method = sm
        return bm.predict_rating(item, **kwargs)
