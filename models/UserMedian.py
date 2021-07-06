import ccobra
import numpy as np

class UserMedian(ccobra.CCobraModel):
    def __init__(self, name='UserMedian', only_integer=True):
        super(UserMedian, self).__init__(name, ['syllogistic-belief'], ['verify'])
        self.database = {}
        self.only_integer = only_integer

    def pre_train_person(self, dataset):
        for task in dataset:
            item = task["item"]
            syl = ccobra.syllogistic.Syllogism(item)
            rating = task["rating"]
            enc_task = syl.encoded_task
            enc_resp = syl.encode_response(item.choices[0])
            
            key = "{}_{}".format(enc_task, enc_resp)
            if key not in self.database:
                self.database[key] = []
            self.database[key].append(rating)
            
        for key, value in self.database.items():
            self.database[key] = np.median(value)

    def predict(self, item, **kwargs):
        rating = self.predict_rating(item, **kwargs)
        return rating > 3

    def predict_rating(self, item, **kwargs):
        syl = ccobra.syllogistic.Syllogism(item)
        enc_task = syl.encoded_task
        enc_resp = syl.encode_response(item.choices[0])
        key = "{}_{}".format(enc_task, enc_resp)
        rating = self.database[key]
        if self.only_integer and int(rating) != rating:
            if np.random.rand() >= 0.5:
                return int(np.ceil(rating))
            else:
                return int(rating)
        return rating