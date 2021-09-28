import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import utils

test = pd.read_pickle("./preprocessed_data/test.pkl")
submit = pd.read_csv("./data/test.csv", usecols=["SK_ID_CURR"])

with open("./trained_model/model.pkl", mode='rb') as f:
    model_list = pickle.load(f)

submit["TARGET"] = utils.pred_proba(test, model_list)

submit.to_csv("./submit/submission.csv", index=False)

print("complete!")