import numpy as np

def pred_proba(X_test, model_list):
    pred_list = []
    for i in range(len(model_list)):
        pred_list.append(model_list[i].predict_proba(X_test)[:, 1])

    p = np.array(pred_list)
    p = np.mean(p, axis = 0)
    return p