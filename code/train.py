import pandas as pd
import warnings   
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train = pd.read_pickle("./preprocessed_data/train.pkl")

# train-val split
X = train[train.columns[train.columns != 'TARGET']]
y = train.TARGET
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=500, test_size=0.2, stratify=y)

# posi-negaのサンプル数調整
tmp = X_train.copy()
tmp["TARGET"] = y_train

tmp_posi = tmp[tmp["TARGET"]==1]
tmp_nega = tmp[tmp["TARGET"]==0]

kf = KFold(n_splits=10, shuffle=True, random_state=500)
i = 0
train_list = []

for _, index in kf.split(tmp_nega):
    train_list.append(pd.concat([tmp_nega.iloc[index], tmp_posi], ignore_index=True, sort=False))

# インスタンス化
pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=250)
)

# モデル10個作成
warnings.simplefilter('ignore') # iterが少ないという警告を無視  

model_list = []

for i in range(len(train_list)):
    tmp_train = train_list[i]
    X = tmp_train[tmp_train.columns[tmp_train.columns != 'TARGET']]
    y = tmp_train.TARGET

    # 学習
    model_list.append(pipe.fit(X, y))

warnings.resetwarnings()

with open('./trained_model/model.pkl', mode='wb') as f:
    pickle.dump(model_list, f)

print("complete!") 