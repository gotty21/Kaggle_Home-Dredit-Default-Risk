import pandas as pd
import numpy as np

train = pd.read_csv("./data/train.csv", encoding='UTF-8')
test = pd.read_csv("./data/test.csv", encoding='UTF-8')
df = pd.concat([train, test], ignore_index=True, sort=False)

# 欠損値を含む列削除
df_t = df.dropna(how='any', axis=1).copy()
df_t["TARGET"] = df["TARGET"].copy()

# EXT_SOURCE追加
for i in range(1, 4):
    col = "EXT_SOURCE_" + str(i)
    df_t[col] = df[col].copy()
    val = df_t[df_t["TARGET"].notnull()][col].mean()
    df_t[col] = df_t[col].fillna(val)

# AMT_ANNUITY追加
df_t["AMT_ANNUITY"] = df["AMT_ANNUITY"].copy()
val = df_t[df_t["TARGET"].notnull()]["AMT_ANNUITY"].mean() # trainデータ内で算出
df_t["AMT_ANNUITY"] = df_t["AMT_ANNUITY"].fillna(val)

# DAYS_EMPLOYED追加
df_t["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
val = df_t[df_t["TARGET"].notnull()]["DAYS_EMPLOYED"].mean()
df_t["DAYS_EMPLOYED"] = df_t["DAYS_EMPLOYED"].fillna(val)


# SK_ID_CURR削除
df_t = df_t.drop('SK_ID_CURR', axis=1)

# ワンホットエンコーディング
df_t = pd.get_dummies(
        df_t,
        drop_first=True,
        dummy_na=True
         )

#print("欠損値数:", df_t[df_t['TARGET'].notnull()].isnull().sum().sum())

# train-test分離
train = df_t[df_t['TARGET'].notnull()]
test = df_t[df_t['TARGET'].isnull()].drop('TARGET', axis=1)

# 保存
train.to_pickle("./preprocessed_data/train.pkl")
test.to_pickle("./preprocessed_data/test.pkl")

print("complete!")

