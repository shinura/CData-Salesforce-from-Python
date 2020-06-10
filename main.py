import cdata.salesforce as mod
import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
##### データ取得 ##### 

# 接続文字列
CONNECTION_STRING = "User=your-user-name;Password=your-password;SecurityToken=your-security-token;"
columns = ['Id','LeadSource','Status','Industry','Rating','AnnualRevenue','NumberOfEmployees']

# Salesforceからリードデータを取得
sql = "SELECT Id,LeadSource,Status,Industry,Rating,AnnualRevenue,NumberOfEmployees FROM Lead WHERE Country='Japan'"
conn = mod.connect(CONNECTION_STRING)
df = pd.read_sql_query(sql,conn)
conn.close()
df.columns = columns


##### データクレンジング ##### 

# ステータス、レーティングのカテゴリ変数を変換
df['Status'] = df['Status'].map( {'Closed - Converted': 1, 'Closed - NotConverted': 0, 'Working - Contacted': -1} ).astype(int)
df['Rating'] = df['Rating'].map( {'Hot': 1, 'Warm': 0.5, 'Cold': 0} ).astype(int)

# リードソース、業種をOne-Hotベクトル化
lead_source = pd.get_dummies(df["LeadSource"], drop_first=True, prefix="LeadSource")
industory = pd.get_dummies(df["Industry"], drop_first=True, prefix="Industry")
df = pd.concat([df, lead_source,industory], axis=1)
df.drop(['LeadSource','Industry'], axis=1, inplace=True)

# 学習データを取得
# ステータスがClosedになっているものを学習データとする
train = df[df.Status > -1]

# Id列を削除
train.drop("Id", axis=1, inplace=True)

# 年間売上、従業員数を標準化
col_names = ["AnnualRevenue","NumberOfEmployees"]
features = train[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
train[col_names] = features

# 特徴量と目的変数とに分ける
X = train.drop("Status", axis=1)
Y = train["Status"]


##### モデル構築 ##### 

# モデルフィッティング
# ロジスティック回帰モデルを適用
model = LogisticRegression()
model.fit(X,Y)

# スコアを計算
score = model.score(X,Y)
print('Score:'+ str(score))


##### 予測 #####

# 予測対象として、クローズされていないレコードを抽出
target = df[df.Status == -1]

# 年間売上、従業員数を標準化
features = target[col_names]
target[col_names] = scaler.transform(features.values)

# Id列を分離(後で予測結果と結合)
ids = target["Id"]
X = target.drop(["Id","Status"], axis=1)

# 予測
pred = model.predict(X)

# 予測結果をId列と結合
pred = pd.concat([pd.Series(pred, name='PredictedStatus'), ids], axis=1)


##### データ出力 #####

# 予測結果のラベルを設定
pred['PredictedStatus'] = pred['PredictedStatus'].map({1 : 'Likely converted', 0 : 'Not likely converted'})

# Salesforceのリードへ予測結果を出力
conn = mod.connect(CONNECTION_STRING)
sql = "UPDATE Lead SET PredictedStatus__c = ? WHERE Id = ?"
params = pred.values
cur = conn.cursor()
cur.executemany(sql, params)
conn.close()


