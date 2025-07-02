#train and test set
xtrain = combined_stats.iloc[:304,3:]
ytrain = combined_stats['result'].loc[:303]

xtest = combined_stats.iloc[304:,3:]
ytest = combined_stats['result'].loc[304:]


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

#encode A,D,H to 0,1,2
le = LabelEncoder()
y_train_encoded = le.fit_transform(ytrain)
y_train_encoded = pd.DataFrame(y_train_encoded)


#train model
model = XGBClassifier(
    objective='multi:softprob',  # for multiclass classification
    num_class=3,                 # 3 possible outcomes
    eval_metric='mlogloss',     # common for multi-class
)

model.fit(xtrain, y_train_encoded)
