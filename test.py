import numpy as np

train_x =  np.load('./_save/x_train.npy',allow_pickle=True)
eval_x =  np.load('./_save/eval_x.npy',allow_pickle=True)
train_x= np.array(train_x)
eval_x = np.array(eval_x)
train_y =  np.load('./_save/train_y.npy',allow_pickle=True)
eval_y =  np.load('./_save/eval_y.npy',allow_pickle=True)
print(train_x.shape)
print(eval_x.shape)
print(train_y.shape)
print(eval_y.shape)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(max_depth=5,max_leaf_nodes=5,min_samples_leaf=3,min_samples_split=3)

model.fit(train_x, train_y)

result = model.score(eval_x, eval_y)
print(result)

y_pred  = model.predict(test_features)

sample_submission['label']=model.predict(test_features)
sample_submission.to_csv('./_save/rf_baseline.csv', index=False)
