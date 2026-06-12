import os
import  numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import data_load01,data_load02,data_load03

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf



feature_vector_length = 256
batch_size = 20



def InputFun(X,y,bs):
        #
        X  = pd.DataFrame(X,columns=[str(i) for i in range(0,X.shape[1])])
        y  = pd.DataFrame(y)
        # print('X.keys()',X.keys())
        # # print('dict(X)',dict(X))
        # print('y',y)
        Datasets = tf.data.Dataset.from_tensor_slices((dict(X),y))
        return Datasets.batch(bs)

def InputFunPredict(X):
     return tf.data.Dataset.from_tensor_slices(dict(X)) # 对数据加以加载,以每次BatchSize个输出
# 加载数据
X,X_T,y,y_T = data_load03()
X = X.reshape(X.shape[0],-1)
# print(X.shape)
# 生成特征索引列
FeatureColumn = []
for key  in range(X.shape[1]):
        FeatureColumn.append(tf.feature_column.numeric_column(key=str(key)))

# print(FeatureColumn)

dnn_clf = tf.estimator.DNNClassifier(feature_columns=FeatureColumn,hidden_units=[feature_vector_length,100],n_classes=4,activation_fn=tf.nn.relu)


# 训练
dnn_clf.train(input_fn=lambda:InputFun(X,y,batch_size),steps=200)
# 预测
# dataset = InputFun(X,y,batch_size)
Predict_Results = dnn_clf.predict(input_fn=lambda:InputFunPredict(X_T))
# 打印验证结果

for k in Predict_Results:
        print(k)
