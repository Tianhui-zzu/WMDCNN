'用SVM预测'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,mean_absolute_error,f1_score,confusion_matrix
from data_loader import data_load01,data_load02,data_load03
# data = sklearn.datasets.load_iris()
# X,y =data['data'],data['target']

X,y =data_load02()
# X,X_T,y,y_T = data_load03()
X = X.reshape(X.shape[0],-1)
print(X)
# 生成模型并传入数据和标签进行训练
rf_clf = RandomForestClassifier(n_estimators=200,n_jobs=-1,criterion='gini',min_samples_split=2,min_samples_leaf=1, min_weight_fraction_leaf=0, max_features="sqrt").fit(X,y)

# 训练
# SVC()

# 预测
y_p = rf_clf.predict(X)
print('knn_clf.predict',y_p)
print('y',y)

print('accuracy',accuracy_score(y_true=y,y_pred=y_p))
confusion_matrix = confusion_matrix(y_true=y,y_pred=y_p)
print('confusion\n',confusion_matrix)
confusion_matrix_normalization  = []
for col in confusion_matrix:
    confusion_matrix_normalization.append(np.around(col/col.sum(),2))
confusion_matrix_normalization = np.array(confusion_matrix_normalization)
print(confusion_matrix_normalization)
plt.matshow(confusion_matrix_normalization , cmap=plt.cm.Blues)

# 在对应的位置上标记上混淆矩阵的具体值
for i in range(len(confusion_matrix_normalization)):
    for j in range(len(confusion_matrix_normalization)):
        plt.annotate(confusion_matrix_normalization[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
# 将横纵坐标上的数字标签等价替换为气体类别
# plt.xticks(range(len(confusion_matrix_normalization)),['CO' ,'Ey', 'Ea', 'Me'])
# plt.yticks(range(len(confusion_matrix_normalization)),['CO' ,'Ey', 'Ea', 'Me'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
# 通过调用 colorbar 函数来加上数值和颜色的对应规则
plt.colorbar()
plt.title('the confusion matrix of RF')
plt.show()