'用SVM预测'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,mean_absolute_error,f1_score,confusion_matrix
from data_loader import data_load01,data_load02,data_load03
# data = sklearn.datasets.load_iris()
# X,y =data['data'],data['target']

X,y =data_load02()
# X,X_T,y,y_T = data_load03()
X = X.reshape(X.shape[0],-1)
# print(y)

# 生成模型并传入数据和标签进行训练
svm_clf = SVC(kernel='linear',gamma=0.01,C=1,degree=3,coef0=0.0,shrinking=True,probability=False,tol=1e-3, cache_size=200, max_iter=-1, decision_function_shape='ovr').fit(X,y)


y_p = svm_clf.predict(X)
print('svm_clf.predict',y_p)
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
plt.title('the confusion matrix of SVM')
plt.show()