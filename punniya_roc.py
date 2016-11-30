import pandas as pd
import numpy
df=pd.read_csv("C:\\Users\\Pavan\\Desktop\\with_pca.csv")
print df.columns
x=pd.DataFrame()
y=pd.DataFrame()
x['Actual']=df['Actual']
y['Predicted']=df['Predicted']
print(x['Actual'])
print(y)

df2=pd.read_csv("C:\\Users\\Pavan\\Desktop\\without_pca.csv")
print df.columns

x2=pd.DataFrame()
y2=pd.DataFrame()
x2['Actual']=df2['Actual']
y2['Predicted']=df2['Predicted']

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

actual_withPCA=x['Actual']
predicted_withPCA=y['Predicted']

predicted_withoutPCA = y2['Predicted']
actual_withoutPCA=x2['Actual']

fpr, tpr, threshold = roc_curve(actual_withoutPCA, predicted_withoutPCA)
roc_auc2 = auc(fpr, tpr)

false_positive_rate_PCA, true_positive_rate_PCA, thresholds_PCA = roc_curve(actual_withPCA, predicted_withPCA)
roc_auc = auc(false_positive_rate_PCA, true_positive_rate_PCA)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate_PCA, true_positive_rate_PCA, 'brown',lw=2, label='logistic regression with PCA Area= %0.2f'% roc_auc)
plt.plot(fpr, tpr,color='darkorange',lw=2, label='logistic regression without PCA area = %0.2f'% roc_auc2)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'k--',lw=2)
#plt.xlim([-0.1,1.2])
#plt.ylim([-0.1,1.2])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
