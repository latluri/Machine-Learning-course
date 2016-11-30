from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
import csv
import numpy

filename ='C:\\Users\\Pavan\\Desktop\\fall17\\ml\\roc\\4.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x_4 = list(reader)
dataset = numpy.array(x_4).astype('float')
print(dataset.shape)
predicted_4 = dataset[:,1]
actual_4= dataset[:,0]
print(predicted_4)
print(actual_4)

fpr_4, tpr_4, threshold_4 = roc_curve(actual_4,predicted_4)
roc_NB_Pairs_PCA_Factors = auc(fpr_4, tpr_4)

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

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
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

actual_withPCA=x['Actual']
predicted_withPCA=y['Predicted']
false_positive_rate_PCA, true_positive_rate_PCA, thresholds_PCA = roc_curve(actual_withPCA, predicted_withPCA)
roc_auc = auc(false_positive_rate_PCA, true_positive_rate_PCA)
#plt.title('Receiver Operating Characteristic')





plt.plot(fpr_4, tpr_4,color='brown',lw=2, label='NB_Pairs_PCA_Factors area = %0.2f'% roc_NB_Pairs_PCA_Factors)

false_positive_rate_PCA, true_positive_rate_PCA, thresholds_PCA = roc_curve(actual_withPCA, predicted_withPCA)
roc_auc = auc(false_positive_rate_PCA, true_positive_rate_PCA)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate_PCA, true_positive_rate_PCA, 'brown',lw=2, label='logistic regression with PCA Area= %0.2f'% roc_auc)
plt.plot(fpr, tpr,color='darkorange',lw=2, label='logistic regression without PCA area = %0.2f'% roc_auc2)
plt.plot(false_positive_rate_PCA, true_positive_rate_PCA, 'blue',lw=2, label='logistic regression with PCA Area= %0.2f'% roc_auc)

plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'k--',lw=2)
#plt.xlim([-0.1,1.2])
#plt.ylim([-0.1,1.2])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
