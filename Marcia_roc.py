from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
import csv
import numpy

filename ='C:\\Users\\Pavan\\Desktop\\fall17\\ml\\roc\\1.csv'
raw_data = open(filename, 'r')
#I don't remember where I found the following code, but it works.
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x_1 = list(reader)
dataset = numpy.array(x_1).astype('float')
print(dataset.shape)
# separate the data from the target attributes
# using X=0:14 somehow made us miss the last column, its a python thing.
# but X and Y are correct as defined below, as I double checked against the csv
predicted_1 = dataset[:,1]
actual_1= dataset[:,0]
print(predicted_1)
print(actual_1)
filename ='C:\\Users\\Pavan\\Desktop\\fall17\\ml\\roc\\2.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x_2 = list(reader)
dataset = numpy.array(x_2).astype('float')
print(dataset.shape)
predicted_2 = dataset[:,1]
actual_2= dataset[:,0]
print(predicted_2)
print(actual_2)
filename ='C:\\Users\\Pavan\\Desktop\\fall17\\ml\\roc\\3.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x_3 = list(reader)
dataset = numpy.array(x_3).astype('float')
print(dataset.shape)
predicted_3 = dataset[:,1]
actual_3= dataset[:,0]
print(predicted_3)
print(actual_3)

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

filename ='C:\\Users\\Pavan\\Desktop\\fall17\\ml\\roc\\5.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x_5 = list(reader)
dataset = numpy.array(x_5).astype('float')
print(dataset.shape)
predicted_5 = dataset[:,1]
actual_5= dataset[:,0]
print(predicted_5)
print(actual_5)


filename ='C:\\Users\\Pavan\\Desktop\\fall17\\ml\\roc\\6.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x_6 = list(reader)
dataset = numpy.array(x_6).astype('float')
print(dataset.shape)
predicted_6 = dataset[:,1]
actual_6= dataset[:,0]
print(predicted_6)
print(actual_6)

fpr_6, tpr_6, threshold_6 = roc_curve(actual_6,predicted_6)
roc_NBHybrid_Pairs_SMExpert_WithEDS = auc(fpr_6, tpr_6)
plt.plot(fpr_6, tpr_6,color='violet',lw=2, label='NBHybrid_Pairs_SMExpert_WithEDS area = %0.2f'% roc_NBHybrid_Pairs_SMExpert_WithEDS)


fpr_5, tpr_5, threshold_5 = roc_curve(actual_5,predicted_5)
roc_NBHybrid_Pairs_SMExpert_NoEDS = auc(fpr_5, tpr_5)
plt.plot(fpr_5, tpr_5,color='blue',lw=2, label='NBHybrid_Pairs_SMExpert_NoEDS area = %0.2f'% roc_NBHybrid_Pairs_SMExpert_NoEDS)

fpr_1, tpr_1, threshold_1 = roc_curve(actual_1,predicted_1)
roc_NB_Pairs_Cont_SMExpert = auc(fpr_1, tpr_1)
plt.title('Receiver Operating Characteristic')

fpr_4, tpr_4, threshold_4 = roc_curve(actual_4,predicted_4)
roc_NB_Pairs_PCA_Factors = auc(fpr_4, tpr_4)
plt.plot(fpr_4, tpr_4,color='brown',lw=2, label='NB_Pairs_PCA_Factors area= %0.2f'% roc_NB_Pairs_PCA_Factors)


fpr_3, tpr_3, threshold_3 = roc_curve(actual_3,predicted_3)
roc_NB_Pairs_Just_EDSPer = auc(fpr_3, tpr_3)
plt.plot(fpr_3, tpr_3,color='green',lw=2, label='NB_Pairs_Just_EDSPer area = %0.2f'% roc_NB_Pairs_Just_EDSPer)


fpr_2, tpr_2, threshold_2 = roc_curve(actual_2,predicted_2)
roc_NB_Pairs_Cont_SMExpert_NoEDS = auc(fpr_2, tpr_2)
plt.plot(fpr_2, tpr_2,color='red',lw=2, label='NB_Pairs_Cont_SMExpert_NoEDS area = %0.2f'% roc_NB_Pairs_Cont_SMExpert_NoEDS)




plt.plot(fpr_1, tpr_1,color='orange',lw=4, linestyle=':', label='roc_NB_Pairs_Cont_SMExpert area = %0.2f'% roc_NB_Pairs_Cont_SMExpert)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'k--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()