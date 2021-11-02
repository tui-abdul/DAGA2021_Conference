import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, precision_recall_curve
import seaborn as sns
from sklearn.metrics import f1_score
LABELS = ["Normal","Faulty"]
width = 0.01
df = pd.read_excel('mse_fix_error_dist_order.xlsx', engine='openpyxl')
plt.rcParams.update({'font.size': 14})


a = df[df['mse_train']>=0.54].index
print('train',a) 

c = df[df['mse_test']>=0.54].index
print('test',c)
mse_test=df['mse_test'].to_numpy()
mse_train=df['mse_train'].to_numpy()

mse_test = mse_test[~np.isnan(mse_test)]


print('mse_test_min',mse_test.min())
print('mse_test_max',mse_test.max())

print('mse_train_min',mse_train.min())
print('mse_train_max',mse_train.max())



bins_test = math.ceil((mse_test[:6].max() - mse_test[:6].min())/width)
bins_test_1 = math.ceil((mse_test[7:].max() - mse_test[7:].min())/width)
bins_train = math.ceil((mse_train.max() - mse_train.min())/width)

print('bins_test',bins_test)

    
plt.hist(mse_test[:6], bins=4, density=True, label="Test Set Fault", alpha=0.75)


    
plt.title("MSE Distribution")
plt.legend(loc='upper right')
plt.xlabel('Reconstruction Error')
plt.ylabel('Occurrences')


plt.hist(mse_test[7:], bins=4, density=True, label="Test Set Normal", alpha=0.75)
  
plt.title("MSE Distribution")
plt.legend(loc='upper right')
plt.xlabel('Reconstruction Error')
plt.ylabel('Occurrences')


plt.title("MSE Distribution")
plt.legend(loc='upper right')
plt.xlabel('Reconstruction Error')
plt.ylabel('Occurrences')

plt.show()
plt.close()#

total_recons = np.concatenate((mse_test[:7], mse_test[7:]), axis=0)

print(total_recons)
print('mse_test.shape',mse_test.shape)
y_test_tr = np.zeros(mse_train.shape)
y_test_ts = np.ones(mse_test[:7].shape)
y_test_ta = np.zeros(mse_test[7:].shape)

y_test = np.concatenate((y_test_ts,y_test_ta),axis=0)
print(y_test)

fpr, tpr, thresholds = roc_curve(y_test, total_recons)
roc_auc = auc(fpr, tpr)
print(roc_auc)
lw = 2
plt.plot(fpr,tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

precision_rt, recall_rt, threshold_rt = precision_recall_curve(y_test,total_recons)

pr_auc = auc(recall_rt, precision_rt)


plt.plot(recall_rt, precision_rt, linewidth=5, label='AUC = %0.3f'% pr_auc)
plt.legend(loc='upper right')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()



plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

threshold_fixed = 0.166
pred_y = [1 if e > threshold_fixed else 0 for e in total_recons]
conf_matrix = confusion_matrix(y_test, pred_y)


sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d",cmap='Greys');
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
print('pred_y',pred_y)
f1_score(y_test, pred_y, average='macro')

