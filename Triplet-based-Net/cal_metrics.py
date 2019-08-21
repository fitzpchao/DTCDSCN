import numpy as np
import os
from sklearn.metrics import confusion_matrix
num=762
path='/mnt/lustre/pangchao/cd_paper/code_v2.0/cexp4/tripletloss/'
matrixs = np.zeros([2,2],np.float32)
for i in range(num):
    print(str(i)+'/'+str(num))
    dist = np.load(path + str(i)+'_dist.npy')

    target = np.load(path + str(i) + '_target.npy')
    dist[dist>0.5]=1
    dist[dist<=0.5]=0
    dist = np.reshape(dist,[-1]).astype(np.uint8)
    target = np.reshape(target,[-1]).astype(np.uint8)
    labels = list(set(np.concatenate((target, dist), axis=0)))
    if (labels == [0]):
        matrixs[0, 0] += confusion_matrix(target, dist)[0, 0]
    elif (labels == [1]):
        matrixs[1, 1] += confusion_matrix(target, dist)[0, 0]
    else:
        matrixs += confusion_matrix(target, dist)
a, b, c, d = matrixs[0][0], matrixs[0][1], matrixs[1][0], matrixs[1][1]
print(matrixs)
accuracy = (a + d) / (a + b + c + d)
if ((d + c) != 0):
    recall = d / (d + c)
else:
    recall = 0
if ((d + b) != 0):
    precision = d / (d + b)
else:
    precision = 0

F1 = 2 * d / (a + b + c + d + d - a)
if ((d + b + c) != 0):
    iou = d / (c + b + d)
else:
    iou = 0
print('iou:',iou)
print('precision:',precision)
print('recall:',recall)
print('F1:',F1)



