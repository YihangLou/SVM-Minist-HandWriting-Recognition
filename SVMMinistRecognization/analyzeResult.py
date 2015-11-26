import os
import numpy as np
f1= open('libsvm_predict result.txt','r')
total = 0
error = 0
error_mat = np.mat(np.zeros((10,10)))
while True:
    splitstr = f1.readline().strip().split()
    if not splitstr:
        break
    total +=1
    gt_label = splitstr[0][0]
    predic_label = splitstr[1]
    if gt_label != predic_label:
        error += 1
        error_mat[gt_label,predic_label] += 1
        

print 'right rate \n', 1-float(error)/float(total)
print 'error_mat \n', error_mat
    
