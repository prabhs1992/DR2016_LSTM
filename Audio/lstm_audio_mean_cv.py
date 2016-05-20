from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import os
from collections import defaultdict
import random

validation_runs = 51
valid_cnt = 5
test_cnt = 5
limit = 36

cv_parameters = defaultdict(list)

for j in range(validation_runs):
    init_scale = 0.1 if j==0 else random.randint(4,10)*0.01
    max_grad_norm = 5 if j==0 else random.randint(5,10)
    max_epoch = 4 if j==0 else random.randint(4,14)
    max_max_epoch = 1 if j==0 else random.randint(13,55)
    keep_prob = 1.0 if j==0 else random.randint(35,100)*0.01
    lr_decay = 0.5 if j==0 else random.randint(50,89)*0.01
    num_layers = 1 if j==0 else random.randint(1,2)
    print('validation #',j+1)
    cv_parameters['init_scale'].append(init_scale)
    cv_parameters['max_grad_norm'].append(max_grad_norm)
    cv_parameters['max_epoch'].append(max_epoch)
    cv_parameters['max_max_epoch'].append(max_max_epoch)
    cv_parameters['keep_prob'].append(keep_prob)
    cv_parameters['lr_decay'].append(lr_decay)
    cv_parameters['num_layers'].append(num_layers)
    command = "python ./validator_audio.py --valid_cnt "+str(valid_cnt)+" --test_cnt "+str(test_cnt)+" --limit "+str(limit)+" --vt "+str(1)+" --init_scale "+str(init_scale)+" --max_grad_norm "+str(max_grad_norm)+" --max_epoch "+str(max_epoch)+" --max_max_epoch "+str(max_max_epoch)+" --keep_prob "+str(keep_prob)+" --lr_decay "+str(lr_decay)+" --num_layers "+str(num_layers)
    os.system(command)
    import time
    time.sleep(2)
    with open('messagepass.txt', 'r') as fh:
        f_acc = fh.read()
    cv_parameters['accuracy'].append(float(f_acc))

#final run with best parameters for training

index, value = max(enumerate(cv_parameters['accuracy']), key=operator.itemgetter(1))
print('max value of framewise accuracy on validation data = %.3f' % value)
print('best value of init_scale on validation data = %.3f' % cv_parameters['init_scale'][index])
print('best value of max_grad_norm on validation data = %.3f' % cv_parameters['max_grad_norm'][index])
print('best value of max_epoch on validation data = %.3f' % cv_parameters['max_epoch'][index])
print('best value of max_max_epoch on validation data = %.3f' % cv_parameters['max_max_epoch'][index])
print('best value of keep_prob on validation data = %.3f' % cv_parameters['keep_prob'][index])
print('best value of lr_decay on validation data = %.3f' % cv_parameters['lr_decay'][index])
print('best value of num_layers on validation data = %.3f' % cv_parameters['num_layers'][index])
print('\n\n')
print('****************Running on test data***********************')
command = "python ./validator_audio.py --valid_cnt "+str(valid_cnt)+" --test_cnt "+str(test_cnt)+" --limit "+str(limit)+" --vt "+str(2)+" --init_scale "+str(cv_parameters['init_scale'][index])+" --max_grad_norm "+str(cv_parameters['max_grad_norm'][index])+" --max_epoch "+str(cv_parameters['max_epoch'][index])+" --max_max_epoch "+str(cv_parameters['max_max_epoch'][index])+" --keep_prob "+str(cv_parameters['keep_prob'][index])+" --lr_decay "+str(cv_parameters['lr_decay'][index])+" --num_layers "+str(cv_parameters['num_layers'][index])
os.system(command)
with open('messagepass.txt', 'r') as fh:
    f_acc = fh.read()
