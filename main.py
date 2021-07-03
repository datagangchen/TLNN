from  network import *
from TLNetwork import *
from init import *
import random
import numpy as np 
from load_data import *
from statistics import mean
import time

Num_atom = 4

predicates =[]
atomics =[]
for i in range(Num_atom):
    predicates.append(generate_predicate())
    atomics.append(generate_atom())

 
and1  = AND(random_weight(Num_atom))
or1 = OR(random_weight(Num_atom))
and2 = AND(random_weight(2))

 
ANDOR= [and1,or1]
And  = and2 

tlnn = TLNN(predicates,atomics,ANDOR,And)

signals,_,labels= load_data('matlab/TLNN_train_norm.mat')



robustness =[]
start_time = time.time()
for i in range(1):
    overall_loss =[]
    overall_robust =[]
    time_cost =[]
    start_time = time.time()
    for i in range(20):
        eta = math.exp(-0.6*i-0.3)
        loss =[]
        robust =[]
        data =  list(zip(signals,labels))
        random.shuffle(data)
  
        for sig, label in data:
            out = tlnn.output(np.array(sig))
            tlnn.train(np.array(sig),np.array(label), eta, 1)
            loss.append(0.5*(out-label)**2)
            robust.append(out*label)
            txt = '---current output---', out*label, 'label', label,  'trainning step', i, 
            print txt


        overall_loss.append(mean(loss))
        if min(robust)>=0:
            show_result(tlnn)
        overall_robust.append(min(robust))
        time_cost.append(time.time()-start_time)

    robustness.append(overall_robust)

with open('robustness_inner.txt', 'w') as f:
    for item in overall_robust:
        f.write("%s\n" % item)

with open('robustness_inner_M.txt', 'w') as f:
    for item in robustness:
        f.write("%s\n" % item)


with open('timecost.txt', 'w') as f:
    for item in time_cost:
        f.write("%s\n" % item)


print overall_loss, '--excution time--', time.time()-start_time





 