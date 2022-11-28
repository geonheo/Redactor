import warnings
from preprocessing import Dataset
import numpy as np
from model import SurrogateModels
import time
from prob_decision_boundary import PDB
from sklearn.metrics import f1_score, accuracy_score
from gen_disinfos import GANcandidates
from gen_disinfos import WMcandidates, agg_disinfo
from tqdm import tqdm
import pandas as pd
import sys
import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"

warnings.filterwarnings(action='ignore')

np.random.seed(227)

test_size = 0.25
n_target = 10
n_sample = int(sys.argv[1])
ti = int(sys.argv[2])
exp_type = int(sys.argv[3])

Diabetes = Dataset('diabetes')
target_indices = ['i18400', 'i7016', 'i45442', 'i32134', 'i55275', 'i42916', 'i64244', 'i58296', 'i27785', 'i15937']
#np.random.choice(Diabetes.data.index, n_target)
print(target_indices)
target_indices = [target_indices[ti]]
print(target_indices)

target_label = Diabetes.label.loc[target_indices].values[0,0]
ratio_list = np.zeros(2)

if exp_type == 5:
    ratio_list[target_label] = 0.7
    ratio_list[1-target_label] = 0.3
    if n_sample <= 30:
        ratio_list[target_label] -= 0.05
        ratio_list[1-target_label] += 0.05
elif exp_type == 7:
    ratio_list[0] = 0.7
    ratio_list[1] = 0.3
        
else:
    ratio_list[target_label] = 0.9
    ratio_list[1-target_label] = 0.1
    

if exp_type == 1:
    Diabetes.nearest_sampling(n_sample, target_indices[0], distribution=True)#, ratio_list=ratio_list)
elif exp_type == 2 or exp_type == 7:
    Diabetes.nearest_sampling(n_sample, target_indices[0], distribution=True, ratio_list=ratio_list)
elif exp_type == 4 or exp_type == 5:
    
    D = Dataset('diabetes')
    D.nearest_sampling(int(n_sample/0.75), target_indices[0])
    
    th = min(ratio_list)
    classes = np.unique(D.label)
    cratio = np.array([(D.label == c).sum().values[0] for c in classes])
    cratio = cratio/cratio.sum()
    print('before: ',cratio)

    if cratio[1-target_label] < ratio_list[1-target_label]:
        print('ratio: ',ratio_list)
        Diabetes.nearest_sampling(int(n_sample/0.75), target_indices[0], distribution=True, ratio_list=ratio_list)
    else:
        Diabetes.nearest_sampling(int(n_sample/0.75), target_indices[0])
        
    classes = np.unique(Diabetes.label)
    cratio = np.array([(Diabetes.label == c).sum().values[0] for c in classes])
    cratio = cratio/cratio.sum()
    print('after: ', cratio)
elif exp_type == 6:
    Diabetes.nearest_sampling(int(n_sample/0.75), target_indices[0])

if exp_type == 3:
    Diabetes_repeat = Dataset('diabetes')
    Diabetes_repeat.nearest_sampling(n_sample, target_indices[0], distribution=True, ratio_list=ratio_list, oversampling=True)
    (x_tr_r,y_tr_r), (x_te_r,y_te_r), (x_ta_r,y_ta_r), tr_scaler = Diabetes_repeat.split_dataset(test_size, target_indices)

    s_models = SurrogateModels()
    s_models.train_all(x_tr_r, y_tr_r)
    perf = s_models.show_performance([(x_tr_r,y_tr_r), (x_te_r,y_te_r), (x_ta_r,y_ta_r)],
                             cnames=['train', 'test','target'])
    
    Diabetes.nearest_sampling(n_sample, target_indices[0], distribution=True, ratio_list=ratio_list)
    (x_tr,y_tr), (x_te,y_te), (x_ta,y_ta), tr_scaler = Diabetes.split_dataset(test_size, target_indices)
    
    start_time = time.time()
    print(start_time)

    a = [l[2:] for l in perf[perf['target acc']==1]['test acc'].sort_values()[-6:].index]
    s_models = SurrogateModels(a)
    s_models.train_all(x_tr_r, y_tr_r)
    
else:
    (x_tr,y_tr), (x_te,y_te), (x_ta,y_ta), tr_scaler = Diabetes.split_dataset(test_size, target_indices)

    s_models = SurrogateModels()
    s_models.train_all(x_tr, y_tr)
    perf = s_models.show_performance([(x_tr,y_tr), (x_te,y_te), (x_ta,y_ta)],
                             cnames=['train', 'test','target'])

    start_time = time.time()
    print(start_time)

    a = [l[2:] for l in perf[perf['target acc']==1]['test acc'].sort_values()[-6:].index]
    s_models = SurrogateModels(a)
    s_models.train_all(x_tr, y_tr)


prob_dec = PDB(s_models.models)
x_all = np.concatenate([x_tr, x_te], axis=0)
prob_dec.fit_all(x_all)

t1 = time.time()

D = Dataset('diabetes')
D.nearest_sampling(10000, target_indices[0])

x_nn = tr_scaler.transform(D.data_1hot)
y_nn = D.label.values[:,0]

sn_te_labels = prob_dec.predict(x_nn)
sn_te_labels[sn_te_labels == -1] = 0
te_acc = sum(sn_te_labels==y_nn)/len(y_nn)

print('%s_%s acc and f1' % (target_indices[0], n_sample))
print(accuracy_score(y_nn, sn_te_labels)*100, f1_score(y_nn, sn_te_labels)*100)

testtime = time.time()-t1

diabetes = Diabetes.data
column_cat = Diabetes.column_cat
column_int = Diabetes.column_int
columns_1hot = Diabetes.data_1hot.columns

gan_gen = GANcandidates()
gan_gen.fit(diabetes, column_cat, column_int)


_ = gan_gen.generate()
gan_cand_list = gan_gen.nearest_points(tr_scaler, target_indices, columns_1hot)

  
diabetes_1hot = Diabetes.data_1hot
diabetes_label = Diabetes.label

wm_gen = WMcandidates(diabetes_1hot, diabetes_label, target_indices)
wm_cand_list = wm_gen.watermarking(tr_scaler, diabetes.columns, column_cat, column_int)

Diabetes_entire = Dataset('diabetes')
(x_tra,y_tra), (x_tea,y_tea), (x_taa,y_taa), tra_scaler = Diabetes_entire.split_dataset(test_size, target_indices)

x_dis, y_dis = [], []

target_i = target_indices[0]
#xt, yt = [Diabetes.data_1hot.loc[target_i]], y_taa[ti]
xt, yt = x_taa[0], y_taa[0]
wm_cand = wm_cand_list[0]
gan_cand = gan_cand_list[0]
candidates = pd.concat((wm_cand, gan_cand))

x_tmp, y_tmp = agg_disinfo(prob_dec, candidates, tra_scaler, x_tra, y_tra, xt, yt, 
                           columns_1hot, n_disinfo=200)
x_dis.extend(x_tmp)
y_dis.extend(y_tmp)

np.save('disinfos/%s_%s_x_%s.npy' % (target_indices[0], n_sample, exp_type), np.array(x_dis))
np.save('disinfos/%s_%s_y_%s.npy' % (target_indices[0], n_sample, exp_type), np.array(y_dis))

finish_time = time.time()
print('%s_%s' % (target_indices[0], n_sample))
print('start : %s s, finish : %s s' % (start_time, finish_time))
print('execution time : %s' % (finish_time-start_time-testtime))





