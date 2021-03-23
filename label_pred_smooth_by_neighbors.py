import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import sys
sys.path.insert(0, './../../../source/')
import eval_cluster
import time

def load_feats(fn, feature_dim=256):
    return np.fromfile(fn, dtype=np.float32).reshape(-1, feature_dim)
	
#load features
data_name='unlabeled/omni'
#feats1 = load_feats('C:/Users/isetsk/Downloads/cdp-master/cdp-master/data/{}/features/{}.bin'.format(data_name, 'part3_test'),256)
#feats1 = load_feats('C:/Users/isetsk/Downloads/cdp-master/cdp-master/data/{}/features/{}.bin'.format(data_name, 'part9_test'),256)

knn1=np.load('./knn/avg_fr_k15.npz')
knn1_dist=knn1['dist']
knn1=knn1['idx']


#print(knn1_dist)

#extract 296 to 330 rows of the features
#296:331
#feats1=feats1[20730:20816]
#feats2=feats2[20730:20816]
#feats3=feats3[20730:20816]
#feats4=feats4[20730:20816]
#feats5=feats5[20730:20816]

#feats1=normalize(feats1,axis=1,norm='l2')
#print(np.shape(feats1))
				
#label=[3,3,3,3,3,3,4,3,3,3,3,3,3,-1,3,3,4,3,3,3,-1,3,4,4,3,3,4,3,3,4,3,4,4,3,3]
#label=np.arange(0,np.size(feats1,0))


label_true=meta=np.loadtxt('meta.txt')

#label_predict=np.loadtxt('C:/Users/isetsk/Downloads/cdp-master/cdp-master/experiments/emore_u200k_cmt4/output/k15_vote_accept4_th0.605/sz600_step0.05/meta.txt')
#label_predict=np.loadtxt('C:/Users/isetsk/Downloads/cdp-master/cdp-master/experiments/emore_u200k_cmt4/output/k15_vote_accept2_th0.605/sz600_step0.05/meta.txt')
label_predict=np.loadtxt('label_predict2.txt')

label_predict2=label_predict.copy()

print('unlabeled count='+str(np.sum(label_predict2==-1)))

start = time.time()
	
#remove class with only one item
#count=np.bincount(label_predict.astype(int)+1)
#print(count)
#print(np.array(np.where(count<=3))-1)
#label_predict2[np.isin(label_predict2,np.array(np.where(count<=3))-1)]=-1#3
#for i in range(0,len(count)):
#	if count[i]<=3:
#		label_predict2[label_predict2==(i-1)]=-1


#label_predict[:]=-1
num_neighbor=3
list=np.zeros((1,num_neighbor))
label_predict_count=np.bincount(label_predict.astype(int)+1)
print(np.sum(label_predict_count<=2))
print(np.sum(label_predict2==-1))
label_predict3=label_predict2
wh=np.squeeze(np.where(label_predict3==-1))
#print(np.shape(wh))
#print(len(wh))
for j in range(0,80):
	found_out_cast=False
	for i in wh:#range(0,np.size(label_predict3,0)):
		#if i%1000==999:
			#print('i='+str(i))
			#break
			
		#if label_predict2[i]!=-1 or (knn1_dist[i,3]+knn2_dist[i,3]+knn3_dist[i,3]+knn4_dist[i,3]+knn5_dist[i,3])/5>0.3:
		#if label_predict_count[label_predict[i].astype(int)+1]>1e10 or
		if label_predict3[i]!=-1 or (knn1_dist[i,1])>0.4:#0.8:#0.4
			continue
			
		#get all neighbours
		list[0,:]=label_predict3[knn1[i,1:(num_neighbor+1)]]
		
		#print(label_predict[knn1[i,1:15]])
		temp=np.ndarray.flatten(list)
		temp=temp.astype(int)
		index = np.argwhere(temp==-1)
		temp=np.delete(temp,index)
		#print(temp)
		count=np.bincount(temp+1)
		if len(count)>0:
			ind=np.argmax(count)
			#print(ind)
			if count[ind]>=1:
				label_predict2[i]=ind-1
				found_out_cast=True
			else:
				label_predict2[i]=-1
		else:
			label_predict2[i]=-1
		#print(ind)
	label_predict3=label_predict2
	if found_out_cast==False:
		break
print(np.sum(label_predict2==-1))

end = time.time()
print('time taken')
print(end - start)

pred_with_singular = label_predict.copy()
valid = np.where(label_predict != -1)
_, unique_idx = np.unique(label_predict[valid], return_index=True)
pred_unique = label_predict[valid][np.sort(unique_idx)]
num_class=len(pred_unique)
pred_with_singular[np.where(label_predict == -1)] = np.arange(num_class, num_class + (label_predict == -1).sum()) # to assign -1 with new labels

avg_pre,avg_rec,f_score=eval_cluster.fscore(label_true,pred_with_singular)
print('avg precision='+str(avg_pre))
print('avg recall='+str(avg_rec))
print('fscore='+str(f_score))

pred_with_singular = label_predict2.copy()
valid = np.where(label_predict2 != -1)
_, unique_idx = np.unique(label_predict2[valid], return_index=True)
pred_unique = label_predict2[valid][np.sort(unique_idx)]
num_class=len(pred_unique)
pred_with_singular[np.where(label_predict2 == -1)] = np.arange(num_class, num_class + (label_predict2 == -1).sum()) # to assign -1 with new labels

avg_pre,avg_rec,f_score=eval_cluster.fscore(label_true,pred_with_singular)
print('avg precision='+str(avg_pre))
print('avg recall='+str(avg_rec))
print('fscore='+str(f_score))

print('unlabeled count='+str(np.sum(label_predict2==-1)))