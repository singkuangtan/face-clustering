import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import sys
sys.path.insert(0, './../../../source/')
import eval_cluster
from sklearn.cluster import KMeans
sys.path.insert(0, 'C:/Users/isetsk/Downloads/cdp-master/cdp-master/data/unlabeled/emore_u200k')
#sys.path.insert(0, 'C:/Users/isetsk/Downloads/cdp-master/cdp-master/data/unlabeled/emore_u200k/sklearn_sk/utils')
from sklearn_sk.cluster import SpectralClustering
import time

#import spectral

def load_feats(fn, feature_dim=256):
    return np.fromfile(fn, dtype=np.float32).reshape(-1, feature_dim)
	
#load features
data_name='unlabeled/omni'
feats1 = load_feats('C:/Users/isetsk/Downloads/cdp-master/cdp-master/data/{}/features/{}.bin'.format(data_name, 'fr'),512)
#feats1 = load_feats('C:/Users/isetsk/Downloads/cdp-master/cdp-master/data/{}/features/{}.bin'.format(data_name, 'part9_test'),256)
				
knn1=np.load('./knn/fr_k15.npz')
#knn1=np.load('./knn/part9_test_k15.npz')
knn1_dist=knn1['dist']
knn1=knn1['idx']

feats1=normalize(feats1,axis=1,norm='l2')
print(np.shape(feats1))

label_true=meta=np.loadtxt('meta.txt')

label_predict=np.loadtxt('C:/Users/isetsk/Downloads/cdp-master/cdp-master/experiments/omni/output/k15_vote_accept0_th0.3/sz600_step0.05/meta.txt')
#label_predict=np.loadtxt('C:/Users/isetsk/Downloads/cdp-master/cdp-master/experiments/affinity_single/output/k15_vote_accept0_th0.7/sz1200_step0.05/meta.txt')
#label_predict=np.loadtxt('C:/Users/isetsk/Downloads/cdp-master/cdp-master/experiments/emore_u200k_single/output/k15_vote_accept0_th0.66/sz600_step0.05/meta.txt')
#label_predict[label_predict==-1]=temp[label_predict==-1]

label_predict2=label_predict.copy()

#label_predict2[label_predict2==-1]=label_true[label_predict2==-1]+np.max(label_predict2)+1
#label_predict2[label_predict2==-1]=np.max(label_predict2)+1
count=np.bincount(label_predict.astype(int)+1)
print(count)
#remove class with only one item
#for i in range(0,len(count)):
#	if count[i]<=3:
#		label_predict2[label_predict2==(i-1)]=-1



start = time.time()

#'''split cluster by spectral clustering
#for every clusters
next_label=np.max(label_predict2)+1
next_label=next_label+1
label_no_process=np.zeros(int(next_label))

max_cluster_size=100 #150
threshold=0.85 #0.85

count=np.bincount(label_predict2[label_predict2>=0].astype(int))
count=np.append(count,[0])
label_no_process[count<=max_cluster_size]=1
for j in range(0,3):
	print('iter='+str(j))
	for i in range(0,next_label.astype(int)):	
		if i<len(label_no_process):
			if label_no_process[i]==1:
				continue
		
		#if split cluster has higher performance
		sel=(label_predict2==i)
		
		label_size=(np.sum(sel))
		#print(label_size)
		
		if label_size<=max_cluster_size:
			if i<len(label_no_process):
				label_no_process[i]=1
			continue
		
		#print(np.shape(feats1))
		#print(np.shape(sel))
		features1=feats1[sel,:]
		
		
		#if np.size(features1,0)<=5:
		#	continue
		
		
		sc=SpectralClustering(n_clusters=5,random_state=0)
		#print(sc)
		clustering1, lambdas1= sc.fit(features1)#,assign_labels="discretize"
		#clustering1, lambdas1= SpectralClustering(n_clusters=5,random_state=0).fit(features1)#,assign_labels="discretize"
		
		
		#print(clustering.labels_)
		#exit(1)
		
		lambdas1=np.abs(lambdas1[::-1])
		
		lambdas=lambdas1
		clustering=clustering1
		
		#lambdas=lambdas[:-1]
		#diff=lambdas[1:]-lambdas[0:-1]
		#ind=np.argmax(diff)+1
		#print(ind+1)
		#print(lambdas[1:5]-lambdas[0:4])
		#exit(1)
		
		'''
		if np.abs(lambdas[2]-lambdas[1])>0.2 and label_size>200:
		#if ind==3 and label_size>200:
			print(diff)
			print('lambdas'+str(lambdas))
			print('cluster_id='+str(i))
			print('label size='+str(label_size))

			clustering, lambdas= SpectralClustering(n_clusters=3,assign_labels="discretize",random_state=0).fit(features)
			
			#split the cluster
			temp_label=clustering.labels_
			temp_label[temp_label==0]=i
			temp_label[temp_label==1]=next_label.copy()
			next_label=next_label+1
			label_predict2[label_predict2==i]=temp_label
		'''
		#if np.abs(lambdas[2]-lambdas[1])>0.05 and label_size>200:#0.05
		if np.abs(lambdas[1])<threshold and label_size>max_cluster_size:
			print('lambdas'+str(lambdas))
			print('cluster_id='+str(i))
			print('label size='+str(label_size))

			if np.abs(lambdas[4])<threshold:
				n_clusters=5
			elif np.abs(lambdas[3])<threshold:
				n_clusters=4
			elif np.abs(lambdas[2])<threshold:
				n_clusters=3
			elif np.abs(lambdas[1])<threshold:
				n_clusters=2
			elif np.abs(lambdas[0])<threshold:
				n_clusters=1
			clustering, lambdas= SpectralClustering(n_clusters=n_clusters,assign_labels="discretize",random_state=0).fit(features1)
			
			#split the cluster
			temp_label=clustering.labels_
			temp_label[temp_label==0]=i
			for k in range(1,n_clusters):
				temp_label[temp_label==k]=next_label.copy()
				next_label=next_label+1
			label_predict2[label_predict2==i]=temp_label
			
		else:
			if i<len(label_no_process):
				label_no_process[i]=1
		
		#else:
		#	print('lambdas,'+str(lambdas))
		#	print('cluster_id,='+str(i))
		#	print('label size,='+str(label_size))
	#'''
#print(label_predict2[label_predict==2949])

end = time.time()
print('time taken')
print(end - start)
	
'''split cluster by kmeans
#for every clusters
next_label=np.max(label_predict)+1
for i in range(0,np.max(label_predict).astype(int)+1):
	#if split cluster has higher performance
	features=feats3[label_predict==i,:]
	
	if np.size(features,0)<=2:
		continue
	
	kmeans1=KMeans(n_clusters=1,random_state=0).fit(features)
	#error1=np.sum(np.sum(np.square(features-kmeans.cluster_centers_[kmeans.labels_,:]),axis=1))
	error1=(np.mean(np.sum(np.square(features-kmeans1.cluster_centers_[kmeans1.labels_,:]),axis=1)))
	kmeans2=KMeans(n_clusters=2,random_state=0).fit(features)
	#error2=np.sum(np.sum(np.square(features-kmeans.cluster_centers_[kmeans.labels_,:]),axis=1))
	error2=(np.mean(np.sum(np.square(features-kmeans2.cluster_centers_[kmeans2.labels_,:]),axis=1)))
	#kmeans3=KMeans(n_clusters=3,random_state=0).fit(features)
	#error3=np.sum(np.sum(np.square(features-kmeans.cluster_centers_[kmeans.labels_,:]),axis=1))
	#error3=(np.mean(np.sum(np.square(features-kmeans3.cluster_centers_[kmeans3.labels_,:]),axis=1)))
	label_size=(np.size(features,0))
	
	if error2<(error1/1.1)  and label_size>200:#1.2
		print('error1='+str(error1)+',error2='+str(error2))
		print('cluster_id='+str(i))
		print('label size='+str(label_size))
		
		#split the cluster
		temp_label=kmeans2.labels_
		temp_label[temp_label==0]=i
		temp_label[temp_label==1]=next_label
		next_label=next_label+1
		label_predict2[label_predict==i]=temp_label
		#exit(1)
		#clustering, lambdas= SpectralClustering(n_clusters=10,assign_labels="discretize",random_state=0).fit(features)
	
		#print(lambdas)
		#print(lambdas[1:10]-lambdas[0:9])
		#exit(1)
	''
	if error3<(error2/1.3) and label_size>200:
		print('error1='+str(error1)+',error2='+str(error2))
		print('cluster_id='+str(i))
		print('label size='+str(label_size))
		
		#split the cluster
		temp_label=kmeans3.labels_
		temp_label[temp_label==0]=i
		temp_label[temp_label==1]=next_label
		next_label=next_label+1
		label_predict2[label_predict==i]=temp_label
	''
#'''

'''outlier relabel
#for every clusters
np.set_printoptions(threshold=sys.maxsize)
next_label=np.max(label_predict)+1
for i in range(0,np.max(label_predict).astype(int)+1):
	nn1=label_predict[knn5[label_predict==i,:]]
	nn1_dist=knn1_dist[label_predict==i,:]
	
	nn1_dist2=nn1_dist.copy()
	nn1_2=nn1.copy()
	
	nn1=nn1[:,1:]
	nn1_dist=nn1_dist[:,1:]
	nn1_dist[nn1!=i]=1e10
	min_dist1=np.min(nn1_dist,axis=1)
	sel=((min_dist1)<=0.05)
	min_dist1[min_dist1==1e10]=0
	
	#relabel the selected
	nn1_dist2[nn1_2==i]=1e10
	
	sel2=np.argmin(nn1_dist2,axis=1)
	min_dist2=np.min(nn1_dist2,axis=1)
	
	
	sel3=np.logical_or(np.logical_or(((min_dist2*1.2)>min_dist1) , ( min_dist2>1)),min_dist2==0)
	
	##print(np.stack((np.arange(0,np.size(nn1_2,0)),sel2),axis=1))
	##temp_label=nn1_2[np.transpose(np.stack((np.arange(0,np.size(nn1_2,0)),sel2),axis=0))]
	temp_label=nn1_2[np.arange(0,np.size(nn1_2,0)),sel2]
	temp_label[:]=-1
	
	temp_label[np.logical_or(sel,sel3)]=i
	
	if np.sum(temp_label==i)<len(temp_label):
		#print(nn1_2)
		#print(min_dist1)
		#print(sel)
		#print(nn1_2)
		#print(sel2)
		print(temp_label)
		label_predict2[label_predict==i]=temp_label
		#exit(1)
#'''

'''combine top n nearest pairs of clusters
mean_embeddings=np.zeros((np.max(label_predict).astype(int)+1,256))
cluster_weight=np.zeros(np.max(label_predict).astype(int)+1)
for i in range(0,np.max(label_predict).astype(int)+1):
	print('processing pred cluster '+str(i))

	features=feats[label_predict==i,:]
	
	mean_embeddings[i,:]=np.mean(features,axis=0)
	cluster_weight[i]=np.sqrt(np.sum(label_predict==i))
	
mean_embeddings=normalize(mean_embeddings,axis=1,norm='l2')
mean_embeddings=np.matmul(np.diag(cluster_weight),mean_embeddings)

pairwise_mean_dist=np.matmul(mean_embeddings,np.transpose(mean_embeddings))
pairwise_mean_dist=pairwise_mean_dist-np.diag(np.diag(pairwise_mean_dist))

for i in range(0,1):
	ind=np.argmax(pairwise_mean_dist)
	ind2=np.unravel_index(ind,(np.size(pairwise_mean_dist,0),np.size(pairwise_mean_dist,1)),order='f')
	
	print(ind2)
	#merge the two clusters
	label_predict2[ind2[1]]=ind2[0]
	pairwise_mean_dist[ind2[0],ind2[1]]=0
	pairwise_mean_dist[ind2[1],ind2[0]]=0
#'''

'''combine clusters if distance within threshold
count_list=(np.zeros(np.max(label_predict).astype(int)+1)).astype(int)
count_ind_list=(np.zeros(np.max(label_predict).astype(int)+1)).astype(int)
for i in range(0,np.max(label_predict).astype(int)+1):
	print('processing pred cluster '+str(i))

	nn1=knn1[label_predict==i,:]
	nn1_dist=knn1_dist[label_predict==i,:]
	nn1=label_predict[nn1]
	nn1[nn1_dist>=0.5]=-1
	nn1[nn1==i]=-1
	count=np.bincount(nn1.flatten().astype(int)+1)
	count=count[1:]
	
	if len(count)>0:#np.sum(count>0)>1:
		#count_i=count[i]
		#count[i]=0
		count_ind_list[i]=np.argmax(count)
		count_list[i]=count[count_ind_list[i]]
	else:
		count_ind_list[i]=-1
		count_list[i]=0
	#print(count_ind_list[i])
	#input("Press Enter to continue...")
	
	features=feats1[np.logical_or(label_predict==i,label_predict==count_ind_list[i]),:]
	#kmeans=KMeans(n_clusters=2,random_state=0).fit(features)
	clustering,lambdas = SpectralClustering(n_clusters=3,assign_labels="discretize",random_state=0).fit(features)
	
	lambdas=np.abs(lambdas[::-1])
	print(count_ind_list[i])
	print(lambdas)
	if lambdas[1]>0.97 and (np.sum(label_predict==i)+np.sum(label_predict==count_ind_list[i]))>100:
		print('cluster='+str(i))
		temp_label=clustering.labels_
		temp_label=i
		#temp_label[temp_label==0]=i
		#temp_label[temp_label==1]=count_ind_list[i]
		label_predict2[np.logical_or(label_predict==i,label_predict==count_ind_list[i])]=temp_label
#'''
'''	
for i in range(0,np.max(label_predict).astype(int)+1):
	#print('processing pred cluster '+str(i))
	for j in range(i+1,np.max(label_predict).astype(int)+1):
		if count_ind_list[i]==j and count_ind_list[j]==i:
		#if ssum1>10 and ssum2>10:
			print('sum1='+str(count_list[i])+',ssum2='+str(count_list[j]))
			print('cluster_id='+str(i)+','+str(j))
			#print('label size='+str(label_size))
		
			#input("Press Enter to continue...")
			#merge the cluster
			label_predict2[np.logical_or(label_predict==i, label_predict==j)]=i
'''
'''
		#merge the clusters if it has higher performance
		features1=feats1[label_predict==i,:]
		features2=feats1[label_predict==j,:]
		features=np.concatenate((features1,features2),axis=0)
		
		center1=np.mean(features1,axis=0)
		center1=np.tile(center1,(np.size(features1,0),1))
		center2=np.mean(features2,axis=0)
		center2=np.tile(center2,(np.size(features2,0),1))
		center=np.mean(features,axis=0)
		center=np.tile(center,(np.size(features,0),1))
		
		error1=np.sqrt(np.sum(np.sum(np.square(features1-center1),axis=1)))
		error2=np.sqrt(np.sum(np.sum(np.square(features2-center2),axis=1)))
		error=np.sqrt(np.sum(np.sum(np.square(features-center),axis=1)))
		
		label_size=np.minimum(np.sum(label_predict==i),np.sum(label_predict==j))
		if error<((0.5*error1+0.5*error2)*1.5) and label_size>200:
			print('error1='+str(error1)+',error2='+str(error2))
			print('cluster_id='+str(i)+','+str(j))
			print('label size='+str(label_size))
		
			#merge the cluster
			label_predict2[np.logical_or(label_predict==i, label_predict==j)]=i
			#exit(1)
'''

np.savetxt('label_predict2.txt',label_predict2)

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