# face-clustering

This face clustering makes use of existing work([CDP codes]https://github.com/XiaohangZhan/cdp). This clustering algorithm has been submitted to...

## Paper
1. [Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaohang_Zhan_Consensus-Driven_Propagation_in_ECCV_2018_paper.pdf), ECCV 2018 [[Project Page](http://mmlab.ie.cuhk.edu.hk/projects/CDP/)]
2. [Linkage-based Face Clustering via Graph Convolution Network](https://arxiv.org/abs/1903.11306), CVPR 2019
3. [Learning to Cluster Faces on an Affinity Graph](https://arxiv.org/abs/1904.02749), CVPR 2019 [[Project Page](http://yanglei.me/project/ltc)]
4. [Efficient Parameter-free Clustering Using First Neighbor Relations](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sarfraz_Efficient_Parameter-Free_Clustering_Using_First_Neighbor_Relations_CVPR_2019_paper.pdf), CVPR 2019

## Requirements
* Python >= 3.6
* scikit-learn

## Setup and get data

Install dependencies
```
pip install -U scikit-learn
```
Please download and use this dataset [DATASET.md](https://github.com/yl-1993/learn-to-cluster/blob/master/DATASET.md) to evaluate the clustering algorithm. 
You may also use the face embeddings dataset from (https://github.com/Zhongdao/gcn_clustering).

## Run
```
git clone git@github.com:XiaohangZhan/cdp.git
```
Put your face embeddings file e.g. part0.bin in the folder.
```
cdp/data/unlabeled/omni/features/part0.bin
```
Create the config.yaml file.
Cluster using the commmands in (https://github.com/XiaohangZhan/cdp).
```
python -u main.py --config experiments/omni/config.yaml
```
Download this github python files *.py into the folder
```
cdp/data/unlabeled/omni/
```
Create a list.txt file from the meta.txt file using our codes
```
python create_list.py
```
Then further split the clusters using our codes
```
python cluster_finetuning_and_perf_other_algo.py
```
Remember to set the data_name and features to the values below 
```
#load features
data_name='unlabeled/omni'
feats1 = load_feats('C:/Users/isetsk/Downloads/cdp-master/cdp-master/data/{}/features/{}.bin'.format(data_name, 'part0'),256)
```
And set the knn file to 
```
knn1=np.load('./knn/part0_k15.npz')
```
Put the ground truth labels meta.txt and cdp clustering labels meta.txt in the correct folder and set e.g.
```
label_true=meta=np.loadtxt('meta.txt')
label_predict=np.loadtxt('..../cdp/experiments/omni/output/k15_vote_accept0_th0.7/sz600_step0.05/meta.txt')
```
Then label Propagation of Remaining Unlabeled Face embeddings using
```
python label_pred_smooth_by_neighbors.py
```

## Results
Look at the results in our paper (... links to our paper).

## Citations
If you use our work, please cite
```
@inproceedings{ourpaper,
  title={,
  author={},
  booktitle={},
  year={}
}
```
