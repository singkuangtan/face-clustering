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
pip install -r requirements.txt
```
Please download and use this dataset [DATASET.md](https://github.com/yl-1993/learn-to-cluster/blob/master/DATASET.md) to evaluate the clustering algorithm. 
You may also use the face embeddings dataset from (https://github.com/Zhongdao/gcn_clustering).

## Run
```
git clone git@github.com:XiaohangZhan/cdp.git
```
Cluster using the commmands in (https://github.com/XiaohangZhan/cdp).
Then further split the clusters using our codes
```
python ....py
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
