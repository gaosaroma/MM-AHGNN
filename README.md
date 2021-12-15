# Object Interaction Recommendation with Multi-Modal Attention-based Hierarchical Graph Neural Network(MM-AHGNN)

Contributors: Lipeng Liang (gaosaroma@outlook.com)

> Zhang H, Liang L, Wang D. Object Interaction Recommendation with Multi-Modal Attention-based Hierarchical Graph Neural Network[C]//2021 IEEE International Conference on Big Data (Big Data). IEEE, December 15-18, 2021.

## Abstract

Object interaction recommendation from Internet of Things (IoT) is a crucial basis for IoT related applications. While many efforts are devoted to suggesting object for interaction, the majority of models rigidly infer relationships from human social network, overlook the neighbor information in their own object social network and the correlation of multiple heterogeneous features, and ignore multi-scale structure of the network. 

To tackle the above challenges, this work focuses on object social network, formulates object interaction recommendation as multimodals object ranking, and proposes Multi-Modal Attentionbased Hierarchical Graph Neural Network (MM-AHGNN), that describes object with multiple knowledge of actions and pairwise interaction feature, encodes heterogeneous actions with multi-modal encoder, integrates neighbor information and fuses correlative multi-modal feature by intra-modal hybrid-attention graph convolution and inter-modal transformer encoder, and employs multi-modal multi-scale encoder to integrate multi-level information, for suggesting object interaction more flexibly.

With extensive experiments on real-world datasets, we prove that MMAHGNN achieves better recommendation results (improve 3-4% HR@3 and 4-5% NDCG@3) than the most advanced baseline. To our knowledge, our MM-AHGNN is the first research in GNN design for object interaction recommendation.

## Framework
The overall structure of MM-AHGNN is shown as following, and it has the four key modules:
- Multi-modal encoder.
- Intra-modal graph convolution based on hybrid-attention.
- Inter-modal fusion based on transformer encoder.
- Multi-modal multi-scale encoder.
![](https://github.com/gaosaroma/MM-AHGNN/blob/main/pic/framework.png?raw=true)

Details in Intra-modal Graph Conv based on Hybrid Attention: involving multi-attention propagation and channel-attention aggregation, which selectively aggregate neighbors and highlights important channels for object representation, which is shown as following:

![](https://github.com/gaosaroma/MM-AHGNN/blob/main/pic/intra-modal.png?raw=true)

## Requirements

The code rely on Python 3.6 using PyTorch 1.9.1 and PyTorch Geometric 2.0.1 (along with their dependencies).

## Datasets
Due to the similarity between object behavior and human’s, we evaluate MM-AHGNN on object interaction recommendation using four datasets from [Yelp](www.yelp.com/dataset/documentation/main).

We only upload Region1 dataset, which is produced by Multi-Modal Encoder:
- Location encoder: fuzzy c-means with $n_c$ = 65
- Semantic service encoder: Biterm Topic Model(BTM) with $n_s$ = 70

We concatenate the three modals as input.

## Running the Model

To train the model, please run

```
python train.py -s btm_o2m -n_s 70 -l cmeans -nhead 2 -nlayer 1 -n_c 65
```