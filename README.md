# On Pruning Adversarially Robust Neural Networks

Repository with code to reproduce the results in our paper on novel pruning techniques with robust training. This repository supports all four robust training objectives: iterative adversarial training, randomized smoothing, MixTrain, and CROWN-IBP.

Following is a snippet of key results where we showed that accounting the robust training objective in pruning strategy can lead to large gains in the robustness of pruned networks. 

​	![...](\images\results_table.png)



In particular, the improvement arises from letting the robust training objective controlling which connections to prune. In almost all cases, it prefers to pruned certain high-magnitude weights while preserving other small magnitude weights, which is orthogonal to the strategy in well-established least-weight magnitude (LWM) based pruning. 

![...](\images\weight_histogram.png)



## Getting started

Let's start with installing all dependencies. 

`pip install -r requirement.txt`



We will use `train.py` for all our experiments on the CIFAR-10 and SVHN dataset. For ImageNet, we will use `train_imagenet.py`. It provides the flexibility to work with pre-training, pruning, and Finetuning steps along with different training objectives.

- `exp_mode`: select from pretrain, prune, finetune
- `trainer`: benign (base), iterative adversarial training (adv), randomized smoothing (smooth), mix train, crown-imp 
- `--dataset`: cifar10, svhn, imagenet



Following [this](https://github.com/allenai/hidden-networks) work, we modify the convolution layer to have an internal mask. We can use masked convolution layer with `--layer-type=subnet`. The argument `k` refers to the fraction of non-pruned connections.



## Pre-training

In pre-training, we train the networks with `k=1` i.e, without pruning. Following example pre-train a WRN-28-4 network with adversarial training.

`python train.py --arch wrn_28_4 --exp-mode pretrain --configs configs/configs.yml --trainer adv --val_method adv --k 1.0`



## Pruning

In pruning steps, we will essentially freeze weights of the network and only update the importance scores. The following command will prune the pre-trained WRN-28-4 network to 99% pruning ratio.  

`python train.py --arch wrn_28_4 --exp-mode prune --configs configs.yml --trainer adv --val_method adv --k 0.01 --scaled-score-init --source-net pretrained_net_checkpoint_path --epochs 20 --save-dense`

It used 20 epochs to optimize for better pruned networks following the proposed scaled initialization of importance scores. It also saves a checkpoint of pruned with dense layers i.e, throws aways masks form each layer after multiplying it with weights. These dense checkpoints are helpful as they are directly loaded in a model based on standard layers from torch.nn. 



## Fine-tuning

In the fine-tuning step, we will update the non-pruned weights but freeze the importance scores. For correct results, we must select the same pruning ratio as the pruning step. 

`python train.py --arch wrn_28_4 --exp-mode finetune --configs configs.yml --trainer adv --val_method adv --k 0.01 --source-net pruned_net_checkpoint_path --save-dense --lr 0.01`



## Least weight magnitude (LWM) based pruning 

We use a single shot pruning approach where we prune the desired number of connections after pre-training in a single step. After that, the network is fine-tuned with a similar configuration as above. 

`python train.py --arch wrn_28_4 --exp-mode finetune --configs configs.yml --trainer adv --val_method adv --k 0.01 --source-net pretrained_net_checkpoint_path --save-dense --lr 0.01 --scaled-score-init`

The only difference from fine-tuning from previous steps is the now we initialized the importance scores with proposed scaling. This scheme effectively prunes the connection with the lowest magnitude at the start. Since the importance scores are not updated with fine-tuning, this will effectively implement the LWM based pruning. 



## Bringing it all together

We can use following scripts to obtain compact network from both LWM and proposed prunign techniques. 

- `get_compact_net_adv_train.sh`: Compact networks with iterative adversarial training. 
- `get_compact_net_rand_smoothing.sh` Compact networks with randomized smoothing.
- `get_compact_net_mixtrain.sh` Compact networks with MixTrain. 
- `get_compact_net_crown-ibp.sh` Compact networks with CROWN-IBP.





## Finding robust sub-networks

It is curious to ask whether pruning certain connections itself can induce robustness in a network. In particular, given a non-robust network, does there exist a highly robust subnetwork? We find that indeed there exist such robust subnetworks with a non-trivial amount of robustness. Here is an example to reproduce these results:

`python train.py --arch wrn_28_4 --configs configs.yml --trainer adv --val-method adv --k 0.5 --source-net pretrained_non-robust-net_checkpoint_path`

Thus, given the checkpoint path of a non-robust network, it aims to find a sub-network with half the connections but having high empirical robust accuracy. We can similarly optimize for verifiably robust accuracy by selecting `--trainer` from `smooth | mixtrain | crown-ibp`, with using proper configs for each. 



## Model Zoo (checkpoints for pre-trained and compressed networks)

We are releasing pruned models for all three pruning ratios (90, 95, 99%) for all three datasets used in the paper. In case you want to compare some additional property of pruned models with a baseline, we are also releasing non-pruned i.e., pre-trained networks. Note that, we use input normalization only for the ImageNet dataset. 

### Adversarial training  

| Dataset  | Architecture | Pre-trained (0%) | 90% pruned | 95% pruned | 99% pruned |
| :------: | :----------: | :--------------: | :--------: | :--------: | :--------: |
| CIFAR-10 |    VGG16     |     [ckpt]()     |  [ckpt]()  |  [ckpt]()  |  [ckpt]()  |
| CIFAR-10 |   WRN-28-4   |     [ckpt]()     |  [ckpt]()  |  [ckpt]()  |  [ckpt]()  |
|   SVHN   |    VGG16     |     [ckpt]()     |  [ckpt]()  |  [ckpt]()  |  [ckpt]()  |
|   SVHN   |   WRN-28-4   |     [ckpt]()     |  [ckpt]()  |  [ckpt]()  |  [ckpt]()  |



### Randomized smoothing

| Dataset  | Architecture | Pre-trained (0%) | 90% pruned | 95% pruned | 99% pruned |
| :------: | :----------: | :--------------: | :--------: | :--------: | :--------: |
| CIFAR-10 |    VGG16     |     [ckpt]()     |  [ckpt]()  |  [ckpt]()  |  [ckpt]()  |
| CIFAR-10 |   WRN-28-4   |     [ckpt]()     |  [ckpt]()  |  [ckpt]()  |  [ckpt]()  |
|   SVHN   |    VGG16     |     [ckpt]()     |  [ckpt]()  |  [ckpt]()  |  [ckpt]()  |
|   SVHN   |   WRN-28-4   |     [ckpt]()     |  [ckpt]()  |  [ckpt]()  |  [ckpt]()  |



### Adversarial training on ImageNet (ResNet50)

| Pre-trained (0%) | 95% pruned | 99% pruned |
| :--------------: | :--------: | :--------: |
|     [ckpt]()     |  [ckpt]()  |  [ckpt]()  |



## Contributors

* Vikash Sehwag
* Shiqi Wang



Some of the code in this repository is based on followng amazing works.

* https://github.com/allenai/hidden-networks
* https://github.com/yaircarmon/semisup-adv
* https://github.com/locuslab/smoothing
* https://github.com/huanzhang12/CROWN-IBP
* https://github.com/tcwangshiqi-columbia/symbolic_interval



## Reference

In you find this work helpful, consider citing it. 



