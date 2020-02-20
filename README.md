# On Pruning Adversarially Robust Neural Networks

Repository with code to reproduce the results in our paper on novel pruning techniques with robust training. This repositoy support all four robsut training objectives: iterative adversarial training, randomized smoothing, MixTrain, and CROWN-IBP.



## Getting Started

Let's start with installing all dependencies. 

`pip install -r requirement.txt`



We will use `train.py` for all our experiment on CIFAR-10 and SVHN dataset. For ImageNet, we will use `train_imagenet.py`. It provide the flexbility to work with pre-training, pruning, and Finetuning steps along different training objectives.

- `exp_mode`: select from pretrain, prune, finetune
- `trainer`: beningn (base), iterative adversarial training (adv), randomized smoothing (smooth), mix train, crown-imp 
- `--dataset`: cifar10, svhn, imagenet



Following [this](https://github.com/allenai/hidden-networks) work, we modify the convolution layer to have an internal mask. We can use masked convolution layer with `layer-type=subnet`. The argument `k` refers to the fraction of non-pruned connections.



## Pre-training

In pre-training, we train the networks with `k=1` i.e, without pruning. Following example pre-train a WRN-28-4 network with adversarial training.

`python train.py --arch wrn_28_4 --exp-mode pretrain --configs configs/configs.yml --trainer adv --val_method adv --k 1.0`



## Pruning

In pruning steps, we will essentially freeze the networks weight and only update the importance scores. Following command will prune the pre-trained WRN-28-4 network to 99% pruning ratio.  

`python train.py --arch wrn_28_4 --exp-mode prune --configs configs.yml --trainer adv --val_method adv --k 0.01 --scaled-score-init --source-net pretrained_net_checkpoint_path --epochs 20 --save-dense`

It used 20 epochs to optimize for a better pruned networks following the proposed scaled initialization of importance scores. It also save a checkpoint of pruned with dense layers i.e, throws aways masks form each layer after multiplying it with weights. These dense checkpoints are helpful as they be directly loaded in model based on standard layer from torch.nn. 



## Fine-tuning

In fine-tuning steps, we will update the non-pruned weights, but freeze the importance scores. For correct results, we must select the same pruning ratio as pruning step. 

`python train.py --arch wrn_28_4 --exp-mode finetune --configs configs.yml --trainer adv --val_method adv --k 0.01 --source-net pruned_net_checkpoint_path --save-dense --lr 0.01`



## Least Weight magnitude based puning (LWM)

We use a single shot pruning approach where we prune the desired number of connection after pre-training in a single step. After that the network is fine-tuned with the similar configuration as above. 

`python train.py --arch wrn_28_4 --exp-mode finetune --configs configs.yml --trainer adv --val_method adv --k 0.01 --source-net pretrained_net_checkpoint_path --save-dense --lr 0.01 --scaled-score-init`

The only difference from fine-tuning from previous steps is the now we initialized the importance scores with proposed scaling. This scheme effective prune the connection with lowest magnitude at start. Since the importance scores are not updated with fine-tuning, this will effectively implement the LWM based pruning. 





## Bringing it all together

We can use following scripts to obtain compact network from both LWM and proposed prunign techniques. 

- `get_compact_net_adv_train.sh`: Compact networks with iterative adversarial training. 
- `get_compact_net_rand_smoothing.sh` Compact networks with randomized smoothing.
- `get_compact_net_mixtrain.sh` Compact networks with MixTrain. 
- `get_compact_net_crown-ibp.sh` Compact networks with CROWN-IBP.





## Contributors

* 
* 



Some part of this repository are based on:

* https://github.com/yaircarmon/semisup-adv
* https://github.com/locuslab/smoothing
* https://github.com/allenai/hidden-networks
* https://github.com/huanzhang12/CROWN-IBP
* https://github.com/tcwangshiqi-columbia/symbolic_interval



## Reference


