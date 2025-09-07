# 
This is code associated with the paper ADDA: An adversarial direction-guided decision-based attack via multiple surrogate models.
(https://www.mdpi.com/2227-7390/11/16/3613)
Requirements:

* Pytorch (torch, torchvision) packages
* argparse package


To reproduce, e.g., the Linf norm untargeted attack experiments against the resnet50 architecture with epsilon = 0.05, you can run with the following command:

```
python add_attack.py --dset imagenet --archTarget resnet50 --norm_type linf --epsilon 0.05 --max_query 1000 --target_class -1 --lr 0.01
```

* For consistency, we fixed a set of 1000 ImageNet validation set images, and performed all the experiments in our paper on this set. 
