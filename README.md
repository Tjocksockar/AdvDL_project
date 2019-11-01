# AdvDL_project

## Introduction

This is a reimplementation of the paper "SCAN: A Scalable Neural Networks Framework
Towards Compact and Efficient Models" [1] published in NIPS19. 
This project is conductd as the final project in the course DD2412 Deep learning advanced course at the Royal Institute of Technology (KTH) in Stockholm Sweden. 
The implementation is made in Keras with Tensorflow. 

[1] Zhang, L., Tan, Z., Song, J., Chen, J., Bao, C., & Ma, K. (2019). SCAN: A Scalable Neural Networks Framework Towards Compact and Efficient Models. arXiv preprint arXiv:1906.03951.ISO 690	

## Instructions for running the code

Before running any code, make sure to download CIFAR100 and place it in the working directory. Name the CIFAR100 dataset outer folder to pics. \\ 
Your dataset folder should look like this\\
pics \\
-- train \\
-- -- class1 \\
-- -- -- image_1 and so on\\
-- -- class 2 and so on\\
-- test \\
-- -- class1\\
-- -- -- image_1 and so on\\
-- -- class 2 and so on\\

To start training of the backbone VGG16 run ```python3 vgg16_train_backbone.py```

To start training of the backbone VGG19 run ```python3 vgg19_train_backbone.py```

To start training of SCAN run ```python3 train.py ```
