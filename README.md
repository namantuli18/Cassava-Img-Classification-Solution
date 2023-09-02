# Cassava-Leaf-Disease-Classification-Solution

## Results
| Public Score | Private Score | Public Rank | Private Rank |
|----------|----------|----------|----------|
| 0.9086 | 0.8993 | 10/3901  | 165/3901

## Problem Statement  
The competition asked the participants to classify leaf images into five categories that cause material harm to the food supply of many African countries.

## Dataset
The training data comprises of ~23K training images belonging to one of the following leaf-disease categories:
* Cassava Bacterial Blight (CBB)
* Cassava Brown Streak Disease (CBSD)
* Cassava Green Mottle (CGM)
* Cassava Mosaic Disease (CMD)
* Healthy

The train images were available as both raw images as well as tf-records. The test dataset consisted of around 15K unseen images for the participants to predict.

### Resources to the dataset:  
Ernest Mwebaze, Jesse Mostipak, Joyce, Julia Elliott, Sohier Dane. (2020). Cassava Leaf Disease Classification. Kaggle. https://kaggle.com/competitions/cassava-leaf-disease-classification

## Evaluation Metric  
Submissions were evaluated based on their [categorization accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy).

## Methodology

### Curating the dataset
1. All the training images were loaded into a dataloader, which was further used to train the model.
2. The corresponding labels were one-hot encoded for training.
3. We used the following soft augmentations from albumentations to oversample the underrepresented classes:
   * RandomResizedCrop
   * Transpose
   * HorizontalFlip
   * VerticalFlip
   * ShiftScaleRotate
   * Cutout
4. In addition to these algorithms, the following set of hard augmentations were used that helped to increase our score :
   * [Cutmix](https://paperswithcode.com/method/cutmix#:~:text=CutMix%20is%20an%20image%20data,of%20pixels%20of%20combined%20images.)
   * [Snapmix](https://arxiv.org/abs/2012.04846)
   * [Fmix](https://paperswithcode.com/method/fmix)
<img src="https://github.com/namantuli18/Feedback-Prize-Longformer-Ensemble/blob/main/imgs/dataset.png" width="600" height="300" />

### Model training 
* For training the model on our set of images, we trained a variety of models in order to strenghten the overall ensemble. Since the training set consisted of a smaller set of images (15K), we trained the models over multiple folds.
The default parameters for our models are listed below:
    ```python
    CFG = {
        'folds': 5,
        'seed': 42,
        'img_size': 600,
        'epochs': 10,
        'train_bs': 8,
        'valid_bs': 32,
        'lr': 1e-4,
        'min_lr': 1e-6,
        'weight_decay':1e-6
    }
    ```
* We used the python package [timm](https://pypi.org/project/timm/) to load the pre-trained weights of the backbone model architectures.
* Another integral part of the training process was the choice of loss function. Even though the default loss function was [cross entropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), training the data using [Focal Loss](https://github.com/clcarwin/focal_loss_pytorch) helped to further increase our scores. 
* We used variants of the following model architectures while training our models:
  1. EfficientNet
  2. Vision Transformers
  3. Resnext
  4. Resnest

* Because of the self-attention mechanism, Vision Transformers provided a different aspect towards this problem, as compared to EfficientNet architectures. Consequently, when used together, both of these models provided great cross-validatioon accuracy. 

### Model Evaluation and Inference

While evaluating model's overall accuracy, we emphasised on the correlation between the cross-validation (CV) and leaderboard (LB) scores to stabilise the algorithm. Our approach tried to ensemble multiple algorithms, aiming to enhance the individual performance of the models.


The performance of our individual models, along with their weightage in our final ensembles, has been encapsulated below:
| S No | Model Name | Link | Weightage in Ensemble |
|----------|----------|----------|----------|
| 1. | EfficientNetB3 | https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB3 | 0.45
| 2. | VitBasePatch-384 | https://huggingface.co/google/vit-base-patch16-384| 0.36
| 3. | Resnext50_32X4D With and without TTA(Test Time Augmentation) | https://pytorch.org/vision/main/models/generated/torchvision.models.resnext50_32x4d.html | 0.1
| 4. | Resnest50D| https://paperswithcode.com/lib/timm/resnest | 0.09

Since the CV scores of EfficientNets and Vision Transformers were significantly higher, we wanted their predictions to be slightly oversampled than their Resnet based counterparts. Resnext and resnest, individually were not as robust, but provided great variability in the final blend.

We logged in the final scores along with the corresponding approach and parameters, which helped us arrive at the proportion of the final blend. It has been shared below for reference:

## Key takeaways
1. Training resnest and resnext architectures over a single fold helped us increase our score in the final stages of the competition.
2. Image augmentations were beneficial in over-sampling the underrepresented classes.
3. Green filtering on the images did not effectively help to increase our scoores.
4. Pertaining to the large number of competitors and a relatively simple problem statement, the Discussions thread seemed to be a gold mine for information and apporaches. However, we felt short of time while applying some techniques that had turned out to be effective for some challengers.
5. We tried training models like RegNety and NFNet, but they were not able to significantly improve the scores of our existing models.

## Code 
For training code, you can refer file `scripts/train_eff5.ipynb`  
For inference script, please refer [Kaggle Notebook](https://www.kaggle.com/code/namantuli/ensemble-inferencev2-1-889683/notebook) or file `scripts/clean-feedback-ensemble-balanced.ipynb`
