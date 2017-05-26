# Mxnet - Deep Learning analysis of Chest XRays 
(port and enhancement of tensorflow repo found at ayush1997/Xvision using Resnet-50)

Chest Xray image analysis using  **Deep Transfer Learning** technique.  Written in python for MxNet deeplearning framework.

Summary: The **flatten_output** layer of the pretrained Inception-BN was stripped away and a new 3 layer fully connected neural net was added on top to convert it to a classifier of **Normal vs Nodular** Chest Xray Images.

## Nodular vs Normal Chest Xray
<img src="https://github.com/kperkins411/XVision_Mxnet/blob/master/images/nodule.png" width="300" height="300" />
<img src="https://github.com/kperkins411/XVision_Mxnet/blob/master/images/normal.png" width="300" height="300" />

## Some specifications

| Property      |Values         |
| ------------- | ------------- |
| Pretrained Model | Inception BN  |
| Optimizer used  | stochastic gradient descent(SGD)  |
| Learning rate  | 0.01 (network is very sensitive, LR=.1 and it never converges)|  
|Mini Batch Size| 16 |
| Epochs | trained until reach 100% on training set, used network with best validation score |
|3 new FC Layers| 512x256x128 |
|GPU trained on| Nvidia GEFORCE GTX 960M|

## Evaluation

**Accuracy** : **72.02 %**

## DataSet
[openi.nlm.nih.gov](https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x&m=1&n=101) has a large base of Xray,MRI, CT scan images publically available.Specifically Chest Xray Images have been scraped, Normal and Nodule labbeled images are futher extrated for this task.

## How to use ?
This code can be used for **Deep Transfer Learning** on any XRAY dataset to train using Inception-BN as the PreTrained network. You can also use any of the models in the MXNet model library to run this code.  It has been tested with Resnet-34 and resnet-50 as well
### Steps to follow 

1. Get images- Goes to NLM website and recursively walks all pages downloading images and metadata, the images go in "../images_all" the metadata goes in a json file defined in settings.json_data_file. 

  ```python A1_getRawImages.py```

2. Now seperate and balance the dataset (there are about 2706 normal images and 211 nodule ones roughly 13 to 1).  So flip and copy the nodule set so that they are roughly equivelant.  Then break into train,test,val sets (.7, .15,.15).
Copy them to seperate folders 
    ../images_Train/nodule
    ../images_Train/nodule
and likewise for test and Val

```python A2_processRawImages.py```

3. Create RecordIO files that contain efecient, concatenated binary files consisting of all images in a particular category (train, test, val)

```python A3_createRecFiles.py```

4. Now the heavy lifting, 
    take a pretrained deep conv net (Inception, Resnet etc.) and strip off the final fully connected layer to form headless         CNN
    Run train and validation images through headless CNN and gather all the outputs, these are called CNN codes, 
    Use CNN codes to train a brand new 3 layer fully connected neural net (converges quickly). This is necessary because          mxnet has no way to freeze layers at this point, plus its much faster to train)
    then append that TRAINED fully connected neural net onto the headless CNN and train the complete net a little (converges         quickly).  
    then save the best complete net (use A_utilities.epoc_end_callback_kp(...) in the fit function.  Whenever accuracy             exceeds previous record save the net.
    Finaly reload complete net and test on validation data
    
    ```python A4_freeze_all_but_last_layer.py```

## Improvements?

> 1. This is a binary classifier, there is no attempt to segment the lungs, this pretraining step may help imensily
> 2. More data
> 3. Deeper network (Resnet 150 maybe?)


