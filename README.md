# Automatic Detection of Coronavirus Desease in X-ray Images


1. **Introduction** 

The newly identified Coronavirus pneumonia, subsequently termed COVID-19, occurred in December 2019, in Wuhan, Hubei province, China. The most common presenting symptoms experienced by patients include dry cough, sore throat, fever, dyspnea, diarrhea, myalgia, shortness of breath, and bilateral lung infiltrates, observable on clinical imaging such as chest X-ray. 

Despite global efforts of travel restrictions and quarantine, the incidence of novel COVID-19 continues to rise globally, with over 29 million confirmed cases and over 939,935 deaths worldwide today according to worldmeters.info. 

It is evident that early detection of COVID-19 is necessary to interrupt the spread of COVID-19 and prevent transmission by early isolation of patients, trace, and quarantine of close contacts. While not recommended for primary diagnosis of COVID-19, medical imaging modalities such as chest X-ray play an important role in confirming the diagnosis of positive COVID-19 pneumonia as well as monitoring the progression of the disease. These types of images show the extent of irregular ground- glass opacities that progress rapidly after COVID- 19 symptom onset. These abnormalities peaked during days 6-11 of the illness. The second most predominant pattern of lung opacity abnormalities peaks during days 12-17 of the illness. Computer-Aided Diagnosis (CAD) systems that incorporate X-ray image processing techniques and deep learning algorithms could assist doctor as diagnostic aids for COVID-19 and help provide a better understanding of the progression of the disease. 

2. **Proposed Methodology** 

In this study, authors of the articles have proposed an automatic prediction of COVID-19 using a deep convolution neural network based pre-trained transfer models and Chest X-ray images. **Convolutional neural networks** 

Convolutional neural networks (CNNs) is a powerful tool for automatically classifying 2D or 3D image patches.  

CNNs have two main parts: 

- A convolution/pooling mechanism that breaks up the image into features and analyzes them 
- A fully connected layer that takes the output of convolution/pooling and predicts the best label to describe the image. 

![](Report.002.png)

*Convolution layer* 

Convolutional layers are the layers where filters are applied to the original image, ![](Report.003.png)or to other feature maps in a deep CNN. A “filter” passes over the image, scanning a few pixels at a time and creating a feature map that predicts the class to which each feature belongs. 

*Non Linearity (ReLU)* 

The most popular activation function for deep neural networks. ReLU stands for Rectified Linear Unit for ![](Report.004.png)a non-linear operation. The output is ƒ(x) = max(0,x).  

ReLU’s purpose is to introduce non-linearity in CNN. Since, the real world data would want our CNN to learn would be non-negative linear values. 

*Pooling layer* 

The Pooling layer is responsible for reducing the spatial size of the Convolved Feature. Spatial pooling also called subsampling or downsampling which reduces the dimensionality ![](Report.005.png)of each map but retains important information. Spatial pooling can be of different types: Max Pooling and Average Pooling 

The Convolutional Layer and the Pooling Layer, together form the i-th layer of a Convolutional Neural Network 

*Fully Connected Layer* 

A Fully-Connected layer  is a cheap way of learning non-linear combinations of the high-level features as represented ![](Report.006.png)by the output of convolutional layer. 

Fully connected layers are placed before the classification output of a CNN and are used to flatten the results before classification. 

**Using pre-trained networks/ transfer learning** 

A big advantage of CNN is that when the number of training images increases, the performance of the networks improves. Training a deep learning model requires a large amount of labeled data. However, in many medical image classification cases, the number of labeled data is limited for training. Transfer learning has been proposed to effectively tackle this problem.  

Transfer learning is a common and effective strategy to train a network on a small dataset, where a network is pretrained on an extremely large dataset, such as ImageNet, which contains 1.4 million images with 1000 classes, then reused and applied to the given task of interest. At present, many models pre-trained on the ImageNet challenge dataset are open to the public and readily accessible, along with their learned kernels and weights, such as VGG, ResNet, Inception, and DenseNet. 

Transfer learning strategies have various advantages, such as avoiding the overfitting issue when the number of training samples is limited, reducing the computational resources, and also speeding up the convergence of the network 

In this method, I used pre-trained well-established CNN models such as DenseNet121, VGG16 are selected for feature extraction with the possibility of transfer learning advantage for limited datasets and also their satisfying performances in different computer vision tasks. 

![](Report.007.png)

VGG16 

VGG16 is a convolution neural network (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014. It is considered to be one of the excellent vision model architecture till date.  

Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. 

![](Report.008.png)

DenseNet 

DenseNet can be regarded as a logical extension of ResNet which was first proposed in 2016 by Huang et al. from Facebook. In DenseNet, each layer of CNN connected to every other layer in the network in a feed-forward manner which helps in reducing the risk of gradient-vanishing, fewer parameters to train, feature-map reuse and each layer takes all preceding layer features as inputs. The authors also point out that when datasets used without augmentation, DenseNet is less prone to overfitting. There are a number of DenseNet architectures, but I opt to use DenseNet121 for analysis of COVID-19 detection from X-ray images by using the weights trained on ImageNet dataset. 

3. **Experiments** 

**Dataset Description** 

In order to evaluate the performance of our feature extracting and classifying approach, I use the pubic dataset of Xray images provided by Dr.Joseph Cohen avaivable from github repository. I use the available 180 chest X-ray images of COVID-19 positive cases (1) and 200 image of healthy cases from Kaggle Chest X-ray Image dataset available (2) 

![](Report.009.png)

Available data are typically split into three sets: a training, a validation, and a test set. 

- *A training set* is used to train a network, where loss values are calculated via forward propagation and learnable parameters are updated via backpropagation. 
- *A validation set* is used to evaluate the model during the training process, fine-tune hyperparameters, and perform model selection. 
- *A test set* is ideally used only once at the very end of the project in order to evaluate the performance of the final model that was fine-tuned and selected on the training process with training and validation sets. 

**Data pre-processing** 

*Resizing:* Re-scaling all images of the origional size to the size of 224x224. In Figure 2 and Figure 3, representative chest X-ray images of normal and COVID-19 patients are given, respectively. 

*Image Standardization* with ImageDataGenerartor 

Standardization typically means rescales data to have a mean of 0 and a standard deviation of 1. 

![](Report.012.png)

*Data Augmentation* is a method of artificially creating a new dataset for training from the existing training dataset to improve the performance of deep learning neural networks with the amount of data available. It is a form of regularization which makes our model generalize better than before. 

Here I have used a Keras ImageDataGenerator object to apply data augmentation for randomly translating, resizing, rotating, etc the images. Each new batch of our data is randomly adjusting according to the parameters supplied to ImageDataGenerator. 

![](Report.013.png)

**Performance Metrics** 

4 criteria were used for the performances of deep transfer learning models. These are: 

*Accuracy* shows the number of correctly classified cases divided by the total number of test images, and is defined as: 
*Recall* or sensitivity is the measure of Covid-19 cases that are correatly classified. Recall is critical, especially in the medical field
*Precision* or positive predictive value is defined as the percentage of correctly clasified labels in truly positive patients: 
*F1-score* is defined as the weighted average of precision and recall that combines both the precision and recall together.                      
Where: 

- True Positive (TP) is the number of instances that correctly predicted 
- False Positive (FP) is the number of negative instances that incorrectly predicted 
- True Negative (TN) is the number of negative instances that predicted correctly. 
- False Negative (FN) is the number of instances that incorrectly predicted. 
4. **Result and Discussion**  

In this study, Chest X-ray images have been used for prediction of coronavirus disease patients (COVID-19). Popular pre-trained models such as DenseNet121 and VGG16 have been trained and tested on chest X-ray images. Training accuracy and loss values of pretrained models are given in Figure 3 and Figure 4 respectively. The training stage has been carried out up to 20 epoches to avoid overfitting for all pre-trained models. It can be seen from Figure 3 that the training accuracy of DenseNet121 is better than VGG16 but both pre-train models obtained really high training accuracy. When the loss figure are analyzed, it is seen that the loss values decrease in teo pre-trained models during the training stage. It can be said that DenseNet121 model both decreases loss values faster and approaches zero. 

![](Report.017.png)

*Figure 3 The performance of  pre-trained models (Training Accuracy)* 

![](Report.018.png)

*Figure 4 The Performance of pre-trained models (Training Loss)* 

In Figure 5, Confusion matrices of COVID-19 and normal test results of the models are given. Firstly, DenseNet121 pre-trained model classified 40 of the COVID-19 as True Positive and classified 40 of the normal as True Negative. Secondly, VGG16  model also classified 40 of the COVID-19 as True Positive and classified 40 of the normal as True Negative. DenseNet121 and VGG16 pretrained models appear to be very high. 

![](Report.019.png)

*a)* 

![](Report.020.png)

*b)* 

Figure 5 The Confusion Matrix a) DenseNet121 b) VGG16 

In another detailed performance, comparisions of two models using test data are shown in Table 1. I have obtained the best performance as an accuracy of 100 %, recall 100%, Precision 100%, F1-score 100% for both DenseNet121 and VGG16. As a result, the DenseNet121 and VGG16 model perform really well both training and testing stage. 



|**Model/Performance** |**Accuracy (%)** |**Recall (%)** |**Precision (%)** |**F1-score (%)** |
| - | - | - | - | - |
|**DenseNet121** |100 |100 |100 |100 |
|**VGG16** |100 |100 |100 |100 |
Table 1: Prediction performance results obtained from different pre-trained CNN models. 

5. **Conclusion** 

Early prediction of COVID-19 patients is vital to prevent the spread of the disease to other people. In this study, I proposed a deep transfer learning based approach using chest X-ray images obtained from COVID-19 patients and normal to predict COVID-19 patients automatically. Performance results show that the DenseNet121 and VGG16 pre-trained model obtained the accuracy of 100%. In the light of findings, it is believed that it will help doctors to make decisions in clinical practice due to the high performance in order to detect COVID-19 at an early stage.  

**References** 

1. https://github.com/ieee8023/covid-chestxray-dataset 
1. https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia 
1. [https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep- learning-99760835f148 ](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)[4]https://www.researchgate.net/publication/325932686\_Convolutional\_neural\_networks\_an\_overvi ew\_and\_application\_in\_radiology [5]https://www.researchgate.net/publication/340859631\_Automatic\_Detection\_of\_Coronavirus\_Dise ase\_COVID-19\_in\_X-ray\_and\_CT\_Images\_A\_Machine\_Learning-Based\_Approach 
