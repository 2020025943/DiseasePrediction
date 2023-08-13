DiseasePrediction
=======================

This project is a collection of my dissertation projects. The first paper, Research on Disease Prediction Based on Improved DeepFM and IoMT: We adopt the modified DeepFM to predict hepatitis. The second paper, Prototypical Network Based on Manhattan Distance: We adopt the Protopical Network in few-shot learning to classify generic image datasets. The third paper, FMPNN in Digital Twins Smart Healthcare: We predict patients with or without stroke. The forth paper, ENFM: Extreme Neural Factorization Machine for Disease Prediction: A new model called Extreme Neural Factorization Machine (ENFM) has been proposed for the detection of depression. The proposed models have excellent performance and can be better used in disease prediction. 

Research on Disease Prediction Based on Improved DeepFM and IoMT
-------------------

### Data:  
ALF_Data.csv file. Use this dataset to predict the presence or absence of hepatitis in patients.  

The dataset contains a total of 6,000 adults over the age of 20 from two surveys in 2008-2009 and 2014-2015.The data was collected from a nationwide survey of adults conducted by the JPAC Center for Health Diagnosis and Control. Through visits and research by professional medical personnel, the data set covers a wide range of population information and their health status information, which comes from direct interviews, physical examinations and blood sample examinations.  

Some fields are explained as follows:  
|Field name |Field type |Example description |  
|Waist |float |90.5 |  
|Obesity |bool |0 |  
|Body Mass Index|float |22.2 |  
|Physical Activity|int |3 |  
|Source of Care |str |clinic |  
|PoorVision |bool |0 |  
|Maximum Blood Pressure |int |120 |  
|Minimum Blood Pressure |int |80 |  
|Good Cholesterol |int |123 |  
|Bad Cholesterol |int |99 |  
|Dyslipidemia |bool |0 |  
|PVD (Peripheral Vascular Disease) |bool |1 |  
|HyperTension|bool |1 |  
|Family HyperTension |bool |1 |  
|Diabetes |bool |1 |  
|Family Diabetes |bool |0 |  
|Hepatitis|bool |1 |  
|Family Hepatitis |bool |1 |  
|Chronic Fatigue |bool |1 |  
|ALF (Acute Liver Failure) |bool |1 |  

### Code:  
Using tensorflow2.0 environment.  

This research uses the deepctr package to implement the DeepFM model. This package is an easy-to-use and scalable deep learning click-through rate prediction algorithm package developed by Weichen Shen, a computer master from Zhejiang University and a current Alibaba’s algorithm engineer. It can quickly build a CTR prediction algorithm from existing components. model. It can quickly build a click-through rate prediction algorithm model from existing components.  

Deepctr abstracts and summarizes the structure of the existing CTR prediction model based on deep learning, and adopts the idea of modularization in the design process. Each module itself has high reusability, and each module is independent of each other. The CTR prediction model based on deep learning can be divided into the following four modules according to the functions of the internal components of the model: input module, embedding module, feature extraction module, and prediction output module.  

deepFMtest2.ipynb. Improves the DeepFM model, and assigns different weights to the linear output of the FM part of the DeepFM algorithm, the second-order combined output, and the output of the Deep part to indicate its proportion in the prediction.  

dessert1.ipynb. Use DeepFM model.  

deep.ipynb. Use DNN model. 

logistic.ipynb. Use logistic regression model.  

### Configuration method:  
(1) Configure the tensorflow2.0 environment (mainly on the command line: pip install tensorflow==2.0)  
(2) Enter pip install deepctr on the command line to download the necessary deepctr package  
(3) Put the data set (ALF_Data.csv) in the location where the file is read in the code  
(4) Upload the code to jupyter  
(5) Open the code and click Run to run  
(6) The final output is the evaluation result and the AUC and loss function diagram on the corresponding training set and test set  

Prototypical Network Based on Manhattan Distance
------------------------

### Data:  
mini-imagenet.zip. Classify the images in this dataset.  

The dataset used in this study is miniImageNet proposed by Vinyals, which is the subset of the famous ILSVRC (ImageNet Large Scale Visual Recognition Challenge)-12 (ImageNet2012 dataset). ILSVRC-12 includes 1000 classes. There are more than 1000 samples in each category, which is very large. MiniImageNet selected 100 categories, including birds, animals, people, daily necessities, etc. Each category includes 600 84*84 RGB color pictures. miniImageNet training is difficult, for Few-shot learning, it still has very large development space. In this study, the dataset is simplified, and each class includes 350 samples. Among these 100 classes, we will randomly select 64 classes of data as the training set, 16 classes as the validation set, and the remaining 20 classes as the test set. We will mainly use the data division method of 5 way 5 shot and 5 way 1 shot for experiments.  

Compared to the CIFAR10 dataset, the miniImageNet dataset is more complex, but more suitable for prototyping and experimental research.  

Here, the original dataset file has been split into train.npy (training set) and test.npy (testing set).  

Since it is a few-sample learning, after dividing the training set and the test set, the N way K shot method is used to divide the data: that is, metadata is divided into tasks instead of samples, and each task is internally divided into training set and test set, which are called support set and query set respectively. For each task, N classes (way) were randomly selected from the metadata set, and K (Shot)+1 or K+M samples were randomly selected from each class. N*K samples were classified into the support set for training, and the remaining N*1 or N*M samples were classified into the query set for verification testing.  

### Code:  
Using tensorflow2.0 environment.  

ProtocalNetwork.ipynb. An improved Prototypical Network is proposed, in which the core distance measurement function of the original Prototypical Network is changed from Euclidean distance to Manhattan distance, and the average pooling layer and Dropout layer are added respectively.  

Metric learning, also called similarity learning, calculates the distance between two samples through a distance function, measures the similarity between them, and determines whether they belong to the same category. The Metric Learning algorithm is composed of an embedding module and a metric module. The embedding module converts the sample into a vector in the vector space. The metric module gives the similarity between the samples. . The Prototypical Network maps the sample data in each category to a space, and calculates their mean value as the prototype of the category. Using Euclidean distance as the measurement function, through training, the distance between the sample data and the prototype of its type is the shortest, and the distance to the prototype of other types is farther. During the test, the distance between the test data and the prototype of each category is processed by softmax function, and the category label of the test data is judged by the output probability value.   

### Configuration method:  
(1) Configure the tensorflow2.0 environment (mainly on the command line: pip install tensorflow==2.0)  
(2) Put the training set and test set path in the location where the file is read in the code  
(3) Upload the code to jupyter  
(4) Open the code and click Run to run  
(5) Finally, the accuracy rate and loss function value of the training set are output once every 50 rounds  
After the training, the accuracy rate and loss function value of the test set are output once every 50 rounds  

FMPNN in Digital Twins Smart Healthcare
-------------------

### Data:  
The dataset healthcare-dataset-stroke-data in this paper is provided by FEDESORIANO on Kaggle, which is used to predict whether a patient is likely to have a stroke based on input parameters (sex, age, various diseases and smoking status, etc.). Each row in the data provides relevant information about the patient.  

The various fields of this dataset are explained as follows:   
(1) id: a unique identifier;   
(2) gender: "Male", "Female" or "Other";  
(3) age: the age of the patient;  
(4) hypertention: 0 means the patient does not have high blood pressure, 1 means the patient has high blood pressure;    
(5) heart_disease: 0 means the patient does not have any heart disease; 1 means the patient has heart disease;   
(6) ever_married: "No", "Yes" ;    
(7) work_type: "children", "Govt_jov", "Never_worked", "Private";  
(8) Residence_type: "Rural" or "Urban";  
(9) avg_glucose_level: average blood glucose level;  
(10) bmi: body mass index;  
(11) smoked_status: "formerly smoked", "never smoked", "smokes" or "Unknown";  
(12) stroke: 1 if the patient has a stroke, 0 otherwise.   

We removed useless id fields, duplicate values and null values.  The samples with a stroke field of 1 only account for 5.1% of the total sample, so this dataset is severely imbalanced, which will cause the model to consistently judge as 0 on the test set, which is not what we want to see. To solve this problem, we adopt the Borderline-SMOTE2 algorithm to upsample the training set, which is one of the effective methods to solve the imbalance problem of binary classification datasets. As for the division ratio of training set and test set, according to Professor Zhihua Zhou's "Machine Learning": "There is no perfect solution, usually 2/3 to 4/5 of the data set is used for training, and the rest is used for testing" , for the sake of simplicity, we divide the training set and test set according to the ratio of 4:1.  

### Code:  
Use tensorflow2.0 environment.  

The models are implemented using the deepctr package. This research uses the deepctr package to implement the DeepFM model. This package is an easy-to-use and scalable deep learning click-through rate prediction algorithm package developed by Weichen Shen, a computer master from Zhejiang University and a current Alibaba’s algorithm engineer. It can quickly build a CTR prediction algorithm from existing components. model. It can quickly build a click-through rate prediction algorithm model from existing components.  

Deepctr abstracts and summarizes the structure of the existing CTR prediction model based on deep learning, and adopts the idea of modularization in the design process. Each module itself has high reusability, and each module is independent of each other. The CTR prediction model based on deep learning can be divided into the following four modules according to the functions of the internal components of the model: input module, embedding module, feature extraction module, and prediction output module.  

disease.ipynb, which contains a total of 8 models of Logistic Regression, DeepFM, AFM, NFM, DeepFEFM, DIFM, FMPNN, and PNN.  

The end-to-end model DeepFM, whose width and depth parts share the same input. The first-order feature, the second-order and high-order feature interactions are input to the output layer at the same time. Compared to Wide&Deep, DeepFM avoids tedious manual feature engineering.  

The Attention Factorization Machine model AFM further improves the representation ability and interpretability of NFM by introducing the attention mechanism into the bilinear interactive pooling operation, but AFM has no high-order part.  

Neural Factorization Machines (NFMs), which designs a new operation—bilinear interaction (bidirectional interaction) in neural network modeling, thereby incorporating FM into neural networks frame. NFM superimposes a nonlinear layer on a bilinear interaction layer, which can effectively simulate high-order and nonlinear feature interactions and improve the expressiveness of FM.  

Deep Field Embedding Factorization Machine (DeepFEFM). FEFM learns symmetric matrix embedding for each field pair and a single vector embedding for each feature, DeepFEFM uses field pair matrix embedding to generate FEFM interactive embedding, which is combined with feature vector embedding through skip connection layers Generate higher-order feature interactions.  

Dual-input perceptual factorization machines (DIFMs), which can automatically learn different representations of given features according to different input examples. Compared with the Input-aware Factorization Machine (IFM) model, the DIFM model can effectively learn the input perception factors (used to reweight the original feature representation) at both the bit and vector-wise levels.   

PNN is improved on the basis of FNN. Without considering the activation function, FNN combines the features in the way of full connection, which is equivalent to weighted summation of all features, but the "add" operation is not enough to capture the correlation between different field features. Studies have shown that the "product" operation is more effective than the "add" operation, and the advantage of the FM model is reflected by the inner product of the feature vector. Based on this, the PNN author introduced a product layer between the embedding layer and the fully connected layer, that is, adding the function of pair wise crossover when embedding features, the feature interaction between fields is modeled, and the fully connected layer further extracts high order feature patterns.   

FMPNN adds a second-order part on the basis of PNN, which can better capture the interaction of low-order and high-order features, and overcomes the shortcomings of the original PNN's lack of low-order feature capture.  

### Configuration method:  
(1) Configure the tensorflow2.0 environment (mainly on the command line: pip install tensorflow==2.0)  
(2) Enter pip install deepctr on the command line to download the necessary deepctr package  
(3) Put the data set in the location where the file is read in the code  
(4) Upload the code to jupyter  
(5) Open the code and click Run to run  
(6) The final output is the evaluation result and the AUC and loss function diagram on the corresponding training set and test set  

ENFM: Extreme Neural Factorization Machine for Disease Prediction
-------------------
### Code:  
Use tensorflow2.0 environment.  

The models are implemented using the deepctr package. This research uses the deepctr package to implement the DeepFM model. This package is an easy-to-use and scalable deep learning click-through rate prediction algorithm package developed by Weichen Shen, a computer master from Zhejiang University and a current Alibaba’s algorithm engineer. It can quickly build a CTR prediction algorithm from existing components. model. It can quickly build a click-through rate prediction algorithm model from existing components.  

Deepctr abstracts and summarizes the structure of the existing CTR prediction model based on deep learning, and adopts the idea of modularization in the design process. Each module itself has high reusability, and each module is independent of each other. The CTR prediction model based on deep learning can be divided into the following four modules according to the functions of the internal components of the model: input module, embedding module, feature extraction module, and prediction output module.  

The Bi-Interaction Layer in Neural Factorization Machine (NFM) enables a perfect interface between FM and DNN, thus combining FM's ability to model low-order feature interactions and DNN's ability to learn high-order feature interactions and nonlinearities. eXtreme DeepFM (xDeepFM) replaces the Cross Network inside the Deep&Cross (DCN) with this Compressed Interaction Network (CIN), which makes the network capable of learning both explicit and implicit higher-order feature interactions (explicitly by the CIN and implicitly by the DNN). Combining the features of the above models, this work applies the Bi-Interaction Layer to xDeepFM respectively, and proposes a Click-Through-Rate (CTR) prediction model, Extreme Neural Factorization Machine (ENFM), and applies it to depression diagnosis to improve the original efficiency of depression diagnosis. This work will be tested on data (small amount of data and a large number of missing values, which are more difficult to train) from 638 depression-related patients treated in the psychology department of Qingdao Municipal Hospital in the last five years from 2017-2021, and compared with existing excellent and advanced models.

### Configuration method:  
(1) Configure the tensorflow2.0 environment (mainly on the command line: pip install tensorflow==2.0)  
(2) Enter pip install deepctr on the command line to download the necessary deepctr package  
(3) Put the data set in the location where the file is read in the code  
(4) Upload the code to jupyter  
(5) Open the code and click Run to run  
(6) The final output is the evaluation result and the AUC and loss function diagram on the corresponding training set and test set  



