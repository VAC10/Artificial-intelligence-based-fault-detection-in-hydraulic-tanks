
# Hi everyone! It is the project I did during my internship at ibss technology and software company. I will try to explain the stages of the project in this section

## In this project I used python libraries in general (sklearn, matplotlib, seaborn, keras, tensorflow, pandas, numpy...)


* my first goal in the project is to recognize and analyze the data set I have.
* Below are some features of the data set.
![Ekran Görüntüsü (93)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/8cddd73c-df8d-4a4e-8705-bc7fc7296246)

![Ekran Görüntüsü (95)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/9548f463-51c5-4a4d-b629-dbbc911bfc9b)

---

## from now on, since I have a lot of parameters, I will reduce the complexity of my dataset by feature selection and remove features from the dataset that I see will have less impact on the model.

* The outputs I got using the correlation matrix are given below
![Ekran Görüntüsü (98)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/e434cf09-b567-48b4-bb7b-3f2e203e2fa8)![Ekran Görüntüsü (102)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/eefd20fc-bce1-4994-9e58-f1379f82a8fe)

* Based on this matrix, I realized that pressure and tox are inversely proportional to the alarm.
* I removed station cycle times from the data set.

 ---

## Now I will make it more concrete by visualizing the data.
![Ekran Görüntüsü (103)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/d948b727-815d-4e5e-b31f-f89727bfd345)
![Ekran Görüntüsü (106)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/a4a489b0-a574-4ec4-95dd-f46d7ba0c15f)
* In the image above, I would visualize the tox alarm with a scatter plot.
I have two alarm types called line alarm and tox alarm. the reason I chose tox alarm here is that there is only one line feeding this alarm.
I did not visualize the line alarm because there are 5 lines feeding the line alarm and it is too big.
---

## After this stage, there are many types of failures in our data set. The only type of failure requested from me was the machine-induced failure type b-07. Therefore, I filtered out other failure types.
- Below is a visualization of only the failure types of type b-07.
  ![Ekran Görüntüsü (111)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/ee7c8ee0-8820-444c-ba0e-ba4a9fc105b1)
![b0-7](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/68de19de-446f-4eaf-8e3a-a8dc5733247e)

---
## At this stage, I gave the data to the model and wanted to experiment. Although I got 97% accuracy, there was a significant difference in the number of 0 and 1 classes in the confusion matrix and the model predicted the majority class, i.e. 0, much better. This was the Unbalanced problem. I used Under-sample, Over-sample methods to solve this problem.
![Ekran Görüntüsü (113)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/76c8ea9b-65a4-402d-96d1-f97cbf8cf13c)
![Ekran Görüntüsü (116)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/5d40fab9-d07d-4d47-8042-bbdaaad7e037)

* As can be seen in the graphical data above, with the over sample method, the minority class is sentet to the majority class.

---
## After this stage, I have completed the Exploratory Data Analysis stage. Now it's time to put the data into the model and test it!

# MACHİNE LEARNİNG

* I first used machine learning algorithms (Random forest, K-nn, Decision Tree, Naive Bayes, Gradient Boosting, Xgboost) to train the model.
* The results from each algorithm are given below.
![Ekran Görüntüsü (142)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/152a6393-c2c7-4e34-bf6e-ad7729bd2698)
![Ekran Görüntüsü (141)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/c38913e6-7f7e-4ad6-90e5-086e77bb2767)
![Ekran Görüntüsü (139)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/db4e68df-e910-4021-994b-dacf64fd6de1)
![Ekran Görüntüsü (138)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/34c1031a-a50b-441b-8b3d-9eacd87d545c)
![Ekran Görüntüsü (137)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/a83aeae0-49f8-4ed8-9a0d-cd1932945b3a)
![Ekran Görüntüsü (136)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/66434f05-52ee-4aac-b2b6-f35b83fff413)

* The results I got are not bad. But it's not what I wanted. The majority class is more predicted, so I will try different methods.

---
## Since the machine learning algorithms I have tested have not yielded sufficient results, I will build models using Deep learning. I will also experiment with hybrid models. 

# DEEP LEARNİNG
## The models I will use will be the following: CNN,LSTM,RNN,FULL CONNECTED,CNN-LSTM.

* Below are the outputs I got from these models and the confusion matrix and training accuracy- validation accuracy, training loss- validation loss graphs to evaluate the performance of the model against data that the model has never seen.

## LSTM

![Ekran Görüntüsü (144)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/a58b630b-8f19-4771-98d0-b283cc2193b2)
![Ekran Görüntüsü (54)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/08953d0a-4c50-4fed-aae5-a88a1e5da4f2)
![Ekran Görüntüsü (46)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/112f0c08-5a74-4e6b-8425-ad968798c16a)
![Ekran Görüntüsü (45)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/1821a0dc-acf9-41aa-ab1e-440bf0e6fd0c)
---
## FULL CONNECTED
![Ekran Görüntüsü (143)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/5762289e-52af-42a3-aebe-3196a75cd60e)
![Ekran Görüntüsü (44)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/d47fe11d-87ec-4ff0-aed2-a1a2ad0ec763)
![Ekran Görüntüsü (43)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/ae207e63-3c68-4802-b1ff-41e2b454c481)
![Ekran Görüntüsü (42)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/df064a38-090d-4b46-b9d7-8024e085d3a8)

---
# CNN-LSTM
![Ekran Görüntüsü (147)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/3c41e718-46d9-4cc6-a81d-18b822de4e8c)
![Ekran Görüntüsü (55)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/024f0717-42f2-4741-9f96-a540f14fabf4)
![%92 acc (2)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/8f4c611f-00c6-481b-a64c-f9f7141d0bca)
![%92 acc (1)](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/9b358335-3b4d-47d9-9947-863a7235ee7e)
![%91 acc](https://github.com/VAC10/Artificial-intelligence-based-fault-detection-in-hydraulic-tanks/assets/81007065/8559d483-107e-425a-8c07-ced351dca174)

* Using deep learning, I got 93% accuracy with the CNN-LSTM hybrid model.
---  
# Project evaluation
## In this project, I had the opportunity to experience Deep learning and machine learning models by working with real-life irregular complex data and processing stages such as data normalization, EDA, data visualization more closely. This project was very useful for my career. 
* I could have gotten better results by normalizing the data more and using different hybrid models. unfortunately the internship time was enough for this.  

  








