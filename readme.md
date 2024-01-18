# internet_traffic_classification

## task
In this task, here are required to leverage machine learning techniques to distinguish applications in a fine-grained fashion among network flows. There are 3 levels of granularity:
(1) 5- non-vpn2016_toplevel_dev - here will use non-vpn2016 training dataset to train one
own model using top-level annotations (chat, audio, video, etc..) and submit the predictions on the non-vpn2016 test-std set with a JSON file containing your results in the
correct format described on the evaluation page.
(2) 6- non-vpn2016_midlevel_dev - here will use non-vpn2016 training dataset to train one
own model using mid-level annotations (facebook, skype, hangouts, etc..) and submit
your predictions on the non-vpn2016 test-std set with a JSON file containing the results
in the correct format described on the evaluation page.
(3) 7- non-vpn2016_finegrained_dev - here will use non-vpn2016 training dataset to train
one own model using fine-grained annotations (facebook_audio, facebook_chat, skype_audio, skype_chat, etc..) and submit the predictions on the non-vpn2016 test-std
set with a JSON file containing the results in the correct format described on the evaluation page.

----------------

## Overview

- NetML dataset is constructed by selecting several PCAP files from www.stratosphereips.org website.
- non-vpn2016 dataset is the subset of ISCX-VPN-nonVPN2016 dataset from https://www.unb.ca/cic/datasets/vpn.html
For non-vpn2016, three level of annotations are available: top-level, mid-level and fine-grained. Annotations on the training sets are publicly available.

----------------

## Reference
For detailed description about the datasets please check the paper https://arxiv.org/abs/2004.13006

```

----------------

## Challenge Guidelines

### 5- non-vpn2016_toplevel_dev
You will use non-vpn2016 training dataset to train you own model using top-level annotations (chat, audio, video, etc..) and submit your predictions on the non-vpn2016 test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 6- non-vpn2016_midlevel_dev
You will use non-vpn2016 training dataset to train you own model using mid-level annotations (facebook, skype, hangouts, etc..) and submit your predictions on the non-vpn2016 test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 7- non-vpn2016_finegrained_dev
You will use non-vpn2016 training dataset to train you own model using fine-grained annotations (facebook_audio, facebook_chat, skype_audio, skype_chat, etc..) and submit your predictions on the non-vpn2016 test-std set with a JSON file containing your results in the correct format described on the evaluation page.


## Dataset Description

### non-vpn2016
PCAP files are downloaded from https://www.unb.ca/cic/datasets/vpn.html. The original dataset has both vpn and non-vpn packet capture files but we only focus on non-vpn captures. In top-level annotation, we categorize the traffic into 7 groups: audio, chat, email, file_transfer, tor, video, P2P. In mid-level annotation, we group into 18 classes according to the application type such as aim_chat, facebook, hangouts, skype, youtube etc. In fine-level annotation, we treat each action as a different category and obtain 31 classes such as facebook_chat, facebook_video, skype_chat, skype_video etc.

The total number of flows for different splits:
- test-challenge set: 16,323 
- test-std set : 16,323
- traininig set: 131,065

----------------

## Quickstart Guideline

Train a simple Random Forest model using sklearn and create submission.json file.

### Requirements
- python 3
- numpy
- pandas
- sklearn
- matplotlib

please check the requirements.txt in this project.

We recommend you to use virtual environment. For details please see https://docs.python.org/3/tutorial/venv.html

```shell
$ git clone https://github.com/andyZHH/NetSec23-24-Aufgabe4.git
```

### Usage Example
* using the help class for loading or reading dataset.


- Load data and split the validation set:

```python
# Get training data in np.array format
Xtrain, ytrain, class_label_pair, Xtrain_ids = get_training_data(training_set, training_anno_file)

# Split validation set from training data
X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain,
                                                test_size=0.2, 
                                                random_state=42,
                                                stratify=ytrain)
```

The above function uses ./featureDict_META.json file to parse the META features from json.gz files but you are encouraged to write your own parser to include other features as well!

- Preprocess for a better performance:

```python
# Preprocess the data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

- Define and train the model. Print training and validation set accuracy:

```python
# Train a Model  Choose a model (here we use Logistic Regression)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Output accuracy of classifier
print("Training Score: ", model.score(X_train_scaled, y_train))
print("Validation Score: ", model.score(X_val_scaled, y_val))
```

- Plot the confusion matrix and save under ./results/<%Y%m%d-%H%M%S>/CM.png:

```python
# Print Confusion Matrix
ypred = model.predict(X_val_scaled)

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plot_confusion_matrix(directory=save_dir, y_true=y_val, y_pred=ypred, 
                        classes=class_names, 
                        normalize=False)
```

- If you want to create a submission json file for a selected set, the following script handles it and saves the selected json under ./results/<%Y%m%d-%H%M%S>/ folder:

```python
# Make submission with JSON format

do_submit(model, test_set_foldername, scaler, class_label_pair, save_dir+"/submission_test-std-"+anno_level+".json")
```

----------------

### Evaluation
Submission file should be as the following JSON format:

```python
{"id1": "label1", "id2": "label2", ...}
```

For multi-class classification problems, we use F1 score and mean average precision (mAP) as two different metrics.

F1=2\*precision\*recall/(precision+recall)

where:

- precision=TP/(TP+FP)
- recall=TP/(TP+FN)

- True Positive [TP] = your prediction is 1, and the ground truth is also 1 - you predicted a positive and that's true!
- False Positive [FP] = your prediction is 1, and the ground truth is 0 - you predicted a positive, and that's false.
- False Negative [FN] = your prediction is 0, and the ground truth is 1 - you predicted a negative, and that's false.

mAP = 1/N\*(\sum(AP<sub>i</sub>))

where:

- N: number of classes
- AP: average precision for each class

'overall' score is bounded between 0 and 1 and higher value represents more accurate results. It is calculated by multiplying the two scores: 

overall = F1\*mAP

----------------

For binary classifications, in other words detection problems, we use True Positive Rate (TPR) as detection rate of malware and False Alarm Rate as two metrics.

- TPR = TP/(TP+FN)
- FAR = FP/(TN+FP)

'overall' score is bounded between 0 and 1 and higher value represents more accurate results. It is calculated by multiplying detection rate with the 1-FAR: 

overall = TPR\*(1-FAR)

----------------

## Directory Structure
* NetSec23-24-Aufgabe4
    - readme.md
    - internet_traffic_classification.ipynb
    * non-vpn2016
            - ...
    * results
        * fine
			-submission_test-std-fine.json
            - CM.png
        * mid
			-submission_test-std-mid.json
            - CM.png
        * top
			-submission_test-std-top.json
            - CM.png
    * featureDict_META.json
    * helper.py

----------------

## License

BSD
