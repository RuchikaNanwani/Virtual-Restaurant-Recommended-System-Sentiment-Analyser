# Virtual-Restaurant-Recommended-System-Sentiment-Analyser

# Requirements

1.	There are some package requirements for this project mentioned in requirements.txt file 

certifi==2021.10.8
click==8.0.4
Flask==2.1.0
h5py==3.6.0
itsdangerous==2.1.2
Jinja2==3.1.1	
Keras==2.7.0
Keras-Preprocessing==1.1.2
requests==2.27.1
tensorflow==2.7.0
tensorflow-estimator==2.7.0
MarkupSafe==2.1.1
Werkzeug==2.0.3
numpy==1.21.5
pandas==1.3.5




2.	app.yaml file contains the python runtime version and instance_Class as high

runtime: python37
instance_class: F4_HIGHMEM

3.	Install and setup Kafka on port 9092 from the URL below.
  	Follow HOW TO INSTALL KAFKA WINDOWS / MAC / LINUX

# Dataset Information

Please refer dataset-final.csv is the final version of dataset. This dataset was obtained by replacing all the user sentiments higher than 3 as positive and the sentiments rating equals to or below 2 as negatives. 
It has three columns which includes ID, Review and Sentiment. 
ID: number assigned for each sentiment
Review: The text review provided by user
Sentiment: This contains values as 0 or 1. ‘0’ being negative and ‘1’ being positive. This field can be considered as label

Please execute below steps to run the project of VRS: Sentiment Analyser

1.	Step 1: Run “python3 preprocess.py dataset-final.csv” in the python terminal. This will generate a pre-processed version of the dataset as “dataset-final-processed.csv”.

2.	Run “python3 statistics.py dataset-final-processed.csv“ in python terminal. This will generate three files. Two of which are pickle files having statistical information about the data and the third file has the list of unique words obtained from our dataset.

•	dataset-final-processed-unique.txt
•	dataset-final-processed-freqdist.pkl
•	dataset-final-processed-freqdist-bi.pkl

# Visualisation and Exploratory Data Analysis

3.	Run “Visualisation_vrs_ruchika.py” to get the data visualisation using dataset-final-processed-freqdist.pkl, dataset-final-processed-freqdist-bi.pkl for plotting unigrams and bigrams.

4.	Run “DataAnalysis.py” to get the detailed statistics of the dataset.

5.	Run “BagOfWords.py” to get the visualisation of words present in the dataset

# LSTM Model

<img width="454" alt="image" src="https://user-images.githubusercontent.com/101907773/166220005-f9f5a8c3-ca0d-405c-aca7-6979b2c76455.png">


6.	Run “LSTM.py” to get the model trained and saved along with saving the tokenizer. Also, training and validation accuracy of 98.43% and minimal visualisation loss is obtained.

7.	Bad model’s python files are in “Bad models” folder. They can be run if needed. But they are those models which didn’t work well and had very low accuracy with over fitting

# Kafka Pipeline

<img width="441" alt="image" src="https://user-images.githubusercontent.com/101907773/166220040-c91daf04-f0e5-4cac-8ca8-64cc0363102b.png">


8.	Install and setup Kafka on port 9092 from the URL below; 

Follow HOW TO INSTALL KAFKA WINDOWS / MAC / LINUX


9.	Run “consumer.py” (consumer) to get the prediction of the review passed by “kafkapython.py” (producer)

10.	Run “kafkapython.py”

# Google Cloud Platform (GCP) Deployment

<img width="454" alt="image" src="https://user-images.githubusercontent.com/101907773/166220093-f3c88d19-0b8f-4f1f-974f-9824c8c9e6f6.png">


Below is the detailed overview of the procedure used in deploying VRS: Sentiment Analyzer as shown in Flowchart 4.

11.	The files involved in hosting the application include the following:
o	main.py, 
o	app. yaml, 
o	BestModel_VRS_Ruchika.json, 
o	BestModel_VRS_Ruchika.h5, 
o	requirements.txt, 
o	tokenizer. json.

•	Requirements.txt: It has all the required packages needed to run the VRS: Sentiment Analyser
•	App.yaml: This file is needed to host the application in GCP. It contains the python version and the name of the instance class
•	Main.py: It has the call for our model. json and .h5 files. It loads the model along with the tokenizer.json file to load the pre-allotted tokens.
•	Templates in HTML format and static, cascading style sheets (CSS)

12.	I have followed these deployment steps for a successful hosting of the VRS: Sentiment Analyzer

a.	Creating an account in Google Cloud Platform (GCP) 
b.	Installing google cloud SDK
c.	Creating a project and a bucket linked to it
d.	“gcloud init” to select the account and project along with the location for hosting 
e.	For creating firewall
i.	“gcloud compute firewall-rules create fw-allow-health-check \
ii.	--network=default \
iii.	--action=allow \
iv.	--direction=ingress \
v.	--source-ranges=130.211.0.0/22,35.191.0.0/16 \
vi.	--target-tags=allow-health-check \
vii.	--rules=tcp”
f.	Deployment of the application
g.	Checking the application logs for troubleshooting the errors. “gcloud app logs tail -s default”
h.	Browsing the application in the default browser “gcloud app browse”
i.	https://vrs-sentiment-analysis.nw.r.appspot.com link can be for browsing the hosted application (please ask me to activate the project before proceeding with the same)

<img width="381" alt="image" src="https://user-images.githubusercontent.com/101907773/166220101-90a12a89-d296-4f46-ae91-0057cefe6631.png">

<img width="372" alt="image" src="https://user-images.githubusercontent.com/101907773/166220109-4cc0c234-6368-4a2f-89a8-d29894660427.png">

<img width="370" alt="image" src="https://user-images.githubusercontent.com/101907773/166220115-751f61d0-2395-4ea8-906e-50f9364a202c.png">

o	requirements.txt, 
