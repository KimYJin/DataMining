# DataMining
## Text Classification with Tensorflow <br><br>

### Corpus Folder <br>
#### This folder is the data provided, which is the news data that has been analyzed by morpheme(NNG/NNP).<br>
#### A total of nine articles are classified into 1,787 categories out of thousands of news articles.<br>
  - Learning Data (Input_DataFolder): 1607<br>
  - Model Learning Assessment Data (Val_Data Folder): 80<br>
  - Evaluation Data (Test_Databolder): 100 <br>
#### News category type<br>
  - children, culture, economy, education, health, life, people, policy, society <br><br>
 
### 1. Feature Set Configuration <br><br>
#### DataMining01.py : Using news articles from folders inside 'Input_Data/Corpus', Find the top 5000 noun morphemes(NNG/NNP) with high frequency.<br>
- List noun morphemes(NNG/NNP) in frequency order
- Arrange in ascending order if frequency is equal
- Save Results to Output01.txt File  <br><br>

### 2. Create learning data & evaluation data for the model <br><br>
#### DataMining02.py : Create learning data using 'Input_Data/Corpus', and create evaluation data using 'Test_Feature_Data/Corpus', 'Val_Feature_Data/Corpus'<br>
#### KimYoonJin Folder : Store the results of the generated learning data and evaluation data.
  
### 2-1. Create learning data for the model <br><br>
- Use Input_Data in the Corpus folder provided to create learning data.
- Create a folder called 'Input_Data' inside the 'KimYoonJin' folder, and create folders corresponding to each category just like the configuration of 'Input_Data/Corpus'.
- Each category stores text files that record the TF-IDF values of each document in each category of 'Input_Data/Corpus'. <br><br>

### 2-2. Create evaluation data for the model <br><br>
- Use Test_Data and Val_Data in the Corpus folder provided to create evaluation data.
- Create a folder with the name 'Test_Feature_Data', 'Val_Feature_Data' inside the 'KimYoonJin' folder.
- Using the evaluation data of 'Test_Data', obtain the TF-IDF value and save it in 'Test_Feature_Data'.
- Using the evaluation data of 'Val_Data', obtain the TF-IDF value and save it in 'Val_Feature_Data'.<br><br>

#### ※ When creating a TF-IDF feature with evaluation data, it is essential that the TF-IDF values are calculated for the 5,000 morphemes(NNG/NNP) determined when generating the learning data. <br>
- That is, when you create a TF-IDF vector for one document,<br>
(1) Calculate the frequency of each of the 5,000 nouns determined in the course of the study (TF). <br>
(2) The IDF values for 5000 nouns in the current document are used in accordance with the IDF values for each morphed element used in the learning process.(IDF) <br>
(3) Since the values of TF and IDF were obtained from the (1)(2), multiply these two values by them and regularize the entire vector. The vectors produced are the TF-IDF vectors for one document in the final evaluation data.<br><br>
