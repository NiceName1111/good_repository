**Overall Content**:
1. Project proposal and introduction
   - topic and questions
   - hypotheses
   - introduction to data collection
2. Preparation
   - upgrade necessary packages
   - import necessary packages
3. Coding Part
   - introduction of methods
   - Data cleaning
   - Data processing
   - Result exhibitions

# **PART 1**: Project Proposal and Introduction

## 1.1 Topic and Research Question
The Federal Open Market Committee (FOMC) plays a critical role in displaying the policymakers’ consideration of U.S. monetary policy, and its releases are always scrutinized word by word by market participants to interpret the direction of interest rates or other economic measures. Its contents often influence the equity market, exchange rate, and credit swap market significantly. However, while qualitative interpretations abound, few studies have applied quantitative or computational methods to analyze these press releases systematically. <br> <br>
The goal of this project is to utilize computational tools to analyze (1) **FOMC policy statement** and  (2) **minutes** to identify patterns in the language that could signal incoming monetary policy actions. Specifically, I aim to answer the following research question:<br> **Can changes in the language and tone of FOMC press releases imply future policy interest rate changes?**



## 1.2 Hypotheses
The project will test two primary hypotheses:<br>
- Hypothesis 1: **An increase in the use of positive language describing economic conditions is associated with a reduced likelihood of an imminent rate cut.**
- Hypothesis 2: A **significant shift in the language used to describe the economic outlook** (e.g., from positive to cautious that describes the concern of a potential economic downturn) **signals a high probability of changes in policy direction in the coming FOMC meetings**.

These hypotheses are based on the assumption that the FOMC, while remaining cautious and neutral in tone, still tries to clearly signal its future policy intentions and its expectation of market performance through its choice of words. At the same time, monetary tightening and easing would consider if the economic outlook can stand the pressure or stimulation from the monetary polcies

## 1.3 Data Collection
### **Text data**
The text data for this project will consist of FOMC policy statements and minutes since 2000. <br>
The following link is the source of the data: https://raw.githubusercontent.com/vtasca/fed-statement-scraping/master/communications.csv <br>
Reasons to pick these data:
1. They cover four rounds of rate hikes and rate cuts covering decisions by Alan Greenspan, Ben Beranke, Janet Yellen, and Jay Powell. Therefore, they are typical
2. Sample size cover over 440 minutes + statements, the size is large enough to extract useful information

The data will be used to testify the hypothesis through 3 rounds of test:
1. Keyword frequency test
2. Sentiment Analysis
3. TF-IDF test

The mainway of showing the result is through the visualizations, which contains charts and two-axis graph



### **Rate data**
Secondly, the numeric data for this project is the federal fund rate set by the Fed since 2000.(Board of Governors of the Federal Reserve System (US),2024) The sources is the following:
1. [Federal Funds Target Range - Lower Limit (data since Dec 2008)](https://fred.stlouisfed.org/series/DFEDTARL)
2. [Federal Funds Target Rate (data before Dec 2008)](https://fred.stlouisfed.org/series/DFEDTAR)

The sample size is greater than 9000

The federal fund rate data contains two parts because the Fed set out a reform in 2008, changing the policy rate from setting a specific rate to setting a target rate range

# **PART 2**: Preparation
This step will include the following work:
1. upgrade the necessary landguage processing package
2. import the usable package and diagnose any issue with using the package

## 2.1: package upgrade: nltk


```python
%matplotlib inline
%matplotlib notebook
```


```python
# upgrade the necessary package nltk for language processing
!pip install --upgrade nltk
```

    Requirement already satisfied: nltk in /usr/local/lib/python3.10/dist-packages (3.9.1)
    Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk) (8.1.7)
    Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk) (1.4.2)
    Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.10/dist-packages (from nltk) (2024.9.11)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk) (4.66.6)
    

## 2.2: import the necessary package


```python
# import the necessary package
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
```


```python
# load spacy to recignize the name entities and process data cleaning
import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")
```


```python
# use for stop words removal
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
# to read the data csv stored in Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
# For data visualization if needed
import matplotlib.pyplot as plt
```


```python
# For two axis graph
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```


```python
# For tf-idf test
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import altair as alt
import numpy as np
pd.options.display.max_rows = 600
from pathlib import Path
import glob
```


```python
!pip install vaderSentiment
```

    Requirement already satisfied: vaderSentiment in /usr/local/lib/python3.10/dist-packages (3.3.2)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from vaderSentiment) (2.32.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->vaderSentiment) (3.4.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->vaderSentiment) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->vaderSentiment) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->vaderSentiment) (2024.8.30)
    


```python
# for sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
```

## 2.3: import the data

**Note on interest rate data**:<br>
There are two sets of data because the Fed change the policy in Dec 2008. Before that, the policy rates were announced as a specific rate; but after that, the policy rates are announced as a rate range.

### 2.3.1 Preparing interest rate data
The procedure in this part is to:
1. read the interest rate data
2. merge two datasets to create a continuous interest rate data from 2000


```python
# import the interest rate data
old_rates = pd.read_csv('/content/drive/MyDrive/text_analysis/rate_new.csv')
new_rates = pd.read_csv('/content/drive/MyDrive/text_analysis/rate_oid.csv')

# Now you can work with the dataframes old_rates and new_rates
print(old_rates.head())
print(new_rates.head())
```

      observation_date  DFEDTARL
    0       2008-12-16       0.0
    1       2008-12-17       0.0
    2       2008-12-18       0.0
    3       2008-12-19       0.0
    4       2008-12-20       0.0
      observation_date  DFEDTAR
    0       2000-01-01      5.5
    1       2000-01-02      5.5
    2       2000-01-03      5.5
    3       2000-01-04      5.5
    4       2000-01-05      5.5
    


```python
old_rates['Date'] = pd.to_datetime(old_rates['observation_date'])
new_rates['Date'] = pd.to_datetime(new_rates['observation_date'])
```


```python
old_rates.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5842 entries, 0 to 5841
    Data columns (total 3 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   observation_date  5842 non-null   object        
     1   DFEDTARL          5842 non-null   float64       
     2   Date              5842 non-null   datetime64[ns]
    dtypes: datetime64[ns](1), float64(1), object(1)
    memory usage: 137.0+ KB
    


```python
new_rates.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3272 entries, 0 to 3271
    Data columns (total 3 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   observation_date  3272 non-null   object        
     1   DFEDTAR           3272 non-null   float64       
     2   Date              3272 non-null   datetime64[ns]
    dtypes: datetime64[ns](1), float64(1), object(1)
    memory usage: 76.8+ KB
    


```python
# Merge the dataframes
merged_rates = pd.merge(old_rates, new_rates, on='Date', how='outer')

# Create the final dataframe with the specified columns
final_rates = pd.DataFrame()
final_rates['Date'] = merged_rates['Date']
final_rates['Policy Rate'] = merged_rates['DFEDTARL'].combine_first(merged_rates['DFEDTAR'])

# Display the final dataframe
final_rates.sample(5)
```





  <div id="df-e92b280e-7301-45e7-9f2c-015b8ce8f9e8" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Policy Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1313</th>
      <td>2003-08-06</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>3439</th>
      <td>2009-06-01</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>8500</th>
      <td>2023-04-10</td>
      <td>4.75</td>
    </tr>
    <tr>
      <th>2682</th>
      <td>2007-05-06</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>5844</th>
      <td>2016-01-01</td>
      <td>0.25</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e92b280e-7301-45e7-9f2c-015b8ce8f9e8')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e92b280e-7301-45e7-9f2c-015b8ce8f9e8 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e92b280e-7301-45e7-9f2c-015b8ce8f9e8');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-6c65ec98-3207-430c-842d-db2ff0bd6904">
  <button class="colab-df-quickchart" onclick="quickchart('df-6c65ec98-3207-430c-842d-db2ff0bd6904')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-6c65ec98-3207-430c-842d-db2ff0bd6904 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
final_rates.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9114 entries, 0 to 9113
    Data columns (total 2 columns):
     #   Column       Non-Null Count  Dtype         
    ---  ------       --------------  -----         
     0   Date         9114 non-null   datetime64[ns]
     1   Policy Rate  9114 non-null   float64       
    dtypes: datetime64[ns](1), float64(1)
    memory usage: 142.5 KB
    

### 2.3.2 Preparing the text data


```python
# import the data and make a preview
read_fomc = pd.read_csv('https://raw.githubusercontent.com/vtasca/fed-statement-scraping/master/communications.csv')
```

# **PART 3**: Coding Part and Methods

## 3.1 Introduction of Methods

### **Data Cleaning methods**
This project utilizes the following methods to clean the data accordingly:
  - Regular Expression
    - All the minutes contains over 15 lines of introductions to the meeting participants. All these lines share a common feature - start with participants' full name. Therefore, this program will use regular expression to identify such a combination. Because people may have middle name and names with French charaters, the regular expression used to identify the names includes four line designed to identify different combination of names in the documents
  - Stop words
    - This program remove the vocabuary that does not contain clear information by utilizin the stop words dictionary
  - Key phrase
    - In all the minutes, the content after "_________________" are not the main context of the document. Therefore, this program will delete the line after this line
    - In all the statement, the content after the line showing the voting result are not useful for text analysis. Therefore, this program will delete the lines after the line starting with "Voting for"
  - spacy
    - spacy cannot fully recognize and delete all the lines starting with names. Meanwhile, the regular expressions cannot help to identify all the lines starting with some names that contains charaters my computer cannot easily type. However, when the program double delete the lines beginning with names by utilizing both spacy and regular expression, the performance is clear and outstanding. All the lines starting with names are deleted.
  - isalpha()
    - used to clean away the numeric conponents within the phrase


After the data cleaning, I have pick few samples to check if the deletion is complete and if the data cleaning would delete the useful text by error. So far, the data cleaning process runs well

### **Data Analysis methods**
This project contains 3 rounds of data analysis:
1. Word Frequency Test
  - the key words of word frequency are "unemployment" and "inflation." These two words best reflect the dual mandates of the Fed because unemployment rate and inflation rate are two indicators and expression used often to describe overall market performance
    - [The Federal Reserve's Dual Mandate (Federal Reserve Bank of Chicago, 2020)](https://www.chicagofed.org/research/dual-mandate/dual-mandate#:~:text=Our%20two%20goals%20of%20price%20stability%20and%20maximum,broad%20concepts%20into%20specific%20longer-run%20goals%20and%20strategies.)
  - the main methods is the built-in function count() and list comprehension to create a new dataframe for data visualization later
  - data visualization includes two analytical result graphs for statament texts and minute texts respectively
2. TF-IDF Test
  - this part use tf-idf test
  - but before processing the data, the program set up a list of words to be removed before process tf-idf test. These words include committee”, “participants", "reserve", "member", "rate", "committee", "federal", "member.' They appear so frequently in many documents that occupy the top ti-idf score in many documents. However, they do not contains much information about policy attitude. Therefore, I choose to remove them from the analysis.
  - nested for loop is also used in this test to testify the score based on individual words
3. Sentiment Test
  - This test uses the [financial sentiment dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/), which is more accurate in calculating the sentiment score for financial documents.
  - data visualization uses two-axis graph to show the movements relationship between sentiment score and interest rates.

The main way of showing the resutls: graph visualization. There are several resons that using graph is a better way of display the analytical result than using regression:
1. Word frequency and sentiment score fluctuate dramatically from one to another FOMC meeting. However, their trends are clear when looking at the big picture of the movement. Therefore, using graph is better to demonstrate the potential correlation
2. Economic shocks have impact the policy reaction differently. Economic shock such as war, financial crisis, and real estate bubble burst happened suddency in history, causing the policy rate to change immediately at the time. These changes are different from the rate changes in 2015 and 2022, which are influenced by the observation of gradual economic changes. Thus, using regression model would fail to find a proper predicting object.

## 3.2 Data Cleaning

### 3.2.1 Data Preview
#### **Note:** This procedure is to have a preview of the data and choose the corresponding methods to commit data cleaning for the later analysis


```python
read_fomc.sample(2)
```





  <div id="df-becc67c9-1261-43f5-8c5c-d99a33736615" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Release Date</th>
      <th>Type</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>64</th>
      <td>2020-11-05</td>
      <td>2020-11-25</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
    </tr>
    <tr>
      <th>168</th>
      <td>2014-09-17</td>
      <td>2014-10-08</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-becc67c9-1261-43f5-8c5c-d99a33736615')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-becc67c9-1261-43f5-8c5c-d99a33736615 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-becc67c9-1261-43f5-8c5c-d99a33736615');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-7aa58152-c71a-4dc6-b3e1-8ac52ac3b927">
  <button class="colab-df-quickchart" onclick="quickchart('df-7aa58152-c71a-4dc6-b3e1-8ac52ac3b927')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-7aa58152-c71a-4dc6-b3e1-8ac52ac3b927 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
read_fomc['Date'] = pd.to_datetime(read_fomc['Date'])
read_fomc.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 440 entries, 0 to 439
    Data columns (total 4 columns):
     #   Column        Non-Null Count  Dtype         
    ---  ------        --------------  -----         
     0   Date          440 non-null    datetime64[ns]
     1   Release Date  411 non-null    object        
     2   Type          440 non-null    object        
     3   Text          440 non-null    object        
    dtypes: datetime64[ns](1), object(3)
    memory usage: 13.9+ KB
    


```python
read_fomc['Type'].unique()
```




    array(['Minute', 'Statement'], dtype=object)



#### **Note on data**
The data contain for series:
1. Date: Record the dates of the FOMC corresponding to each statement and minute
2. Release Date: Record the release dates of the documents. Usually, the policy statement is released on the same day of the FOMC, and the minute is released few weeks afterward
3. Type: Record the types of the documents, categorizing them into 'Minute' and 'Statement'
4. Text: the content of the document

### 3.2.2 Text Preview
#### **Note**: this part will print out the texts of a minute and a statement. This step will check what lines in the texts need to be removed so that the contexts left are all useful for text analysis

##### **Example of a minute**


```python
print(read_fomc.iloc[202]['Text'])
```

    Minutes of the Federal Open Market Committee
                                                    
                        
                        
                        
                        September 12-13, 2012
    
    A meeting of the Federal Open Market Committee was held in the offices of the Board of Governors of the Federal Reserve System in Washington, D.C., on Wednesday, September 12, 2012, at 10:30 a.m. and continued on Thursday, September 13, 2012, at 8:30 a.m.
    
    PRESENT:
    Ben Bernanke, Chairman
    William C. Dudley, Vice Chairman
    Elizabeth Duke
    Jeffrey M. Lacker
    Dennis P. Lockhart
    Sandra Pianalto
    Jerome H. Powell
    Sarah Bloom Raskin
    Jeremy C. Stein
    Daniel K. Tarullo
    John C. Williams
    Janet L. Yellen
    
    James Bullard, Christine Cumming, Charles L. Evans, Esther L. George, and Eric Rosengren, Alternate Members of the Federal Open Market Committee
    
    Richard W. Fisher, Narayana Kocherlakota, and Charles I. Plosser, Presidents of the Federal Reserve Banks of Dallas, Minneapolis, and Philadelphia, respectively
    
    William B. English, Secretary and Economist
    Deborah J. Danker, Deputy Secretary
    Matthew M. Luecke, Assistant Secretary
    David W. Skidmore, Assistant Secretary
    Michelle A. Smith, Assistant Secretary
    Scott G. Alvarez, General Counsel
    Thomas C. Baxter, Deputy General Counsel
    Steven B. Kamin, Economist
    David W. Wilcox, Economist
    
    David Altig, Thomas A. Connors, Michael P. Leahy, William Nelson, David Reifschneider, Glenn D. Rudebusch, William Wascher, and John A. Weinberg, Associate Economists
    
    Simon Potter, Manager, System Open Market Account
    
    Nellie Liang, Director, Office of Financial Stability Policy and Research, Board of Governors
    
    Jon W. Faust, Special Adviser to the Board, Office of Board Members, Board of Governors
    
    James A. Clouse, Deputy Director, Division of Monetary Affairs, Board of Governors; Maryann F. Hunter, Deputy Director, Division of Banking Supervision and Regulation, Board of Governors
    
    Andreas Lehnert,1 Deputy Director, Office of Financial Stability Policy and Research, Board of Governors
    
    Linda Robertson, Assistant to the Board, Office of Board Members, Board of Governors
    
    Seth B. Carpenter, Senior Associate Director, Division of Monetary Affairs, Board of Governors
    
    Thomas Laubach, Senior Adviser, Division of Research and Statistics, Board of Governors; Ellen E. Meade and Joyce K. Zickler, Senior Advisers, Division of Monetary Affairs, Board of Governors
    
    Brian J. Gross,2 Special Assistant to the Board, Office of Board Members, Board of Governors
    
    Eric M. Engen, Michael G. Palumbo, and Wayne Passmore, Associate Directors, Division of Research and Statistics, Board of Governors
    
    Fabio M. Natalucci, Deputy Associate Director, Division of Monetary Affairs, Board of Governors
    
    Edward Nelson, Section Chief, Division of Monetary Affairs, Board of Governors
    
    Jeremy B. Rudd, Senior Economist, Division of Research and Statistics, Board of Governors
    
    Kelly J. Dubbert, First Vice President, Federal Reserve Bank of Kansas City
    
    Loretta J. Mester, Harvey Rosenblum, and Daniel G. Sullivan, Executive Vice Presidents, Federal Reserve Banks of Philadelphia, Dallas, and Chicago, respectively
    
    Cletus C. Coughlin, Troy Davig, Mark E. Schweitzer, and Kei-Mu Yi, Senior Vice Presidents, Federal Reserve Banks of St. Louis, Kansas City, Cleveland, and Minneapolis, respectively
    
    Lorie K. Logan, Jonathan P. McCarthy, Giovanni Olivei, and Nathaniel Wuerffel,3 Vice Presidents, Federal Reserve Banks of New York, New York, Boston, and New York, respectively
    
    Michelle Ezer,4 Markets Officer, Federal Reserve Bank of New York
    
    Potential Effects of a Large-Scale Asset Purchase Program
    The staff presented an analysis of various aspects of possible large-scale asset purchase programs, including a comparison of flow-based purchase programs to programs of fixed size. The presentation reviewed the modeling approach used by the staff in estimating the financial and macroeconomic effects of such purchases. While significant uncertainty surrounds such estimates, the presentation indicated that asset purchases could be effective in fostering more rapid progress toward the Committee's objectives. The staff noted that, for a flow-based program, the public's understanding of the conditions under which the Committee would end purchases would shape expectations of the magnitude of the Federal Reserve's holdings of longer-term securities, and thus also influence the financial and economic effects of such a program. The staff also discussed the potential implications of additional asset purchases for the evolution of the Federal Reserve's balance sheet and income. The presentation noted that significant additional asset purchases should not adversely affect the ability of the Committee to tighten the stance of policy when doing so becomes appropriate. In their discussion of the staff presentation, a few participants noted the uncertainty surrounding estimates of the effects of large-scale asset purchases or the need for additional work regarding the implications of such purchases for the normalization of policy.
    
    Developments in Financial Markets and the Federal Reserve's Balance Sheet
    The Manager of the System Open Market Account (SOMA) reported on developments in domestic and foreign financial markets during the period since the Federal Open Market Committee (FOMC) met on July 31-August 1, 2012. He also reported on System open market operations, including the ongoing reinvestment into agency-guaranteed mortgage-backed securities (MBS) of principal payments received on SOMA holdings of agency debt and agency-guaranteed MBS as well as the operations related to the maturity extension program authorized at the June 19-20, 2012, FOMC meeting. By unanimous vote, the Committee ratified the Desk's domestic transactions over the intermeeting period. There were no intervention operations in foreign currencies for the System's account over the intermeeting period.
    
    Staff Review of the Economic Situation
    The information reviewed at the September 12-13 meeting suggested that economic activity continued to increase at a moderate pace in recent months. Employment rose slowly, and the unemployment rate was still high. Consumer price inflation stayed subdued, while measures of long-run inflation expectations remained stable.
    
    Private nonfarm employment increased in July and August at only a slightly faster pace than in the second quarter, and the rate of decline in government employment eased somewhat. The unemployment rate was 8.1 percent in August, just a bit lower than its average during the first half of the year, and the labor force participation rate edged down further. The share of workers employed part time for economic reasons remained large, and the rate of long-duration unemployment continued to be high. Indicators of job openings and firms' hiring plans were little changed, on balance, and initial claims for unemployment insurance were essentially flat over the intermeeting period.
    
    Manufacturing production increased at a faster pace in July than in the second quarter, and the rate of manufacturing capacity utilization rose slightly. However, automakers' schedules indicated that the pace of motor vehicle assemblies would be somewhat lower in the coming months than it was in July, and broader indicators of manufacturing activity, such as the diffusion indexes of new orders from the national and regional manufacturing surveys, generally remained quite muted in recent months at levels consistent with only meager gains in factory output in the near term.
    
    Following a couple of months when real personal consumption expenditures (PCE) were roughly flat, spending increased in July, and the gains were fairly widespread across categories of consumer goods and services. Incoming data on factors that tend to support household spending were somewhat mixed. Real disposable incomes increased solidly in July, boosted in part by lower energy prices. The continued rise in house values through July, and the increase in equity prices during the intermeeting period, suggested that households' net worth may have improved a little in recent months. However, consumer sentiment remained more downbeat in August than earlier in the year.
    
    Housing market conditions continued to improve, but construction activity was still at a low level, reflecting the restraint imposed by the substantial inventory of foreclosed and distressed properties and by tight credit standards for mortgage loans. Starts of new single-family homes declined in July, but permits increased, which pointed to further gains in single-family construction in the coming months. Both starts and permits for new multifamily units rose in July. Home prices increased for the sixth consecutive month in July, and sales of both new and existing homes also rose.
    
    Real business expenditures on equipment and software appeared to be decelerating. Both nominal shipments and new orders for nondefense capital goods excluding aircraft declined in July, and the backlog of unfilled orders decreased. Other forward-looking indicators, such as downbeat readings from surveys of business conditions and capital spending plans, also pointed toward only muted increases in real expenditures for business equipment in the near term. Nominal business spending for new nonresidential construction declined in July after only edging up in the second quarter. Inventories in most industries looked to be roughly aligned with sales in recent months.
    
    Real federal government purchases appeared to decrease further, as data for nominal federal spending in July pointed to continued declines in real defense expenditures. Real state and local government purchases also appeared to still be trending down. State and local government payrolls contracted in July and August, although at a somewhat slower rate than in the second quarter, and nominal construction spending by these governments decreased slightly in July.
    
    The U.S. international trade deficit was about unchanged in July after narrowing significantly in June. Exports declined in July, as decreases in the exports of industrial supplies, automotive products, and consumer goods were only partially offset by greater exports of agricultural products. Imports also declined in July, reflecting lower imports of capital goods and petroleum products and somewhat higher imports of automotive products. The trade data for July pointed toward real net exports having a roughly neutral effect on the growth of U.S. real gross domestic product (GDP) in the third quarter after they made a positive contribution to the increase in real GDP in the second quarter.
    
    Overall U.S. consumer prices, as measured by the PCE price index, were flat in July. Consumer food prices were essentially unchanged, but the substantial increases in spot and futures prices of farm commodities in recent months, reflecting the effects of the drought in the Midwest, pointed toward some temporary upward pressures on retail food prices later this year. Consumer energy prices declined slightly in July, but survey data indicated that retail gasoline prices rose in August. Consumer prices excluding food and energy also were flat in July. Near-term inflation expectations from the Thomson Reuters/University of Michigan Surveys of Consumers increased somewhat in August, while longer-term inflation expectations in the survey edged up but remained within the narrow range that they have occupied for many years. Long-run inflation expectations from the Federal Reserve Bank of Philadelphia Survey of Professional Forecasters continued to be stable in the third quarter.
    
    Measures of labor compensation indicated that increases in nominal wages remained modest. The rise in compensation per hour in the nonfarm business sector was muted over the year ending in the second quarter, and with small gains in productivity, unit labor costs rose only slightly. The employment cost index increased a little more slowly than the measure of compensation per hour over the same period. More recently, the gains in average hourly earnings for all employees in July and August were small.
    
    Overall foreign economic growth appeared to be subdued in the third quarter after slowing in the second quarter. In the euro area, policy developments contributed to an improvement in financial conditions; recent indicators pointed to further decreases in production, however, and both business and consumer confidence continued to decline. Indicators of activity in the emerging market economies generally weakened. In China, export growth slowed, while retail sales and investment spending changed little. The rate of economic growth rose in Brazil but was still sluggish, and increases in economic activity in Mexico were below the faster pace seen earlier in the year. Consistent with the slowing in foreign economic growth, readings on foreign inflation continued to moderate.
    
    Staff Review of the Financial Situation
    Sentiment in financial markets improved somewhat since the time of the August FOMC meeting. Investors' concerns about the situation in Europe seemed to ease somewhat, and market participants also appeared to have increased their expectations of additional monetary policy accommodation.
    
    On balance, the nominal Treasury yield curve steepened over the intermeeting period, with yields on longer-dated Treasury securities rising notably. Following the August FOMC statement, Treasury yields moved up, reportedly in part because investors had factored in some probability that the anticipated liftoff date for the federal funds rate in the forward-guidance language would be moved back at that meeting. Treasury yields subsequently rose further as concerns about the situation in the euro area moderated. Later in the period, Treasury yields retraced some of their earlier gains as market participants' expectations of additional policy action increased following the release of the minutes of the August FOMC meeting, the Chairman's speech at the economic symposium in Jackson Hole, and the weaker-than-expected August employment report. On net, the expected path of the federal funds rate derived from overnight index swap rates was little changed. Indicators of inflation expectations derived from nominal and inflation-protected Treasury securities edged up over the period but stayed in the ranges observed over recent quarters.
    
    Conditions in unsecured short-term dollar funding markets remained stable over the intermeeting period. In secured funding markets, conditions were also little changed.
    
    In the September Senior Credit Officer Opinion Survey on Dealer Financing Terms, respondents reported no significant changes in credit terms for important classes of counterparties over the past three months, although a few noted a slight easing in terms for some clients. The use of leverage by hedge funds was reported to have remained basically unchanged. However, respondents noted greater demand for funding of agency and non-agency residential MBS.
    
    Broad price indexes for U.S. equities rose moderately, on net, over the intermeeting period, prompted by generally better-than-expected readings on economic activity released early in the period, somewhat reduced concerns about the situation in Europe, and some additional anticipation of monetary policy easing later in the period. Option-implied volatility on the S&P 500 index fell in early August to levels not seen since the middle of 2007; it subsequently partially retraced. Equity prices for large domestic banks rose about in line with the broad equity price indexes, and credit default swap (CDS) spreads for the largest bank holding companies continued to move down.
    
    Yields on investment-grade corporate bonds were little changed at near-record low levels over the intermeeting period, while yields on speculative-grade corporate bonds edged down. The spread of yields on corporate bonds over those on comparable-maturity Treasury securities narrowed. Net debt issuance by nonfinancial firms continued to be strong over the period. Investment- and speculative-grade bond issuance increased in August from an already robust pace in preceding months, and commercial and industrial (C&I) loans rose further. In the syndicated leveraged loan market, gross issuance of institutional loans continued to be solid in July and August. Issuance of collateralized loan obligations remained on pace to post its strongest year since 2007. The rate of gross public equity issuance by nonfinancial firms increased slightly in August but was still at a subdued level.
    
    Financial conditions in the commercial real estate (CRE) market were still somewhat strained against a backdrop of weak fundamentals and tight underwriting standards. Nevertheless, issuance of commercial mortgage-backed securities continued at a solid pace over the intermeeting period.
    
    Mortgage rates remained at very low levels over the intermeeting period. Refinancing activity increased but was still restrained by tight underwriting conditions, capacity constraints at mortgage originators, and low levels of home equity. Nonrevolving consumer credit continued to expand briskly in June, largely due to robust growth in student loans originated by the federal government, while revolving credit remained subdued. Delinquency rates for consumer credit were still low, mostly reflecting a shift in lending toward higher-credit-quality borrowers.
    
    Gross issuance of long-term municipal bonds picked up in August from the subdued pace in July, but net issuance continued to decline. CDS spreads for debt issued by state governments moved lower over the intermeeting period, and the ratio of yields on long-term general obligation municipal bonds to yields on comparable-maturity Treasury securities decreased, on balance.
    
    Bank credit continued to expand at a moderate pace over the intermeeting period, as growth in C&I loans remained brisk while CRE and home equity loans both trended down further. The August Survey of Terms of Business Lending indicated that overall interest-rate spreads on C&I loans were little changed; spreads on loans drawn on recently established commitments narrowed materially, although they remained wide.
    
    M2 growth was rapid in July, likely reflecting investors' heightened demand for safe and liquid assets amid concerns about the situation in Europe, but it slowed to a moderate pace in August as those concerns eased somewhat. The monetary base rose in July and August as reserve balances and currency expanded.
    
    Sentiment improved in foreign financial markets as the European Central Bank (ECB) outlined a plan to make additional sovereign bond purchases in conjunction with the European Financial Stability Facility and the European Stability Mechanism. Spreads of shorter-term yields on peripheral euro-area sovereign bonds over those on comparable-maturity German bunds declined substantially over the period. The staff's broad nominal index of the foreign exchange value of the dollar declined and benchmark sovereign yields in the major advanced foreign economies increased as safe-haven demands eased with the lessening of concerns about the European situation. Most global benchmark indexes for equity prices moved up, and the equity prices of European banks rose sharply. Funding conditions for euro-area banks improved, although these conditions remained fragile, and draws on the Federal Reserve's liquidity swap facility with the ECB fell.
    
    The staff also reported on potential risks to financial stability, including those owing to the developments in Europe and to the current environment of low interest rates. Although the support for economic activity provided by low interest rates enhances financial stability, low interest rates also could eventually contribute to excessive borrowing or risk-taking and possibly leave some aspects of the financial system vulnerable to a future rise in interest rates. The staff surveyed a wide range of asset markets and financial institutions for signs of excessive valuations, leverage, or risk-taking that could pose systemic risks. Valuations for broad asset classes did not appear stretched, or supported by excessive leverage. The staff also did not find evidence that excessive risk-taking was widespread, although such behavior had appeared in a few smaller and less liquid markets.
    
    Staff Economic Outlook
    In the economic projection prepared by the staff for the September FOMC meeting, the forecast for real GDP growth in the near term was broadly similar, on balance, to the previous projection. The near-term forecast incorporated a larger negative effect of the drought on farm output in the second half of this year than the staff previously anticipated, but this effect was mostly offset by the staff's expectation of a smaller drag from net exports. The staff's medium-term projection for real GDP growth, which was conditioned on the assumption of no changes in monetary policy, was revised up a little, mostly reflecting a slight improvement in the outlook for the European situation and a somewhat higher projected path for equity prices. Nevertheless, with fiscal policy assumed to be tighter next year than this year, the staff expected that increases in real GDP would not materially exceed the growth of potential output in 2013. In 2014, economic activity was projected to accelerate gradually, supported by an easing in fiscal policy restraint, increases in consumer and business confidence, further improvements in financial conditions and credit availability, and accommodative monetary policy. The expansion in economic activity was expected to narrow the significant margin of slack in labor and product markets only slowly over the projection period, and the unemployment rate was anticipated to still be elevated at the end of 2014.
    
    The staff's near-term forecast for inflation was revised up from the projection prepared for the August FOMC meeting, reflecting increases in consumer energy prices that were greater than anticipated. However, the staff's projection for inflation over the medium term was little changed. With crude oil prices expected to gradually decline from their current levels, the boost to retail food prices from the drought anticipated to be only temporary and comparatively small, long-run inflation expectations assumed to remain stable, and substantial resource slack persisting over the projection period, the staff continued to forecast that inflation would be subdued through 2014.
    
    The staff viewed the uncertainty around the forecast for economic activity as elevated and the risks skewed to the downside, largely reflecting concerns about the situation in Europe and the possibility of a more severe tightening in U.S. fiscal policy than anticipated. Although the staff saw the outlook for inflation as uncertain, the risks were viewed as balanced and not unusually high.
    
    Participants' Views on Current Conditions and the Economic Outlook
    In conjunction with this FOMC meeting, meeting participants--the 7 members of the Board of Governors and the presidents of the 12 Federal Reserve Banks, all of whom participate in the deliberations of the FOMC--submitted their assessments of real output growth, the unemployment rate, inflation, and the target federal funds rate for each year from 2012 through 2015 and over the longer run, under each participants' judgment of appropriate monetary policy. The longer-run projections represent each participant's assessment of the rate to which each variable would be expected to converge, over time, under appropriate monetary policy and in the absence of further shocks to the economy. These economic projections and policy assessments are described in the Summary of Economic Projections, which is attached as an addendum to these minutes.
    
    In their discussion of the economic situation and outlook, meeting participants regarded the information received during the intermeeting period as indicating that economic activity had continued to expand at a moderate pace in recent months. However, recent gains in employment were small and the unemployment rate remained high. Although consumer spending had continued to advance, growth in business fixed investment appeared to have slowed. The housing sector showed some further signs of improvement, albeit from a depressed level. Consumer price inflation had been subdued despite recent increases in the prices of some key commodities, and longer-term inflation expectations had remained stable.
    
    Regarding the economic outlook, participants generally agreed that the pace of the economic recovery would likely remain moderate over coming quarters but would pick up over the 2013-15 period. In the near term, the drought in the Midwest was expected to weigh on economic growth. Moreover, participants observed that the pace of economic recovery would likely continue to be held down for some time by persistent headwinds, including continued weakness in the housing market, ongoing household sector deleveraging, still-tight credit conditions for some households and businesses, and fiscal consolidation at all levels of government. Many participants also noted that a high level of uncertainty regarding the European fiscal and banking crisis and the outlook for U.S. fiscal and regulatory policies was weighing on confidence, thereby restraining household and business spending. However, others questioned the role of uncertainty about policy as a factor constraining aggregate demand. In addition, participants still saw significant downside risks to the outlook for economic growth. Prominent among these risks were a possible intensification of strains in the euro zone, with potential spillovers to U.S. financial markets and institutions and thus to the broader U.S. economy; a larger-than-expected U.S. fiscal tightening; and the possibility of a further slowdown in global economic growth. A few participants, however, mentioned the possibility that economic growth could be more rapid than currently anticipated, particularly if major sources of uncertainty were resolved favorably or if faster-than-expected advances in the housing sector led to improvements in household balance sheets, increased confidence, and easier credit conditions. Participants' forecasts for economic activity, which in most cases were conditioned on an assumption of additional, near-term monetary policy accommodation, were also associated with an outlook for the unemployment rate to remain close to recent levels through 2012 and then to decline gradually toward levels judged to be consistent with the Committee's mandate.
    
    In the household sector, incoming data on retail sales were somewhat stronger than expected. Participants noted, however, that households were still in the process of deleveraging, confidence was low, and consumers appeared to remain particularly pessimistic about the prospects for the future, raising doubts that the somewhat stronger pace of spending would persist. Although the level of activity in the housing sector remained low, the somewhat faster pace of home sales and construction provided some encouraging signs of improvement. A number of participants also observed that house prices were rising. It was noted that such increases, coupled with historically low mortgage rates, could lead to a stronger upturn in housing activity, although constraints on the capacity for loan origination and still-tight credit terms for some borrowers continued to weigh on mortgage lending.
    
    Business contacts in many parts of the country were reported to be highly uncertain about the outlook for the economy and for fiscal and regulatory policies. Although firms' balance sheets were generally strong, these uncertainties had led them to be particularly cautious and to remain reluctant to hire or expand capacity. Reports on manufacturing activity were mixed, with production related to autos and housing the most notable areas of relative strength. In one District, business surveys pointed to further growth; however, readings on forward-looking indicators of orders around the country were less positive. In addition, business contacts noted that export demand was showing signs of weakness as a result of the slowdown in economic activity in Europe. The energy sector continued to expand. In the agricultural sector, high grain prices and crop insurance payments were supporting farm incomes, helping offset declines in production and reduced profits on livestock. The drought was expected to reduce farm inventories and have a transitory impact on broader measures of economic growth.
    
    Participants generally expected that fiscal policy would continue to be a drag on economic activity over coming quarters. In addition to ongoing weakness in spending at the federal, state, and local government levels, uncertainties about tax and spending policies reportedly were restraining business decisionmaking. Participants also noted that if an agreement was not reached to tackle the expiring tax cuts and scheduled spending reductions, a sharp consolidation of fiscal policy would take place at the beginning of 2013.
    
    The available indicators pointed to continued weakness in overall labor market conditions. Growth in employment had been disappointing, with the average monthly increases in payrolls so far this year below last year's pace and below the pace that would be required to make significant progress in reducing the unemployment rate. The unemployment rate declined around the turn of the year but had not fallen significantly since then. In addition, the labor force participation rate and employment-to-population ratios were at or near post-recession lows.
    
    Meeting participants again discussed the extent of slack in labor markets. A few participants reiterated their view that the persistently high level of unemployment reflected the effect of structural factors, including mismatches across and within sectors between the skills of the unemployed and those demanded in sectors in which jobs were currently available. It was also suggested that there was an ongoing process of polarization in the labor market, with the share of job opportunities in middle-skill occupations continuing to decline while the shares of low and high skill occupations increased. Both of these views would suggest a lower level of potential output and thus reduced scope for combating unemployment with additional monetary policy stimulus. Several participants, while acknowledging some evidence of structural changes in the labor market, stated again that weak aggregate demand was the principal reason for the high unemployment rate. They saw slack in resource utilization as remaining wide, indicating an important role for additional policy accommodation. Several participants noted the risk that continued high levels of unemployment, even if initially cyclical, might ultimately induce adverse structural changes. In particular, they expressed concerns about the risk that the exceptionally high level of long-term unemployment and the depressed level of labor participation could ultimately lead to permanent negative effects on the skills and prospects of those without jobs, thereby reducing the longer-run normal level of employment and potential output.
    
    Sentiment in financial markets improved notably during the intermeeting period. Participants indicated that recent decisions by the ECB helped ease investors' anxiety about the near-term prospects for the euro. However, participants also observed that significant risks related to the euro-area banking and fiscal crisis remained, and that a number of important issues would have to be resolved in order to achieve further progress toward a comprehensive solution to the crisis. Participants noted that indicators of financial stress in the United States were not especially high and overall conditions in U.S. financial markets remained favorable. Longer-term interest rates were low and supportive of economic growth, while equity prices had risen. One participant noted that, while there were few current signs of excessive risk-taking, low interest rates could ultimately lead to financial imbalances that would be challenging to detect before they became serious problems.
    
    The incoming information on inflation over the intermeeting period was largely in line with participants' expectations. Despite recent increases in the prices of some key commodities, consumer price inflation remained subdued. With longer-term inflation expectations stable and the unemployment rate elevated, participants generally anticipated that inflation over the medium run would likely run at or below the 2 percent rate that the Committee judges to be most consistent with its mandate. Most participants saw the risks to the outlook for inflation as roughly balanced. A few participants felt that maintaining a highly accommodative stance of monetary policy over an extended period could unmoor longer-term inflation expectations and, against a backdrop of higher energy and commodity prices, posed upside risks to inflation. Other participants, by contrast, saw inflation risks as tilted to the downside, given their expectations for sizable and persistent resource slack.
    
    Participants again exchanged views on the likely benefits and costs of a new large-scale asset purchase program. Many participants anticipated that such a program would provide support to the economic recovery by putting downward pressure on longer-term interest rates and promoting more accommodative financial conditions. A number of participants also indicated that it could lift consumer and business confidence by emphasizing the Committee's commitment to continued progress toward its dual mandate. In addition, it was noted that additional purchases could reinforce the Committee's forward guidance regarding the federal funds rate. Participants discussed the effectiveness of purchases of Treasury securities relative to purchases of agency MBS in easing financial conditions. Some participants suggested that, all else being equal, MBS purchases could be preferable because they would more directly support the housing sector, which remains weak but has shown some signs of improvement of late. One participant, however, objected that purchases of MBS, when compared to purchases of longer-term Treasury securities, would likely result in higher interest rates for many borrowers in other sectors. A number of participants highlighted the uncertainty about the overall effects of additional purchases on financial markets and the real economy. Some participants thought past purchases were useful because they were conducted during periods of market stress or heightened deflation risk and were less confident of the efficacy of additional purchases under present circumstances. A few expressed skepticism that additional policy accommodation could help spur an economy that they saw as held back by uncertainties and a range of structural issues. In discussing the costs and risks that such a program might entail, several participants reiterated their concern that additional purchases might complicate the Committee's efforts to withdraw monetary policy accommodation when it eventually became appropriate to do so, raising the risk of undesirably high inflation in the future and potentially unmooring inflation expectations. One participant noted that an extended period of accommodation resulting from additional asset purchases could lead to excessive risk-taking on the part of some investors and so undermine financial stability over time. The possible adverse effects of large purchases on market functioning were also noted. However, most participants thought these risks could be managed since the Committee could make adjustments to its purchases, as needed, in response to economic developments or to changes in its assessment of their efficacy and costs.
    
    Participants also discussed issues related to the provision of forward guidance regarding the future path of the federal funds rate. It was noted that clear communication and credibility allow the central bank to help shape the public's expectations about policy, which is crucial to managing monetary policy when the federal funds rate is at its effective lower bound. A number of participants questioned the effectiveness of continuing to use a calendar date to provide forward guidance, noting that a change in the calendar date might be interpreted pessimistically as a downgrade of the Committee's economic outlook rather than as conveying the Committee's determination to support the economic recovery. If the public interpreted the statement pessimistically, consumer and business confidence could fall rather than rise. Many participants indicated a preference for replacing the calendar date with language describing the economic factors that the Committee would consider in deciding to raise its target for the federal funds rate. Participants discussed the benefits of such an approach, including the potential for enhanced effectiveness of policy through greater clarity regarding the Committee's future behavior. That approach could also bolster the stimulus provided by the System's holdings of longer-term securities. It was noted that forward guidance along these lines would allow market expectations regarding the federal funds rate to adjust automatically in response to incoming data on the economy. Many participants thought that more-effective forward guidance could be provided by specifying numerical thresholds for labor market and inflation indicators that would be consistent with maintaining the federal funds rate at exceptionally low levels. However, reaching agreement on specific thresholds could be challenging given the diversity of participants' views, and some were reluctant to specify explicit numerical thresholds out of concern that such thresholds would necessarily be too simple to fully capture the complexities of the economy and the policy process or could be incorrectly interpreted as triggers prompting an automatic policy response. In addition, numerical thresholds could be confused with the Committee's longer-term objectives, and so undermine the Committee's credibility. At the conclusion of the discussion, most participants agreed that the use of numerical thresholds could be useful to provide more clarity about the conditionality of the forward guidance but thought that further work would be needed to address the related communications challenges.
    
    Committee Policy Action
    Committee members saw the information received over the intermeeting period as suggesting that economic activity had continued to expand at a moderate pace in recent months. However, growth in employment had been slow, and almost all members saw the unemployment rate as still elevated relative to levels that they viewed as consistent with the Committee's mandate. Members generally judged that without additional policy accommodation, economic growth might not be strong enough to generate sustained improvement in labor market conditions. Moreover, while the sovereign and banking crisis in Europe had eased some recently, members still saw strains in global financial conditions as posing significant downside risks to the economic outlook. The possibility of a larger-than-expected fiscal tightening in the United States and slower global growth were also seen as downside risks. Inflation had been subdued, even though the prices of some key commodities had increased recently. Members generally continued to anticipate that, with longer-term inflation expectations stable and given the existing slack in resource utilization, inflation over the medium term would run at or below the Committee's longer-run objective of 2 percent.
    
    In their discussion of monetary policy for the period ahead, members generally expressed concerns about the slow pace of improvement in labor market conditions and all members but one agreed that the outlook for economic activity and inflation called for additional monetary accommodation. Members agreed that such accommodation should be provided through both a strengthening of the forward guidance regarding the federal funds rate and purchases of additional agency MBS at a pace of $40 billion per month. Along with the ongoing purchases of $45 billion per month of longer-term Treasury securities under the maturity extension program announced in June, these purchases will increase the Committee's holdings of longer-term securities by about $85 billion each month through the end of the year, and should put downward pressure on longer-term interest rates, support mortgage markets, and help make broader financial conditions more accommodative. Members also agreed to maintain the Committee's existing policy of reinvesting principal payments from its holdings of agency debt and agency MBS into agency MBS. The Committee agreed that it would closely monitor incoming information on economic and financial developments in coming months, and that if the outlook for the labor market did not improve substantially, it would continue its purchases of agency MBS, undertake additional asset purchases, and employ its other policy tools as appropriate until such improvement is achieved in a context of price stability. This flexible approach was seen as allowing the Committee to tailor its policy response over time to incoming information while incorporating conditional features that clarified the Committee's intention to improve labor market conditions, thereby enhancing the effectiveness of the action by helping to bolster business and consumer confidence. While members generally viewed the potential risks associated with these purchases as manageable, the Committee agreed that in determining the size, pace, and composition of its asset purchases, it would, as always, take appropriate account of the likely efficacy and costs of such purchases. With regard to the forward guidance, the Committee agreed on an extension through mid-2015, in conjunction with language in the statement indicating that it expects that a highly accommodative stance of policy will remain appropriate for a considerable time after the economic recovery strengthens. That new language was meant to clarify that the maintenance of a very low federal funds rate over that period did not reflect an expectation that the economy would remain weak, but rather reflected the Committee's intention to support a stronger economic recovery. One member dissented from the policy decision, on the grounds that he opposed additional asset purchases and preferred to omit the calendar date from the forward guidance; in his view, it would be better to use qualitative language to describe the factors that would influence the Committee's decision to increase the target federal funds rate.
    
    At the conclusion of the discussion, the Committee voted to authorize and direct the Federal Reserve Bank of New York, until it was instructed otherwise, to execute transactions in the System Account in accordance with the following domestic policy directive:
    
    
    "The Federal Open Market Committee seeks monetary and financial conditions that will foster price stability and promote sustainable growth in output. To further its long-run objectives, the Committee seeks conditions in reserve markets consistent with federal funds trading in a range from 0 to 1/4 percent. The Committee directs the Desk to continue the maturity extension program it announced in June to purchase Treasury securities with remaining maturities of 6 years to 30 years with a total face value of about $267 billion by the end of December 2012, and to sell or redeem Treasury securities with remaining maturities of approximately 3 years or less with a total face value of about $267 billion. For the duration of this program, the Committee directs the Desk to suspend its policy of rolling over maturing Treasury securities into new issues. The Committee directs the Desk to maintain its existing policy of reinvesting principal payments on all agency debt and agency mortgage-backed securities in the System Open Market Account in agency mortgage-backed securities. The Desk is also directed to begin purchasing agency mortgage-backed securities at a pace of about $40 billion per month. The Committee directs the Desk to engage in dollar roll and coupon swap transactions as necessary to facilitate settlement of the Federal Reserve's agency MBS transactions. The System Open Market Account Manager and the Secretary will keep the Committee informed of ongoing developments regarding the System's balance sheet that could affect the attainment over time of the Committee's objectives of maximum employment and price stability."
    
    
    The vote encompassed approval of the statement below to be released at 12:30 p.m.:
    
    
    "Information received since the Federal Open Market Committee met in August suggests that economic activity has continued to expand at a moderate pace in recent months. Growth in employment has been slow, and the unemployment rate remains elevated. Household spending has continued to advance, but growth in business fixed investment appears to have slowed. The housing sector has shown some further signs of improvement, albeit from a depressed level. Inflation has been subdued, although the prices of some key commodities have increased recently. Longer-term inflation expectations have remained stable.
    
    Consistent with its statutory mandate, the Committee seeks to foster maximum employment and price stability. The Committee is concerned that, without further policy accommodation, economic growth might not be strong enough to generate sustained improvement in labor market conditions. Furthermore, strains in global financial markets continue to pose significant downside risks to the economic outlook. The Committee also anticipates that inflation over the medium term likely would run at or below its 2 percent objective.
    
    To support a stronger economic recovery and to help ensure that inflation, over time, is at the rate most consistent with its dual mandate, the Committee agreed today to increase policy accommodation by purchasing additional agency mortgage-backed securities at a pace of $40 billion per month. The Committee also will continue through the end of the year its program to extend the average maturity of its holdings of securities as announced in June, and it is maintaining its existing policy of reinvesting principal payments from its holdings of agency debt and agency mortgage-backed securities in agency mortgage-backed securities. These actions, which together will increase the Committee's holdings of longer-term securities by about $85 billion each month through the end of the year, should put downward pressure on longer-term interest rates, support mortgage markets, and help to make broader financial conditions more accommodative.
    
    The Committee will closely monitor incoming information on economic and financial developments in coming months. If the outlook for the labor market does not improve substantially, the Committee will continue its purchases of agency mortgage-backed securities, undertake additional asset purchases, and employ its other policy tools as appropriate until such improvement is achieved in a context of price stability. In determining the size, pace, and composition of its asset purchases, the Committee will, as always, take appropriate account of the likely efficacy and costs of such purchases.
    
    To support continued progress toward maximum employment and price stability, the Committee expects that a highly accommodative stance of monetary policy will remain appropriate for a considerable time after the economic recovery strengthens. In particular, the Committee also decided today to keep the target range for the federal funds rate at 0 to 1/4 percent and currently anticipates that exceptionally low levels for the federal funds rate are likely to be warranted at least through mid-2015."
    
    
    Voting for this action: Ben Bernanke, William C. Dudley, Elizabeth Duke, Dennis P. Lockhart, Sandra Pianalto, Jerome H. Powell, Sarah Bloom Raskin, Jeremy C. Stein, Daniel K. Tarullo, John C. Williams, and Janet L. Yellen.
    
    Voting against this action: Jeffrey M. Lacker.
    
    Mr. Lacker dissented because he believed that additional monetary stimulus at this time was unlikely to result in a discernible improvement in economic growth without also causing an unwanted increase in inflation. Moreover, he expressed his opposition to the purchase of more MBS, because he viewed it as inappropriate for the Committee to choose a particular sector of the economy to support; purchases of Treasury securities instead would have avoided this effect. Finally, he preferred to omit the description of the time period over which exceptionally low levels for the federal funds rate were likely to be warranted.
    
    Consensus Forecast Experiment
    In light of the discussion at the previous FOMC meeting, the subcommittee on communications developed a second experimental exercise intended to shed light on the feasibility and desirability of constructing an FOMC consensus forecast. At this meeting, participants discussed possible formulations of the monetary policy assumptions on which to condition an FOMC consensus forecast and alternative approaches for participants to express their endorsement of the consensus forecast. In conclusion, participants agreed to have a broad discussion of the experiences gathered from the two experimental exercises in conjunction with the October FOMC meeting.
    
    It was agreed that the next meeting of the Committee would be held on Tuesday-Wednesday, October 23-24, 2012. The meeting adjourned at 12:10 p.m. on September 13, 2012.
    
    Notation Vote
    By notation vote completed on August 21, 2012, the Committee unanimously approved the minutes of the FOMC meeting held on July 31-August 1, 2012.
    
    _____________________________
    
    William B. English
    Secretary
    
    1. Attended Wednesday's session only. Return to text
    2. Attended Thursday's session only. Return to text
    3. Attended after the discussion on potential effects of a large-scale asset purchase program. Return to text
    4. Attended the discussion on potential effects of a large-scale asset purchase program. Return to text
    

By check multiple examples of these minutes, this project confirm the **Data to be cleaned:**
1. The lines introducing the people's names and position
2. The title of the minute
3. A time record of the minute
4. The voting result and the appendix following the voting result

##### **Example of a statement**


```python
print(read_fomc.iloc[439]['Text'])
```

    The Federal Open Market Committee voted today to raise its target for the federal funds rate by 25 basis points to 5-3/4 percent.  In a related action, the Board of Governors approved a 25 basis point increase in the discount rate to 5-1/4 percent.
      
    The Committee remains concerned that over time increases in demand will continue to exceed the growth in potential supply, even after taking account of the pronounced rise in productivity growth.  Such trends could foster inflationary imbalances that would undermine the economy's record economic expansion.
      
    Against the background of its long-run goals of price stability and sustainable economic growth and of the information currently available, the Committee believes the risks are weighted mainly toward conditions that may generate heightened inflation pressures in the foreseeable future.
    
    	In taking the discount rate action, the Federal Reserve Board approved requests submitted by the Boards of Directors of the Federal Reserve Banks of Boston, New York, Philadelphia, Cleveland, Richmond, Atlanta, Chicago, St. Louis, Kansas City and San Francisco.  The discount rate is the rate charged depository institutions when they borrow short-term adjustment credit from their district Federal Reserve Banks.
    


```python
print(read_fomc.iloc[2]['Text'])
```

    Recent indicators suggest that economic activity has continued to expand at a solid pace. Job gains have slowed, and the unemployment rate has moved up but remains low. Inflation has made further progress toward the Committee's 2 percent objective but remains somewhat elevated.
    
    The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run. The Committee has gained greater confidence that inflation is moving sustainably toward 2 percent, and judges that the risks to achieving its employment and inflation goals are roughly in balance. The economic outlook is uncertain, and the Committee is attentive to the risks to both sides of its dual mandate.
    
    In light of the progress on inflation and the balance of risks, the Committee decided to lower the target range for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 percent. In considering additional adjustments to the target range for the federal funds rate, the Committee will carefully assess incoming data, the evolving outlook, and the balance of risks. The Committee will continue reducing its holdings of Treasury securities and agency debt and agency mortgageâbacked securities. The Committee is strongly committed to supporting maximum employment and returning inflation to its 2 percent objective.
    
    In assessing the appropriate stance of monetary policy, the Committee will continue to monitor the implications of incoming information for the economic outlook. The Committee would be prepared to adjust the stance of monetary policy as appropriate if risks emerge that could impede the attainment of the Committee's goals. The Committee's assessments will take into account a wide range of information, including readings on labor market conditions, inflation pressures and inflation expectations, and financial and international developments.
    
    Voting for the monetary policy action were Jerome H. Powell, Chair; John C. Williams, Vice Chair; Thomas I. Barkin; Michael S. Barr; Raphael W. Bostic; Lisa D. Cook; Mary C. Daly; Beth M. Hammack; Philip N. Jefferson; Adriana D. Kugler; and Christopher J. Waller. Voting against this action was Michelle W. Bowman, who preferred to lower the target range for the federal funds rate by 1/4 percentage point at this meeting.
    
    For media inquiries, please email [email protected] or call 202-452-2955.
    
    Implementation Note issued September 18, 2024
    

By check multiple examples of these policy statemetnts, this project confirm the **Data to be cleaned:** <br>
The lines after "Voting for the monetary policy action " are not the context of the policy statement

### 3.2.3-1 Information Cleaning for minutes

#### **Preparation**
Policy statements and minutes are **processed separately** because they have different structure and text flows


```python
# Split into statements and minutes
statements = read_fomc.loc[read_fomc['Type'] == 'Statement'].copy().reset_index()
minutes = read_fomc.loc[read_fomc['Type'] == 'Minute'].copy().reset_index()
```


```python
minutes.sample(5)
```





  <div id="df-7e551fc1-b49e-4deb-af28-9e568ee38479" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Date</th>
      <th>Release Date</th>
      <th>Type</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>147</th>
      <td>285</td>
      <td>2008-03-18</td>
      <td>2008-04-08</td>
      <td>Minute</td>
      <td>A meeting of the Federal Open Market Committee...</td>
    </tr>
    <tr>
      <th>177</th>
      <td>343</td>
      <td>2005-03-22</td>
      <td>2005-04-12</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
    </tr>
    <tr>
      <th>164</th>
      <td>317</td>
      <td>2006-10-25</td>
      <td>2006-11-15</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
    </tr>
    <tr>
      <th>163</th>
      <td>315</td>
      <td>2006-12-12</td>
      <td>2007-01-03</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
    </tr>
    <tr>
      <th>98</th>
      <td>195</td>
      <td>2013-01-30</td>
      <td>2013-02-20</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7e551fc1-b49e-4deb-af28-9e568ee38479')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7e551fc1-b49e-4deb-af28-9e568ee38479 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7e551fc1-b49e-4deb-af28-9e568ee38479');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-71aa6a6f-f4c7-4d42-9477-5d94b27c80ee">
  <button class="colab-df-quickchart" onclick="quickchart('df-71aa6a6f-f4c7-4d42-9477-5d94b27c80ee')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-71aa6a6f-f4c7-4d42-9477-5d94b27c80ee button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




#### **Procedure 1:** Defining a funtion to clean the text

The following code seeked for help from AI. The prompt is "please design a function that can help me identify and delete the lines starting with names such as XXXX, XXXX". This is a function combining several prompts because I found the formats of names of different linguistic characters vary and I need to use different regular expression to take all those names into account.

Besides these, this part will also delete the following:
1. lines starting with "Voting for this action:"
2. uniform lines of the following:
  - Title: "Minutes of the Federal Open Market Committee"
  - Date
  - A general descrption without discussion detail
  - A line with content "PRESENT:"
3. some statemetns contains another useless line starting with "Members of the Federal Open Market Committee"
4. after all these deletion, useless infomration are still there and are shown before the section ""Developments in Financial Markets and Open Market Operations","Developments in Financial Markets and Open Market Operations " in the statements.


```python
def text_clean (text_content):
# this function will be applied to the 'Text' cell for each minute

  # Step 1: convert the input text into a list of lines
  lines = text_content.split('\n')

  # Step 2: delete the part after the line starting with "Voting for this action:"
  # define the key phrase that activate the deletion
  key_phrase = "Voting for this action:"

  # new list to store the portion of the text that will be kept after data cleaning
  truncated_lines = []
  for line in lines:
    if line.startswith(key_phrase):
      break
      # add the lines into the new list if the reader does not hit the key words
    truncated_lines.append(line)

  # Step 3: departing from the truncated lines, delete all the lines starting with names
  # here, the program uses Regular Expression to identify the names
  name_start_pattern = re.compile(
  r'^[A-Z][a-z]+(?:\s[A-Z]\.)?\s[A-Z][a-z]+.*'
  )
  '''
  [A-Z][a-z]+: Matches the first name (mandatory, starts with a capital letter)
  (?:\s[A-Z]\.)?: Optionally matches a middle initial with a period
  \s[A-Z][a-z]+: Matches the mandatory last name
  '''

  # here, the project recognize that in the early years, the minutes would call the members name with titles such as Mr., Ms., etc
  prefix_pattern = re.compile(r'^(?:Mr\.|Mrs\.|Ms\.|Dr\.|Messrs\.)\s.*')

  # some name include special alphabet
  other_pattern = re.compile(
  r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\s[A-Z][a-zA-Z'’-]+(?:-[A-Z][a-zA-Z'’-]+)?\b.*$"
  )
  other_2name_pattern = re.compile(
    r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\s[A-Z][a-zA-ZóÓ'-]+(?:-[A-Z][a-zA-ZóÓ'-]+)?\b.*"
  )



  # Filter out lines starting with names
  cleaned_lines = [line for line in truncated_lines if line.strip() and not name_start_pattern.match(line.strip())]
  cleaned_lines = [line for line in cleaned_lines if line.strip() and not prefix_pattern.match(line.strip())]
  cleaned_lines = [line for line in cleaned_lines if line.strip() and not other_pattern.match(line.strip())]
  cleaned_lines = [line for line in cleaned_lines if line.strip() and not other_2name_pattern.match(line.strip())]
  '''
  Until this part the text will have the following content:
  1. Title: "Minutes of the Federal Open Market Committee"
  2. Date
  3. A general descrption without discussion detail
  4. A line with content "PRESENT:"
  These four lines are in the top four line within the list
  '''
  # delete the useless part of the text by directly remove the first four lines
  cleaned_lines = cleaned_lines[4:]

  # Herefore, For the early year minutes, the document contains 6 lines of position titles left to be deleted
  # these old minutes often contain "Members of the Federal Open Market Committee" in the first line
  if cleaned_lines and "Members of the Federal Open Market Committee" in cleaned_lines[0]:
    cleaned_lines = cleaned_lines[5:]

  # delete the part of useless information appearing before the section "Developments in Financial Markets and Open Market Operations"
  new_key_phrases = [
  "Developments in Financial Markets and Open Market Operations",
  "Developments in Financial Markets and Open Market Operations "
  ]
  if  new_key_phrases in cleaned_lines:
    start_index = cleaned_lines.index(new_key_phrases)
    cleaned_lines = cleaned_lines[start_index:]
  else:
    cleaned_lines = cleaned_lines

  # Join the cleaned lines back into text
  return cleaned_lines
```


```python
def is_name_line(line):
    """Check if the line starts with a name using spaCy NER."""
    doc = nlp(line)
    for ent in doc.ents:
        # If the first entity is a PERSON and starts at the beginning of the line
        if ent.label_ == "PERSON" and ent.start_char == 0:
            return True
    return False
```


```python
# prompt: generate a function, that can be applied by minutes['Text']. The function will apply function text_clean to the cell first; then, the function will run this: [line for line in lines if not is_name_line(line)] for the list converted from the content of each cell

def full_clean(row):
    cleaned_text = text_clean(row['Text'])
    cleaned_lines = [line for line in cleaned_text if not is_name_line(line)]
    return '\n'.join(cleaned_lines)
```

##### Example of cleaning a cell of "Text" belonged to "Minute"


```python
example_clean = full_clean(read_fomc.iloc[7])
```


```python
example_clean
```




    'Developments in Financial Markets and Open Market Operations\nThe manager turned first to a review of developments in financial markets. Financial conditions eased modestly over the intermeeting period mainly because of higher equity prices. Taking a somewhat longer perspective, the manager noted that financial conditions had changed little since March but eased notably since the fall. The main drivers of that easing were again higher equity prices, which appeared to respond to the reductions in the perceived odds of a recession, and a consensus among market participants that the federal funds rate has reached its peak. Nominal Treasury yields declined moderately across the curve, on net, but continued to be very sensitive to incoming data surprises, especially those pertaining to inflation and the labor market. The net decline in nominal yields over the period was primarily due to lower real yields. Inflation compensation also fell somewhat, especially at shorter horizons. Longer-term inflation expectations remained well anchored.\nThe manager turned next to policy rate expectations. The path of the federal funds rate implied by futures prices shifted a bit lower over the intermeeting period and indicated one and one-half 25 basis point cuts by year-end. This shift appeared to reflect mostly changes in perceived risks rather than base-case expectations because the modal path implied by options was virtually unchanged and remained consistent with, at most, one cut this year. The median of modal paths of the federal funds rate obtained from the Open Market Desk\'s Survey of Primary Dealers and Survey of Market Participantsâ\x80\x94taken before the May employment reportâ\x80\x94was also little changed.\nThe manager then discussed expectations regarding balance sheet policy. Responses to the Desk surveys showed a median expected timing for the end of balance sheet runoff of April 2025, one month later than in the previous surveys, though individual respondents\' views of the exact timing remained dispersed. Respondents\' expectations about the size of the portfolio at the end of runoff had changed little in recent surveys.\nIn international developments, the European Central Bank (ECB) and the Bank of Canada (BOC) initiated rate-cutting cycles this period, as generally expected. Market participants reportedly had not expected easing cycles to begin at the same time across economies but appeared to expect that most advanced-economy central banks will have started easing policy within the next several months.\nThe manager then turned to money markets and Desk operations. Unsecured overnight rates were stable over the intermeeting period. In secured funding markets, repurchase agreement (repo) rates remained steady for most of the period but firmed close to the end of May because of month-end pressures and the effect of large settlements of Treasury coupon securities. Rate firmness around reporting and settlement dates was consistent with historical patterns. Use of the overnight reverse repurchase agreement (ON RRP) facility remained sensitive to market rates and the availability of alternative investments. Usage was little changed over much of the period but dipped late in the period, coincident with the month-end firming in private repo rates. The staff projected ON RRP usage to decline in coming months, as net Treasury bill issuance was expected to turn positive and private repo rates were expected to continue to move higher relative to administered rates amid large issuance of Treasury coupon securities. The staff also projected that reserves will not change much in the near term, with the exception of quarter-end dates, and then will decline about in line with the shrinking of the Federal Reserve\'s portfolio after ON RRP balances are nearly fully drained. The uncertainty surrounding both projections, however, was considerable.\nThe manager also discussed the responses to a Desk survey question about the most likely spread between the effective federal funds rate and the interest rate on reserve balances at different levels of the sum of reserves and ON RRP balances. The responses indicated considerable uncertainty and dispersion of views about when and how the spread would move as the sum declines. The manager observed that indicators based on market prices and activity were likely the best gauges of how quickly reserves are transitioning from abundant to ample. Over the intermeeting period, the federal funds market continued to be insensitive to day-to-day changes in the supply of reserves; various other indicators suggested that reserves remained abundant and that the risk of money market strains in the near term was low.\nBy unanimous vote, the Committee ratified the Desk\'s domestic transactions over the intermeeting period. There were no intervention operations in foreign currencies for the System\'s account during the intermeeting period.\nThe information available at the time of the meeting suggested that U.S. economic activity had expanded at a solid pace so far this year. Labor market conditions remained solid. Job gains continued to be strong, while the unemployment rate had edged up but was still low. Consumer price inflation was running well below where it was a year earlier, but further progress toward the Committee\'s 2 percent inflation objective had been modest in recent months.\nConsumer price inflationâ\x80\x94as measured by the 12-month change in the price index for personal consumption expenditures (PCE)â\x80\x94was about the same in April as at the end of last year, although recent month-over-month readings of PCE prices were lower than earlier this year. Total PCE price inflation was 2.7 percent in April, and core PCE price inflationâ\x80\x94which excludes changes in energy prices and many consumer food pricesâ\x80\x94was 2.8 percent. The consumer price index (CPI) in May showed that the 12-month change measure of total CPI inflation was 3.3 percent and core CPI inflation was 3.4 percent, and recent monthly CPI readings were lower than earlier this year. Al\xadthough some survey-based measures of short-term inflation expectations had moved up, longer-term expectations were little changed and stood at levels consistent with those that prevailed just before the pandemic.\nLabor demand and supply continued to move into better balance. Total nonfarm payroll employment increased at only a somewhat slower average monthly pace over April and May than the strong rate recorded in the first quarter. The recently released fourth-quarter data from the Quarterly Census of Employment and Wages suggested that while the strong reported rate of payroll increases last year may have been overstated, job gains were still solid. In May, the unemployment rate ticked up further to 4.0 percent, while the labor force participation rate and the employment-to-population ratio both moved down a little. The unemployment rates for African Americans and for Hispanics were somewhat higher in May than in the first quarter; both rates were above those for Asians and for Whites. The ratio of job vacancies to unemployment declined further to 1.2 in May, about the same as its pre-pandemic level. Most measures of the increase in nominal wages from a year earlier continued to trend down, including the 12-month change in average hourly earnings for all employees, which was 4.1 percent in May, 0.2 percentage point lower than at the end of last year.\nReal gross domestic product (GDP) rose modestly in the first quarter, held down by significant negative contributions from inventory investment and net exports, which tend to be volatile components. In contrast, private domestic final purchases (PDFP)â\x80\x94which comprises PCE and private fixed investment and which often provides a better signal than GDP of underlying economic momentumâ\x80\x94increased at a solid pace, similar to last year. Recent spending indicators suggested that GDP and PDFP were increasing at solid rates in the second quarter.\nReal exports of goods edged up in April relative to March, following tepid growth in the first quarter. Real imports of goods jumped in April, driven by higher imports of autos and capital goods. Overall, the nominal U.S. international trade deficit widened in April, as imports of goods and services rose more than exports.\nHeadline inflation continued to ease in the advanced foreign economies (AFEs) through May, albeit at a slower pace than last year. While core inflation had slowed significantly, the core nonhousing services component remained elevated in several regions, partly reflecting strong nominal wage growth. Inflation inched up in EMEs, in part because of weather-related increases in food prices in some countries. The Riksbank, the BOC, and the ECB cut their policy rates as market participants expected, amid easing inflation. Communications about future policy decisions varied and were focused on domestic economic conditions.\nOver the intermeeting period, the market-implied path for the federal funds rate beyond the next few months edged down. Options on interest rate futures suggested that market participants were placing higher odds on policy easing by early 2025 than they did just before the April FOMC meeting. Consistent with the slight downward shift in the implied policy path, nominal Treasury yields at all maturities also moved down moderately, driven primarily by declines in real Treasury yields. Inflation compensation also fell some, with larger declines at nearer horizons. Market-based measures of interest rate uncertainty ticked down but remained elevated by historical standards.\nBroad stock price indexes increased substantially, on net, amid a positive investor outlook on corporate profits and economic activity. Yield spreads on investment- and speculative-grade corporate bonds were little changed, remaining at about the lowest decile of their respective historical distributions. The one-month option-implied volatility on the S&P 500 index remained low by historical standards, suggesting that investors perceived only modest near-term risks to the economic outlook.\nChanges in AFE yields were mixed, as spillovers from declines in U.S. yields were partly offset by upside surprises in economic data releases in Europe and by somewhat more restrictive-than-expected communications by the ECB. The dollar depreciated against most AFE currencies as differentials between U.S. and AFE yields narrowed. Nonetheless, the broad dollar index slightly increased as the dollar appreciated sharply against the Mexican peso amid heightened policy uncertainty following Mexico\'s presidential election results. On balance, moves in foreign risky asset prices were mixed and modest, and EME funds saw small inflows.\nConditions in U.S. short-term funding markets remained stable over the intermeeting period. Average usage of the ON RRP facility was little changed, primarily reflecting the portfolio decisions of money market funds amid lower net Treasury bill supply. Banks\' total deposit levels were roughly unchanged over the intermeeting period, as outflows of core deposits were about offset by inflows of large time deposits.\nIn domestic credit markets, borrowing costs remained elevated despite declining modestly over the intermeeting period. Rates on 30-year conforming residential mortgages edged down, on net, over the intermeeting period but remained near recent high levels. Interest rates on new credit card offers were little changed in April at high levels, as were rates on new auto loans. Interest rates on commercial and industrial (C&I) loans and small business loans also remained elevated. Yields on an array of fixed-income securities, including commercial mortgage-backed securities (CMBS), investment- and speculative-grade corporate bonds, and residential mortgage-backed securities, moved lower to still-elevated levels relative to recent history.\nFinancing was readily accessible for public corporations and large and middle-market private corporations through capital markets and nonbank lenders. Credit availability for leveraged loan borrowers remained solid over the intermeeting period, while in private credit markets, loan issuance through direct lending was strong. Bank C&I loan balances picked up in April and May. For small firms, the volume of loan originations ticked down in April, and credit availability remained tight.\nCredit remained largely available to commercial real estate (CRE) borrowers outside of construction and land development loans. CRE loans at banks continued to increase in April and May, driven by growth in multifamily and nonfarm nonresidential loans. Agency and non-agency CMBS issuance rose in April and May, as falling yields extended the recent wave of refinancing.\nConsumer credit remained generally available over the intermeeting period despite some signs of tightening. In the residential mortgage market, access to credit was little changed and continued to depend on borrowers\' credit risk attributes. Although credit card limits continued to rise through March, credit card balances at banks leveled off in April and May. Auto lending at finance companies continued to grow at a moderate pace through April, more than offsetting the decline in auto loan balances at banks and credit unions on net.\nCredit quality continued to be solid for large and midsize firms, home mortgage borrowers, and municipalities but deteriorated further for other sectors in recent months. While delinquency rates on residential mortgages remained near pre-pandemic lows, credit card and auto loan delinquency rates continued to rise in the first quarter, signaling a further deterioration of balance sheets of some households. The credit quality of nonfinancial firms borrowing in the corporate bond and leveraged loan markets remained stable overall. Available indicators suggested that delinquency rates for the private credit market and for bank C&I loans remained comparable to the levels just before the pandemic despite ticking up further in the first quarter. For small business loans, delinquency rates stayed slightly above pre-pandemic levels. In the CRE market, credit quality deteriorated further, as the average CMBS delinquency rate rose in April and May to the highest levels since 2021, driven by the office, hotel, and retail sectors, and the credit quality of CRE borrowers at banks weakened slightly further in the first quarter.\nThe economic forecast prepared by the staff for the June meeting was similar to the projection at the time of the previous meeting. The economy was expected to maintain a high rate of resource utilization over the next few years, with real GDP growth projected to be roughly similar to the staff\'s estimate of potential output growth. The unemployment rate was expected to edge down slightly over the remainder of this year and the next and then to remain roughly flat in 2026.\nTotal and core PCE price inflation were both projected to be lower at the end of this year than they were at the end of last year. The staff\'s inflation projections for this yearâ\x80\x94which included a preliminary reaction to the May CPI dataâ\x80\x94were little changed, on balance, from the inflation forecast at the time of the previous meeting. The inflation forecast was higher, however, than at the time of the March meeting and the March Summary of Economic Projections (SEP) submissions. Inflation was still expected to decline further in 2025 and 2026, as demand and supply in product and labor markets continued to move into better balance; by 2026, total and core PCE price inflation were expected to be close to 2 percent.\nThe staff continued to view the uncertainty around the baseline projection as close to the average over the past 20 years. Risks to the inflation forecast were seen as tilted to the upside, reflecting the possibility that more persistent inflation dynamics or supply-side disruptions could unexpectedly materialize. The risks around the forecast for economic activity were seen as skewed to the downside on the grounds that more-persistent inflation could result in tighter financial conditions than in the staff\'s baseline projection; in addition, deteriorating household financial positions, especially for lower-income households, might prove to have a larger negative effect on economic activity than the staff anticipated.\nParticipants\' Views on Current Conditions and the Economic Outlook\nIn conjunction with this FOMC meeting, participants submitted their projections of the most likely outcomes for real GDP growth, the unemployment rate, and inflation for each year from 2024 through 2026 and over the longer run. These projections were based on their individual assessments of appropriate monetary policy, including their projections of the federal funds rate. The longer-run projections represented each participant\'s assessment of the rate to which each variable would tend to converge under appropriate monetary policy and in the absence of further shocks to the economy. The SEP was released to the public after the meeting.\nIn their discussion of inflation developments, participants noted that after a significant decline in inflation during the second half of 2023, the early part of this year had seen a lack of further progress toward the Committee\'s 2 percent objective. Participants judged that although inflation remained elevated, there had been modest further progress toward the 2 percent goal in recent months. Participants observed that some of this progress was evident in the smaller monthly change in the core PCE price index and a lower trimmed mean inflation rate for April, with the May CPI reading providing additional evidence. Recent data had also indicated improvements across a range of price categories, including market-based services. Some participants commented that sustained achievement of the 2 percent inflation objective would be aided by lower overall services price inflation, and some noted that shelter price inflation had so far been slow to come down. A few participants also highlighted the strong increases recorded this year in core import prices. Nevertheless, participants suggested that a number of developments in the product and labor markets supported their judgment that price pressures were diminishing. In particular, a few participants emphasized that nominal wage growth, though still above rates consistent with price stability, had declined, notably in labor-intensive sectors. A few participants also noted reports that various retailers had cut prices and offered discounts. Participants further indicated that business contacts reported that their pricing power had declined. Participants suggested that evidence of firms\' reduced pricing power reflected increased customer resistance to price increases, slower growth in economic activity, and a reassessment by businesses of prospective economic conditions.\nWith regard to the outlook for inflation, participants emphasized that they were strongly committed to their 2 percent objective and that they remained concerned that elevated inflation continued to harm the purchasing power of households, especially those least able to meet the higher costs of essentials like food, housing, and transportation. Participants highlighted a variety of factors that were likely to help contribute to continued disinflation in the period ahead. The factors included continued easing of demandâ\x80\x93supply pressures in product and labor markets, lagged effects on wages and prices of past monetary policy tightening, the delayed response of measured shelter prices to rental market developments, or the prospect of additional supply-side improvements. The latter prospect included the possibility of a boost to productivity associated with businesses\' deployment of artificial intelligenceâ\x80\x93related technology. Participants observed that longer-term inflation expectations had remained well anchored and viewed this anchoring as underpinning the disinflation process. Participants affirmed that additional favorable data were required to give them greater confidence that inflation was moving sustainably toward 2 percent.\nParticipants remarked that demand and supply in the labor market had continued to come into better balance. Participants observed that many labor market indicators pointed to a reduced degree of tightness in labor market conditions. These included a declining job openings rate, a lower quits rate, increases in part-time employment for economic reasons, a lower hiring rate, a further step-down in the ratio of job vacancies to unemployed workers, and a gradual uptick in the unemployment rate. In addition, a few participants indicated that business contacts were reporting less difficulty in hiring and retaining workers, although contacts in several Districts continued to report tight labor market conditions in certain sectors, such as health care, construction, or specialty manufacturing. Many participants noted that labor supply had been boosted by increased labor force participation rates as well as by immigration. A few participants noted that it was unlikely that immigration would continue at the pace seen in recent years. However, several participants judged that, with recent immigrants gradually becoming part of the workforce, past immigration likely would continue to add to labor supply. A few participants observed that increases in labor force participation would likely now be limited and so would not be a major source of additional labor supply. In considering recent payrolls data, some participants observed that, although increases in payrolls had continued to be strong, the monthly increase in employment consistent with labor market equilibrium might now be higher than in the past because of immigration. Several participants also suggested that the establishment survey may have overstated actual job gains. Several participants remarked that a variety of indicators, including wage gains for job switchers, suggested that nominal wage growth was slowing, consistent with easing labor market pressures. A number of participants noted that, although the labor market remained strong, the ratio of vacancies to unemployment had returned to pre-pandemic levels and there was some risk that further cooling in labor market conditions could be associated with an increased pace of layoffs. Some participants observed that, with the risks to the Committee\'s dual-mandate goals having now come into better balance, labor market conditions would need careful monitoring. Participants generally observed that continued labor market strength could be consistent with the Committee achieving both its employment and inflation goals, though they noted that some further gradual cooling in the labor market may be required.\nParticipants noted that recent indicators suggested that economic activity had continued to expand at a solid pace. Participants expected that real GDP growth this year would be below the strong pace recorded in 2023, and they remarked that recent data on economic activity were largely consistent with the anticipated slowing. Participants observed that a lower rate of output growth this year could aid the disinflation process while also being consistent with a strong labor market. Participants generally viewed the Committee\'s restrictive monetary policy stance as having a restraining effect on growth in consumption and investment spending and as contributing to a gradual slowing in the pace of economic activity. A couple of participants particularly stressed that the Committee\'s past policy tightening had contributed to higher rates for home mortgage loans and other longer-term borrowing, which were moderating spending and production, including households\' discretionary purchases and residential construction activity. A few participants remarked that spending by some higher-income households was likely being bolstered by increasing asset prices. Many participants observed that, in contrast, lower- and moderate-income households were encountering increasing strains as they attempted to meet higher living costs after having largely run down savings accumulated during the pandemic. These participants noted that such strains, which were evident in rising credit card utilization and delinquency rates as well as motor vehicle loan delinquencies, were a significant concern.\nParticipants continued to assess that the risks to achieving their employment and inflation goals had moved toward better balance over the past year. Participants cited a number of downside risks to economic activity, including those associated with a sharper-than-anticipated slowing in aggregate demand alongside a marked deterioration in labor market conditions, or with strains on lower- and moderate-income households\' budgets leading to an abrupt curtailment of consumer spending. A few participants pointed to downside risks to economic activity associated with the fragility of some parts of the CRE sector or the vulnerable balance sheet positions of some banks. Some participants highlighted reasons why inflation could remain above 2 percent for longer than expected. These participants pointed to risks that inflation could stay elevated as a result of worsening geopolitical developments, heightened trade tensions, more persistent shelter price inflation, financial conditions that might be or could become insufficiently restrictive, or U.S. fiscal policy becoming more expansionary than expected; the latter two scenarios were also seen as implying upside risks to economic activity. Several participants also cited the risk of an unanchoring of longer-term inflation expectations.\nIn their consideration of monetary policy at this meeting, participants observed that incoming data indicated continued solid growth in economic activity and a strong labor market while also pointing to modest further progress toward the Committee\'s 2 percent inflation objective in recent months. Participants remained highly attentive to inflation risks. All participants judged that, in light of current economic conditions and their implications for the outlook for employment and inflation, as well as the balance of risks, it was appropriate to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent. Participants furthermore judged that it was appropriate to continue the process of reducing the Federal Reserve\'s securities holdings.\nIn discussing the outlook for monetary policy, participants noted that progress in reducing inflation had been slower this year than they had expected last December. They emphasized that they did not expect that it would be appropriate to lower the target range for the federal funds rate until additional information had emerged to give them greater confidence that inflation was moving sustainably toward the Committee\'s 2 percent objective. In discussing their individual outlooks for the target range for the federal funds rate, participants emphasized the importance of conditioning future policy decisions on incoming data, the evolving economic outlook, and the balance of risks. Several participants noted that financial market reactions to data and feedback received from contacts suggested that the Committee\'s policy approach was generally well understood. Some participants suggested that further clarity about the FOMC\'s reaction function might be provided by communications that emphasized the Committee\'s data-dependent approach, with monetary policy decisions being conditional on the evolution of the economy rather than being on a preset path. A couple of participants remarked that providing more information about the Committee\'s views on the economic outlook and the risks around the outlook would improve the public\'s understanding of the Committee\'s decisions.\nIn discussing risk-management considerations that could bear on the outlook for monetary policy, participants assessed that, with labor market tightness having eased and inflation having declined over the past year, the risks to achieving the Committee\'s employment and inflation goals had moved toward better balance, leaving monetary policy well positioned to deal with the risks and uncertainties faced in pursuing both sides of the Committee\'s dual mandate. The vast majority of participants assessed that growth in economic activity appeared to be gradually cooling, and most participants remarked that they viewed the current policy stance as restrictive. Some participants noted that there was uncertainty about the degree of restrictiveness of current policy. Some remarked that the continued strength of the economy, as well as other factors, could mean that the longer-run equilibrium interest rate was higher than previously assessed, in which case both the stance of monetary policy and overall financial conditions may be less restrictive than they might appear. A couple of participants noted that the longer-run equilibrium interest rate was a better guide for determining where the federal funds rate may need to move over the longer run than for assessing the restrictiveness of current policy. Participants noted the uncertainty associated with the economic outlook and with how long it would be appropriate to maintain a restrictive policy stance. Some participants emphasized the need for patience in allowing the Committee\'s restrictive policy stance to restrain aggregate demand and further moderate inflation pressures. Several participants observed that, were inflation to persist at an elevated level or to increase further, the target range for the federal funds rate might need to be raised. A number of participants remarked that monetary policy should stand ready to respond to unexpected economic weakness. Several participants specifically emphasized that with the labor market normalizing, a further weakening of demand may now generate a larger unemployment response than in the recent past when lower demand for labor was felt relatively more through fewer job openings.\nIn their discussions of monetary policy for this meeting, members agreed that economic activity continued to expand at a solid pace. Job gains remained strong, and the unemployment rate remained low. Inflation eased over the past year but remained elevated. Members concurred that, in recent months, there was modest further progress toward the Committee\'s 2 percent inflation objective and agreed to acknowledge this development in the postmeeting statement. Members judged that the risks to achieving the Committee\'s employment and inflation goals had moved toward better balance over the past year. Members viewed the economic outlook as uncertain and agreed that they remained highly attentive to inflation risks.\nIn support of the Committee\'s goals to achieve maximum employment and inflation at the rate of 2 percent over the longer run, members agreed to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent. Members concurred that, in considering any adjustments to the target range for the federal funds rate, they would carefully assess incoming data, the evolving outlook, and the balance of risks. Members agreed that they did not expect that it would be appropriate to reduce the target range until they have gained greater confidence that inflation is moving sustainably toward 2 percent. In addition, members agreed to continue to reduce the Federal Reserve\'s holdings of Treasury securities and agency debt and agency mortgage-backed securities. All members affirmed their strong commitment to returning inflation to the Committee\'s 2 percent objective.\nMembers agreed that, in assessing the appropriate stance of monetary policy, they would continue to monitor the implications of incoming information for the economic outlook. They would be prepared to adjust the stance of monetary policy as appropriate if risks emerged that could impede the attainment of the Committee\'s goals. Members also agreed that their assessments would take into account a wide range of information, including readings on labor market conditions, inflation pressures and inflation expectations, and financial and international developments.\nAt the conclusion of the discussion, the Committee voted to direct the Federal Reserve Bank of New York, until instructed otherwise, to execute transactions in the SOMA in accordance with the following domestic policy directive, for release at 2:00 p.m.:\n"Effective June 13, 2024, the Federal Open Market Committee directs the Desk to:\n\tUndertake open market operations as necessary to maintain the federal funds rate in a target range of 5-1/4 to 5-1/2 percent.\n\tConduct standing overnight repurchase agreement operations with a minimum bid rate of 5.5 percent and with an aggregate operation limit of $500 billion.\n\tConduct standing overnight reverse repurchase agreement operations at an offering rate of 5.3 percent and with a per-counterparty limit of $160 billion per day.\n\tRoll over at auction the amount of principal payments from the Federal Reserve\'s holdings of Treasury securities maturing in each calendar month that exceeds a cap of $25 billion per month. Redeem Treasury coupon securities up to this monthly cap and Treasury bills to the extent that coupon principal payments are less than the monthly cap.\n\tReinvest the amount of principal payments from the Federal Reserve\'s holdings of agency debt and agency mortgage-backed securities (MBS) received in each calendar month that exceeds a cap of $35 billion per month into Treasury securities to roughly match the maturity composition of Treasury securities outstanding.\n\tAllow modest deviations from stated amounts for reinvestments, if needed for operational reasons.\n\tEngage in dollar roll and coupon swap transactions as necessary to facilitate settlement of the Federal Reserve\'s agency MBS transactions."\nThe vote also encompassed approval of the statement below for release at 2:00 p.m.:\n"Recent indicators suggest that economic activity has continued to expand at a solid pace. Job gains have remained strong, and the unemployment rate has remained low. Inflation has eased over the past year but remains elevated. In recent months, there has been modest further progress toward the Committee\'s 2 percent inflation objective.\nIn support of its goals, the Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent. In considering any adjustments to the target range for the federal funds rate, the Committee will carefully assess incoming data, the evolving outlook, and the balance of risks. The Committee does not expect it will be appropriate to reduce the target range until it has gained greater confidence that inflation is moving sustainably toward 2 percent. In addition, the Committee will continue reducing its holdings of Treasury securities and agency debt and agency mortgageâ\x80\x91backed securities. The Committee is strongly committed to returning inflation to its 2 percent objective.\nIn assessing the appropriate stance of monetary policy, the Committee will continue to monitor the implications of incoming information for the economic outlook. The Committee would be prepared to adjust the stance of monetary policy as appropriate if risks emerge that could impede the attainment of the Committee\'s goals. The Committee\'s assessments will take into account a wide range of information, including readings on labor market conditions, inflation pressures and inflation expectations, and financial and international developments."'



#### **Procedure 2:** Applying the funtion to the cells


```python
# prompt: I want to apply all the function full_clean to the df minutes, and use a new column name "clean text" to store the cleaned text

minutes['clean text'] = minutes.apply(full_clean, axis=1)
```


```python
minutes.sample(2)
```





  <div id="df-03a94a98-5bc1-4a22-996e-3542276cc862" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Date</th>
      <th>Release Date</th>
      <th>Type</th>
      <th>Text</th>
      <th>clean text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>80</td>
      <td>2020-01-29</td>
      <td>2020-02-19</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
      <td>Òscar Jordà, Senior Policy Advisor, Federal Re...</td>
    </tr>
    <tr>
      <th>225</th>
      <td>432</td>
      <td>2000-06-28</td>
      <td>2000-08-24</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
      <td>By unanimous vote, David J. Stockton was elect...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-03a94a98-5bc1-4a22-996e-3542276cc862')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-03a94a98-5bc1-4a22-996e-3542276cc862 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-03a94a98-5bc1-4a22-996e-3542276cc862');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5bfb0a59-3a71-403e-8acb-da0ce0288bd8">
  <button class="colab-df-quickchart" onclick="quickchart('df-5bfb0a59-3a71-403e-8acb-da0ce0288bd8')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5bfb0a59-3a71-403e-8acb-da0ce0288bd8 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




### 3.2.3 - 2 Information Cleaning for policy statements


```python
statements.sample(2)
```





  <div id="df-9f12333a-eeda-4416-8a92-ce7fc5505c2c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Date</th>
      <th>Release Date</th>
      <th>Type</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>172</th>
      <td>356</td>
      <td>2004-06-30</td>
      <td>2004-06-30</td>
      <td>Statement</td>
      <td>The Federal Open Market Committee decided toda...</td>
    </tr>
    <tr>
      <th>157</th>
      <td>326</td>
      <td>2006-05-10</td>
      <td>2006-05-10</td>
      <td>Statement</td>
      <td>The Federal Open Market Committee decided toda...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9f12333a-eeda-4416-8a92-ce7fc5505c2c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9f12333a-eeda-4416-8a92-ce7fc5505c2c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9f12333a-eeda-4416-8a92-ce7fc5505c2c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-14fc3d43-0883-4dac-9654-16192258110e">
  <button class="colab-df-quickchart" onclick="quickchart('df-14fc3d43-0883-4dac-9654-16192258110e')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-14fc3d43-0883-4dac-9654-16192258110e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




#### **Procedure 1:** Defining a funtion to clean the text


```python
def clean_state (context):
  text_content = context['Text']
  lines = text_content.split('\n')

  # delete the part after the line starting with "Voting for"
  key_phrase = "Voting for"

  truncated_lines = []
  for line in lines:
      if line.startswith(key_phrase):
          break
      truncated_lines.append(line)

  return '\n'.join(truncated_lines)
```


```python
example_clean = clean_state(read_fomc.iloc[2])
```


```python
example_clean
```




    "Recent indicators suggest that economic activity has continued to expand at a solid pace. Job gains have slowed, and the unemployment rate has moved up but remains low. Inflation has made further progress toward the Committee's 2 percent objective but remains somewhat elevated.\n\nThe Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run. The Committee has gained greater confidence that inflation is moving sustainably toward 2 percent, and judges that the risks to achieving its employment and inflation goals are roughly in balance. The economic outlook is uncertain, and the Committee is attentive to the risks to both sides of its dual mandate.\n\nIn light of the progress on inflation and the balance of risks, the Committee decided to lower the target range for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 percent. In considering additional adjustments to the target range for the federal funds rate, the Committee will carefully assess incoming data, the evolving outlook, and the balance of risks. The Committee will continue reducing its holdings of Treasury securities and agency debt and agency mortgageâ\x80\x91backed securities. The Committee is strongly committed to supporting maximum employment and returning inflation to its 2 percent objective.\n\nIn assessing the appropriate stance of monetary policy, the Committee will continue to monitor the implications of incoming information for the economic outlook. The Committee would be prepared to adjust the stance of monetary policy as appropriate if risks emerge that could impede the attainment of the Committee's goals. The Committee's assessments will take into account a wide range of information, including readings on labor market conditions, inflation pressures and inflation expectations, and financial and international developments.\n"



#### **Procedure 2:** Applying the funtion to the cells


```python
statements['clean text'] = statements.apply(clean_state, axis=1)
```


```python
statements.iloc[10]['clean text']
```




    "Recent indicators suggest that economic activity has been expanding at a moderate pace. Job gains have been robust in recent months, and the unemployment rate has remained low. Inflation remains elevated.\n\nThe U.S. banking system is sound and resilient. Tighter credit conditions for households and businesses are likely to weigh on economic activity, hiring, and inflation. The extent of these effects remains uncertain. The Committee remains highly attentive to inflation risks.\n\nThe Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run. In support of these goals, the Committee decided to raise the target range for the federal funds rate to 5-1/4 to 5-1/2 percent. The Committee will continue to assess additional information and its implications for monetary policy. In determining the extent of additional policy firming that may be appropriate to return inflation to 2 percent over time, the Committee will take into account the cumulative tightening of monetary policy, the lags with which monetary policy affects economic activity and inflation, and economic and financial developments. In addition, the Committee will continue reducing its holdings of Treasury securities and agency debt and agency mortgage-backed securities, as described in its previously announced plans. The Committee is strongly committed to returning inflation to its 2 percent objective.\n\nIn assessing the appropriate stance of monetary policy, the Committee will continue to monitor the implications of incoming information for the economic outlook. The Committee would be prepared to adjust the stance of monetary policy as appropriate if risks emerge that could impede the attainment of the Committee's goals. The Committee's assessments will take into account a wide range of information, including readings on labor market conditions, inflation pressures and inflation expectations, and financial and international developments.\n"




```python
statements.sample(3)
```





  <div id="df-9d5c11c4-f658-446a-a3a5-e34d0cf7b3b3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Date</th>
      <th>Release Date</th>
      <th>Type</th>
      <th>Text</th>
      <th>clean text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>12</td>
      <td>2024-01-31</td>
      <td>2024-01-31</td>
      <td>Statement</td>
      <td>Recent indicators suggest that economic activi...</td>
      <td>Recent indicators suggest that economic activi...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>38</td>
      <td>2022-06-15</td>
      <td>2022-06-15</td>
      <td>Statement</td>
      <td>Overall economic activity appears to have pick...</td>
      <td>Overall economic activity appears to have pick...</td>
    </tr>
    <tr>
      <th>206</th>
      <td>431</td>
      <td>2000-08-22</td>
      <td>2000-08-22</td>
      <td>Statement</td>
      <td>The Federal Open Market Committee at its meeti...</td>
      <td>The Federal Open Market Committee at its meeti...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9d5c11c4-f658-446a-a3a5-e34d0cf7b3b3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9d5c11c4-f658-446a-a3a5-e34d0cf7b3b3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9d5c11c4-f658-446a-a3a5-e34d0cf7b3b3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-42048269-d809-44cc-9398-dac566e63a88">
  <button class="colab-df-quickchart" onclick="quickchart('df-42048269-d809-44cc-9398-dac566e63a88')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-42048269-d809-44cc-9398-dac566e63a88 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




### 3.2.4 Removing stop words from the clean text
Now, the dataframes storing statements and minutes contains the clean texts that do not have unnecessary information (such as the name and title of conference participants, dates, and voting results). The next step is to remove the stop words from the 'clean text'.


```python
# defining the function used to remove the stop words
def no_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)
```


```python
len(statements.iloc[2]['clean text'])
```




    1957




```python
# apply the function to the text
minutes['clean no_stop text'] = minutes['clean text'].apply(no_stopwords)
statements['clean no_stop text'] = statements['clean text'].apply(no_stopwords)
```


```python
len(statements.iloc[2]['clean no_stop text'])
```




    1471




```python
'''
minutes = pd.read_csv('/content/drive/MyDrive/text_analysis/minutes_clean.csv')
statements = pd.read_csv('/content/drive/MyDrive/text_analysis/statements_clean.csv')
'''
```




    "\nminutes = pd.read_csv('/content/drive/MyDrive/text_analysis/minutes_clean.csv')\nstatements = pd.read_csv('/content/drive/MyDrive/text_analysis/statements_clean.csv')\n"




```python
minutes.head()
```





  <div id="df-c0a0fd58-b433-4bbe-b097-832b92a63041" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Date</th>
      <th>Release Date</th>
      <th>Type</th>
      <th>Text</th>
      <th>clean text</th>
      <th>clean no_stop text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2024-11-07</td>
      <td>2024-11-26</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
      <td>The manager turned first to a review of develo...</td>
      <td>manager turned first review developments finan...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>2024-09-18</td>
      <td>2024-10-09</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
      <td>The manager turned first to a review of develo...</td>
      <td>manager turned first review developments finan...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>2024-07-31</td>
      <td>2024-08-21</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
      <td>The manager turned first to a review of develo...</td>
      <td>manager turned first review developments finan...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>2024-06-12</td>
      <td>2024-07-03</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
      <td>Developments in Financial Markets and Open Mar...</td>
      <td>Developments Financial Markets Open Market Ope...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>2024-05-01</td>
      <td>2024-05-22</td>
      <td>Minute</td>
      <td>Minutes of the Federal Open Market Committee\n...</td>
      <td>Developments in Financial Markets and Open Mar...</td>
      <td>Developments Financial Markets Open Market Ope...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c0a0fd58-b433-4bbe-b097-832b92a63041')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c0a0fd58-b433-4bbe-b097-832b92a63041 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c0a0fd58-b433-4bbe-b097-832b92a63041');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-1f645aad-efc7-4564-8f46-773c927ad637">
  <button class="colab-df-quickchart" onclick="quickchart('df-1f645aad-efc7-4564-8f46-773c927ad637')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-1f645aad-efc7-4564-8f46-773c927ad637 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




### 3.2.5 Keep words only


```python
def all_vocab(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower().isalpha()]
    return " ".join(filtered_words)

# ... (rest of your existing code)

minutes['clean no_stop text'] = minutes['clean no_stop text'].apply(all_vocab)
statements['clean no_stop text'] = statements['clean no_stop text'].apply(all_vocab)
```

### 3.2.5 Summary of Data Cleaning

Now, the eventual cleaned data is stored in the series named "clean no_stop text' in each dataframe of "statements" and "minutes". The data ran through:
1. information cleaning
2. stop words removals.

However, the data does not run through Lemmatization for two reasons:
1. policies are time sensitive. Therefore, the tenses of the words matter
2. economic data is time-sensitive. Thus, the tenses of the words matter

## 3.3 Data Analysis

### 3.3.1 Word Frequency Test

**Note on word frequency test**:<br>
This part, I intend to use the keywords frequency to see if any model can be used to predict the rate change or relate to rate change. <br>

The keywords used to do the frequency test are directly derived from the dual mandates of the Fed:
1. unemployment
2. inflation



```python
# record the key words
keywords = ['unemployment', 'inflation']
```

#### 3.3.1.1 Crete dataframes that will record the freuency of keywords in each document


```python
def create_word_frequency_dataframe(df, keywords):

    new_df = pd.DataFrame(columns=['Date'] + keywords)
    new_df['Date'] = df['Date']

    for keyword in keywords:
        new_df[keyword] = df['clean no_stop text'].apply(lambda x: x.lower().count(keyword))

    return new_df

statements_fre = create_word_frequency_dataframe(statements, keywords)
minutes_fre = create_word_frequency_dataframe(minutes, keywords)

print(statements_fre.head())
print(minutes_fre.head())
```

            Date  unemployment  inflation
    0 2024-11-07             1          6
    1 2024-09-18             1          8
    2 2024-07-31             1          8
    3 2024-06-12             1          9
    4 2024-05-01             1          9
            Date  unemployment  inflation
    0 2024-11-07             8         46
    1 2024-09-18            18         60
    2 2024-07-31            13         50
    3 2024-06-12            11         63
    4 2024-05-01             8         58
    

#### 3.3.1.2 merge the policy rate with keywords frequency


```python
# Merge interest rates with statements
statements_fre['Date'] = pd.to_datetime(statements_fre['Date'])
merged_statements = pd.merge(statements_fre, final_rates, left_on='Date', right_on='Date', how='left')

# Merge interest rates with minutes
minutes_fre['Date'] = pd.to_datetime(minutes_fre['Date'])
merged_minutes = pd.merge(minutes_fre, final_rates, left_on='Date', right_on='Date', how='left')

# Print the merged dataframes (optional)
print(merged_statements.head())
print(merged_minutes.head())
```

            Date  unemployment  inflation  Policy Rate
    0 2024-11-07             1          6         4.75
    1 2024-09-18             1          8         5.25
    2 2024-07-31             1          8         5.25
    3 2024-06-12             1          9         5.25
    4 2024-05-01             1          9         5.25
            Date  unemployment  inflation  Policy Rate
    0 2024-11-07             8         46         4.75
    1 2024-09-18            18         60         5.25
    2 2024-07-31            13         50         5.25
    3 2024-06-12            11         63         5.25
    4 2024-05-01             8         58         5.25
    

#### 3.3.1.3 Data visualization from statements

This part utilize the code from another class Computing in Context


```python
# Create figure with secondary y-axis
fig1 = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig1.add_trace(
    go.Scatter(
        x=merged_statements["Date"],
        y=merged_statements["inflation"],
        name="Frequency of keyword 'inflation'",
    ),
    secondary_y=False,
)

fig1.add_trace(
    go.Scatter(
        x=merged_statements["Date"],
        y=merged_statements["Policy Rate"],
        name="Policy Rate",
    ),
    secondary_y=True,
)

# Add figure title
fig1.update_layout(title_text=f"Inflation Appearing Frequency vs. Policy Rate (statement information)")

# Set x-axis title
fig1.update_xaxes(title_text="Date")

# Set y-axes titles
fig1.update_yaxes(title_text="Frequency of keyword 'inflation", secondary_y=False)
fig1.update_yaxes(title_text="Policy Rate", secondary_y=True)

fig1.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="da7b23c0-08e6-4280-a612-b1aa77cb44c2" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("da7b23c0-08e6-4280-a612-b1aa77cb44c2")) {                    Plotly.newPlot(                        "da7b23c0-08e6-4280-a612-b1aa77cb44c2",                        [{"name":"Frequency of keyword 'inflation'","x":["2024-11-07T00:00:00","2024-09-18T00:00:00","2024-07-31T00:00:00","2024-06-12T00:00:00","2024-05-01T00:00:00","2024-03-20T00:00:00","2024-01-31T00:00:00","2023-12-13T00:00:00","2023-11-01T00:00:00","2023-09-20T00:00:00","2023-07-26T00:00:00","2023-06-14T00:00:00","2023-05-03T00:00:00","2023-03-22T00:00:00","2023-02-01T00:00:00","2022-12-14T00:00:00","2022-11-02T00:00:00","2022-09-21T00:00:00","2022-07-27T00:00:00","2022-06-15T00:00:00","2022-05-04T00:00:00","2022-03-16T00:00:00","2022-01-26T00:00:00","2021-12-15T00:00:00","2021-11-03T00:00:00","2021-09-22T00:00:00","2021-07-28T00:00:00","2021-06-16T00:00:00","2021-04-28T00:00:00","2021-03-17T00:00:00","2021-01-27T00:00:00","2020-12-16T00:00:00","2020-11-05T00:00:00","2020-09-16T00:00:00","2020-07-29T00:00:00","2020-06-10T00:00:00","2020-04-29T00:00:00","2020-03-23T00:00:00","2020-03-15T00:00:00","2020-03-03T00:00:00","2020-01-29T00:00:00","2019-12-11T00:00:00","2019-10-30T00:00:00","2019-10-04T00:00:00","2019-09-18T00:00:00","2019-07-31T00:00:00","2019-06-19T00:00:00","2019-05-01T00:00:00","2019-03-20T00:00:00","2019-01-30T00:00:00","2018-12-19T00:00:00","2018-11-08T00:00:00","2018-09-26T00:00:00","2018-08-01T00:00:00","2018-06-13T00:00:00","2018-05-02T00:00:00","2018-03-21T00:00:00","2018-01-31T00:00:00","2017-12-13T00:00:00","2017-11-01T00:00:00","2017-09-20T00:00:00","2017-07-26T00:00:00","2017-06-14T00:00:00","2017-05-03T00:00:00","2017-03-15T00:00:00","2017-02-01T00:00:00","2016-12-14T00:00:00","2016-11-02T00:00:00","2016-09-21T00:00:00","2016-07-27T00:00:00","2016-06-15T00:00:00","2016-04-27T00:00:00","2016-03-16T00:00:00","2016-01-27T00:00:00","2015-12-16T00:00:00","2015-10-28T00:00:00","2015-09-17T00:00:00","2015-07-29T00:00:00","2015-06-17T00:00:00","2015-04-29T00:00:00","2015-03-18T00:00:00","2015-01-28T00:00:00","2014-12-17T00:00:00","2014-10-29T00:00:00","2014-09-17T00:00:00","2014-07-30T00:00:00","2014-06-18T00:00:00","2014-04-30T00:00:00","2014-03-19T00:00:00","2014-01-29T00:00:00","2013-12-18T00:00:00","2013-10-30T00:00:00","2013-09-18T00:00:00","2013-07-31T00:00:00","2013-06-19T00:00:00","2013-05-01T00:00:00","2013-03-20T00:00:00","2013-01-30T00:00:00","2012-12-12T00:00:00","2012-10-24T00:00:00","2012-09-13T00:00:00","2012-08-01T00:00:00","2012-06-20T00:00:00","2012-04-25T00:00:00","2012-03-13T00:00:00","2012-01-25T00:00:00","2011-12-13T00:00:00","2011-11-02T00:00:00","2011-09-21T00:00:00","2011-08-09T00:00:00","2011-06-22T00:00:00","2011-04-27T00:00:00","2011-03-15T00:00:00","2011-01-26T00:00:00","2010-12-14T00:00:00","2010-11-03T00:00:00","2010-09-21T00:00:00","2010-08-10T00:00:00","2010-06-23T00:00:00","2010-05-09T00:00:00","2010-04-28T00:00:00","2010-03-16T00:00:00","2010-01-27T00:00:00","2009-12-15T00:00:00","2009-11-04T00:00:00","2009-09-22T00:00:00","2009-08-11T00:00:00","2009-06-24T00:00:00","2009-04-29T00:00:00","2009-03-17T00:00:00","2009-01-28T00:00:00","2008-12-16T00:00:00","2008-10-29T00:00:00","2008-10-07T00:00:00","2008-09-16T00:00:00","2008-08-08T00:00:00","2008-06-25T00:00:00","2008-04-30T00:00:00","2008-03-18T00:00:00","2008-03-10T00:00:00","2008-01-30T00:00:00","2008-01-21T00:00:00","2007-12-11T00:00:00","2007-10-31T00:00:00","2007-09-18T00:00:00","2007-08-16T00:00:00","2007-08-10T00:00:00","2007-08-07T00:00:00","2007-06-28T00:00:00","2007-05-09T00:00:00","2007-03-21T00:00:00","2007-01-31T00:00:00","2006-12-12T00:00:00","2006-10-25T00:00:00","2006-09-20T00:00:00","2006-08-08T00:00:00","2006-06-29T00:00:00","2006-05-10T00:00:00","2006-03-28T00:00:00","2006-01-31T00:00:00","2005-12-13T00:00:00","2005-11-01T00:00:00","2005-09-20T00:00:00","2005-08-09T00:00:00","2005-06-30T00:00:00","2005-05-03T00:00:00","2005-03-22T00:00:00","2005-02-02T00:00:00","2004-12-14T00:00:00","2004-11-10T00:00:00","2004-09-21T00:00:00","2004-08-10T00:00:00","2004-06-30T00:00:00","2004-05-04T00:00:00","2004-03-16T00:00:00","2004-01-28T00:00:00","2003-12-09T00:00:00","2003-10-28T00:00:00","2003-09-16T00:00:00","2003-08-12T00:00:00","2003-06-25T00:00:00","2003-05-06T00:00:00","2003-03-18T00:00:00","2003-01-29T00:00:00","2002-12-10T00:00:00","2002-11-06T00:00:00","2002-09-24T00:00:00","2002-08-13T00:00:00","2002-06-26T00:00:00","2002-05-07T00:00:00","2002-03-19T00:00:00","2002-01-30T00:00:00","2001-12-11T00:00:00","2001-11-06T00:00:00","2001-10-02T00:00:00","2001-09-17T00:00:00","2001-08-21T00:00:00","2001-06-27T00:00:00","2001-05-15T00:00:00","2001-04-18T00:00:00","2001-03-20T00:00:00","2001-01-31T00:00:00","2001-01-03T00:00:00","2000-12-19T00:00:00","2000-11-15T00:00:00","2000-10-03T00:00:00","2000-08-22T00:00:00","2000-06-28T00:00:00","2000-05-16T00:00:00","2000-03-21T00:00:00","2000-02-02T00:00:00"],"y":[6,8,8,9,9,8,8,7,7,7,7,7,7,7,7,8,8,7,7,7,7,6,4,5,9,9,9,9,9,9,8,9,9,9,5,5,5,0,9,0,9,9,9,0,10,10,10,9,9,9,7,7,7,7,7,9,10,10,11,12,10,9,9,10,10,9,9,9,9,9,9,9,9,8,10,11,12,11,11,11,11,11,15,16,12,12,13,13,14,13,14,11,11,11,12,10,9,9,8,3,3,4,4,5,5,4,6,6,6,7,7,9,7,5,6,6,5,5,5,0,4,4,4,4,4,2,1,1,2,2,2,2,1,4,4,5,5,5,5,0,2,2,3,4,3,0,0,4,4,4,4,5,6,6,6,6,6,3,2,3,3,4,4,4,3,3,3,3,3,3,3,2,2,3,2,2,2,3,3,3,3,1,0,0,0,2,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,1,1,2,3,2,1,1,2,2,2],"type":"scatter","xaxis":"x","yaxis":"y"},{"name":"Policy Rate","x":["2024-11-07T00:00:00","2024-09-18T00:00:00","2024-07-31T00:00:00","2024-06-12T00:00:00","2024-05-01T00:00:00","2024-03-20T00:00:00","2024-01-31T00:00:00","2023-12-13T00:00:00","2023-11-01T00:00:00","2023-09-20T00:00:00","2023-07-26T00:00:00","2023-06-14T00:00:00","2023-05-03T00:00:00","2023-03-22T00:00:00","2023-02-01T00:00:00","2022-12-14T00:00:00","2022-11-02T00:00:00","2022-09-21T00:00:00","2022-07-27T00:00:00","2022-06-15T00:00:00","2022-05-04T00:00:00","2022-03-16T00:00:00","2022-01-26T00:00:00","2021-12-15T00:00:00","2021-11-03T00:00:00","2021-09-22T00:00:00","2021-07-28T00:00:00","2021-06-16T00:00:00","2021-04-28T00:00:00","2021-03-17T00:00:00","2021-01-27T00:00:00","2020-12-16T00:00:00","2020-11-05T00:00:00","2020-09-16T00:00:00","2020-07-29T00:00:00","2020-06-10T00:00:00","2020-04-29T00:00:00","2020-03-23T00:00:00","2020-03-15T00:00:00","2020-03-03T00:00:00","2020-01-29T00:00:00","2019-12-11T00:00:00","2019-10-30T00:00:00","2019-10-04T00:00:00","2019-09-18T00:00:00","2019-07-31T00:00:00","2019-06-19T00:00:00","2019-05-01T00:00:00","2019-03-20T00:00:00","2019-01-30T00:00:00","2018-12-19T00:00:00","2018-11-08T00:00:00","2018-09-26T00:00:00","2018-08-01T00:00:00","2018-06-13T00:00:00","2018-05-02T00:00:00","2018-03-21T00:00:00","2018-01-31T00:00:00","2017-12-13T00:00:00","2017-11-01T00:00:00","2017-09-20T00:00:00","2017-07-26T00:00:00","2017-06-14T00:00:00","2017-05-03T00:00:00","2017-03-15T00:00:00","2017-02-01T00:00:00","2016-12-14T00:00:00","2016-11-02T00:00:00","2016-09-21T00:00:00","2016-07-27T00:00:00","2016-06-15T00:00:00","2016-04-27T00:00:00","2016-03-16T00:00:00","2016-01-27T00:00:00","2015-12-16T00:00:00","2015-10-28T00:00:00","2015-09-17T00:00:00","2015-07-29T00:00:00","2015-06-17T00:00:00","2015-04-29T00:00:00","2015-03-18T00:00:00","2015-01-28T00:00:00","2014-12-17T00:00:00","2014-10-29T00:00:00","2014-09-17T00:00:00","2014-07-30T00:00:00","2014-06-18T00:00:00","2014-04-30T00:00:00","2014-03-19T00:00:00","2014-01-29T00:00:00","2013-12-18T00:00:00","2013-10-30T00:00:00","2013-09-18T00:00:00","2013-07-31T00:00:00","2013-06-19T00:00:00","2013-05-01T00:00:00","2013-03-20T00:00:00","2013-01-30T00:00:00","2012-12-12T00:00:00","2012-10-24T00:00:00","2012-09-13T00:00:00","2012-08-01T00:00:00","2012-06-20T00:00:00","2012-04-25T00:00:00","2012-03-13T00:00:00","2012-01-25T00:00:00","2011-12-13T00:00:00","2011-11-02T00:00:00","2011-09-21T00:00:00","2011-08-09T00:00:00","2011-06-22T00:00:00","2011-04-27T00:00:00","2011-03-15T00:00:00","2011-01-26T00:00:00","2010-12-14T00:00:00","2010-11-03T00:00:00","2010-09-21T00:00:00","2010-08-10T00:00:00","2010-06-23T00:00:00","2010-05-09T00:00:00","2010-04-28T00:00:00","2010-03-16T00:00:00","2010-01-27T00:00:00","2009-12-15T00:00:00","2009-11-04T00:00:00","2009-09-22T00:00:00","2009-08-11T00:00:00","2009-06-24T00:00:00","2009-04-29T00:00:00","2009-03-17T00:00:00","2009-01-28T00:00:00","2008-12-16T00:00:00","2008-10-29T00:00:00","2008-10-07T00:00:00","2008-09-16T00:00:00","2008-08-08T00:00:00","2008-06-25T00:00:00","2008-04-30T00:00:00","2008-03-18T00:00:00","2008-03-10T00:00:00","2008-01-30T00:00:00","2008-01-21T00:00:00","2007-12-11T00:00:00","2007-10-31T00:00:00","2007-09-18T00:00:00","2007-08-16T00:00:00","2007-08-10T00:00:00","2007-08-07T00:00:00","2007-06-28T00:00:00","2007-05-09T00:00:00","2007-03-21T00:00:00","2007-01-31T00:00:00","2006-12-12T00:00:00","2006-10-25T00:00:00","2006-09-20T00:00:00","2006-08-08T00:00:00","2006-06-29T00:00:00","2006-05-10T00:00:00","2006-03-28T00:00:00","2006-01-31T00:00:00","2005-12-13T00:00:00","2005-11-01T00:00:00","2005-09-20T00:00:00","2005-08-09T00:00:00","2005-06-30T00:00:00","2005-05-03T00:00:00","2005-03-22T00:00:00","2005-02-02T00:00:00","2004-12-14T00:00:00","2004-11-10T00:00:00","2004-09-21T00:00:00","2004-08-10T00:00:00","2004-06-30T00:00:00","2004-05-04T00:00:00","2004-03-16T00:00:00","2004-01-28T00:00:00","2003-12-09T00:00:00","2003-10-28T00:00:00","2003-09-16T00:00:00","2003-08-12T00:00:00","2003-06-25T00:00:00","2003-05-06T00:00:00","2003-03-18T00:00:00","2003-01-29T00:00:00","2002-12-10T00:00:00","2002-11-06T00:00:00","2002-09-24T00:00:00","2002-08-13T00:00:00","2002-06-26T00:00:00","2002-05-07T00:00:00","2002-03-19T00:00:00","2002-01-30T00:00:00","2001-12-11T00:00:00","2001-11-06T00:00:00","2001-10-02T00:00:00","2001-09-17T00:00:00","2001-08-21T00:00:00","2001-06-27T00:00:00","2001-05-15T00:00:00","2001-04-18T00:00:00","2001-03-20T00:00:00","2001-01-31T00:00:00","2001-01-03T00:00:00","2000-12-19T00:00:00","2000-11-15T00:00:00","2000-10-03T00:00:00","2000-08-22T00:00:00","2000-06-28T00:00:00","2000-05-16T00:00:00","2000-03-21T00:00:00","2000-02-02T00:00:00"],"y":[4.75,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.0,5.0,4.75,4.5,4.25,3.75,3.0,2.25,1.5,0.75,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.5,1.5,1.5,1.75,1.75,2.0,2.25,2.25,2.25,2.25,2.25,2.0,2.0,1.75,1.75,1.5,1.5,1.25,1.25,1.0,1.0,1.0,1.0,0.75,0.75,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,2.0,2.0,2.0,2.0,2.25,3.0,3.0,4.25,4.25,4.5,4.75,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.0,4.75,4.5,4.25,4.0,3.75,3.5,3.25,3.0,2.75,2.5,2.25,2.0,1.75,1.5,1.25,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.25,1.25,1.25,1.25,1.25,1.75,1.75,1.75,1.75,1.75,1.75,1.75,2.0,2.5,3.0,3.5,3.75,4.0,4.5,5.0,5.5,6.0,6.5,6.5,6.5,6.5,6.5,6.5,6.0,5.75],"type":"scatter","xaxis":"x","yaxis":"y2"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.94],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Frequency of keyword 'inflation"}},"yaxis2":{"anchor":"x","overlaying":"y","side":"right","title":{"text":"Policy Rate"}},"title":{"text":"Inflation Appearing Frequency vs. Policy Rate (statement information)"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('da7b23c0-08e6-4280-a612-b1aa77cb44c2');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```python
fig2 = make_subplots(specs=[[{"secondary_y": True}]])

fig2.add_trace(
    go.Scatter(
        x=merged_statements["Date"],
        y=merged_statements["unemployment"],
        name="Frequency of keyword 'unemployment'",
    ),
    secondary_y=False,
)

fig2.add_trace(
    go.Scatter(
        x=merged_statements["Date"],
        y=merged_statements["Policy Rate"],
        name="Policy Rate",
    ),
    secondary_y=True,
)

fig2.update_layout(title_text=f"Unemployment Appearing Frequency vs. Policy Rate (statement information)")

fig2.update_xaxes(title_text="Date")

fig2.update_yaxes(title_text="Frequency of keyword 'unemployment", secondary_y=False)
fig2.update_yaxes(title_text="Policy Rate", secondary_y=True)

fig2.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="7aae5b9d-862e-4eea-b815-b63bfe1d4fae" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("7aae5b9d-862e-4eea-b815-b63bfe1d4fae")) {                    Plotly.newPlot(                        "7aae5b9d-862e-4eea-b815-b63bfe1d4fae",                        [{"name":"Frequency of keyword 'unemployment'","x":["2024-11-07T00:00:00","2024-09-18T00:00:00","2024-07-31T00:00:00","2024-06-12T00:00:00","2024-05-01T00:00:00","2024-03-20T00:00:00","2024-01-31T00:00:00","2023-12-13T00:00:00","2023-11-01T00:00:00","2023-09-20T00:00:00","2023-07-26T00:00:00","2023-06-14T00:00:00","2023-05-03T00:00:00","2023-03-22T00:00:00","2023-02-01T00:00:00","2022-12-14T00:00:00","2022-11-02T00:00:00","2022-09-21T00:00:00","2022-07-27T00:00:00","2022-06-15T00:00:00","2022-05-04T00:00:00","2022-03-16T00:00:00","2022-01-26T00:00:00","2021-12-15T00:00:00","2021-11-03T00:00:00","2021-09-22T00:00:00","2021-07-28T00:00:00","2021-06-16T00:00:00","2021-04-28T00:00:00","2021-03-17T00:00:00","2021-01-27T00:00:00","2020-12-16T00:00:00","2020-11-05T00:00:00","2020-09-16T00:00:00","2020-07-29T00:00:00","2020-06-10T00:00:00","2020-04-29T00:00:00","2020-03-23T00:00:00","2020-03-15T00:00:00","2020-03-03T00:00:00","2020-01-29T00:00:00","2019-12-11T00:00:00","2019-10-30T00:00:00","2019-10-04T00:00:00","2019-09-18T00:00:00","2019-07-31T00:00:00","2019-06-19T00:00:00","2019-05-01T00:00:00","2019-03-20T00:00:00","2019-01-30T00:00:00","2018-12-19T00:00:00","2018-11-08T00:00:00","2018-09-26T00:00:00","2018-08-01T00:00:00","2018-06-13T00:00:00","2018-05-02T00:00:00","2018-03-21T00:00:00","2018-01-31T00:00:00","2017-12-13T00:00:00","2017-11-01T00:00:00","2017-09-20T00:00:00","2017-07-26T00:00:00","2017-06-14T00:00:00","2017-05-03T00:00:00","2017-03-15T00:00:00","2017-02-01T00:00:00","2016-12-14T00:00:00","2016-11-02T00:00:00","2016-09-21T00:00:00","2016-07-27T00:00:00","2016-06-15T00:00:00","2016-04-27T00:00:00","2016-03-16T00:00:00","2016-01-27T00:00:00","2015-12-16T00:00:00","2015-10-28T00:00:00","2015-09-17T00:00:00","2015-07-29T00:00:00","2015-06-17T00:00:00","2015-04-29T00:00:00","2015-03-18T00:00:00","2015-01-28T00:00:00","2014-12-17T00:00:00","2014-10-29T00:00:00","2014-09-17T00:00:00","2014-07-30T00:00:00","2014-06-18T00:00:00","2014-04-30T00:00:00","2014-03-19T00:00:00","2014-01-29T00:00:00","2013-12-18T00:00:00","2013-10-30T00:00:00","2013-09-18T00:00:00","2013-07-31T00:00:00","2013-06-19T00:00:00","2013-05-01T00:00:00","2013-03-20T00:00:00","2013-01-30T00:00:00","2012-12-12T00:00:00","2012-10-24T00:00:00","2012-09-13T00:00:00","2012-08-01T00:00:00","2012-06-20T00:00:00","2012-04-25T00:00:00","2012-03-13T00:00:00","2012-01-25T00:00:00","2011-12-13T00:00:00","2011-11-02T00:00:00","2011-09-21T00:00:00","2011-08-09T00:00:00","2011-06-22T00:00:00","2011-04-27T00:00:00","2011-03-15T00:00:00","2011-01-26T00:00:00","2010-12-14T00:00:00","2010-11-03T00:00:00","2010-09-21T00:00:00","2010-08-10T00:00:00","2010-06-23T00:00:00","2010-05-09T00:00:00","2010-04-28T00:00:00","2010-03-16T00:00:00","2010-01-27T00:00:00","2009-12-15T00:00:00","2009-11-04T00:00:00","2009-09-22T00:00:00","2009-08-11T00:00:00","2009-06-24T00:00:00","2009-04-29T00:00:00","2009-03-17T00:00:00","2009-01-28T00:00:00","2008-12-16T00:00:00","2008-10-29T00:00:00","2008-10-07T00:00:00","2008-09-16T00:00:00","2008-08-08T00:00:00","2008-06-25T00:00:00","2008-04-30T00:00:00","2008-03-18T00:00:00","2008-03-10T00:00:00","2008-01-30T00:00:00","2008-01-21T00:00:00","2007-12-11T00:00:00","2007-10-31T00:00:00","2007-09-18T00:00:00","2007-08-16T00:00:00","2007-08-10T00:00:00","2007-08-07T00:00:00","2007-06-28T00:00:00","2007-05-09T00:00:00","2007-03-21T00:00:00","2007-01-31T00:00:00","2006-12-12T00:00:00","2006-10-25T00:00:00","2006-09-20T00:00:00","2006-08-08T00:00:00","2006-06-29T00:00:00","2006-05-10T00:00:00","2006-03-28T00:00:00","2006-01-31T00:00:00","2005-12-13T00:00:00","2005-11-01T00:00:00","2005-09-20T00:00:00","2005-08-09T00:00:00","2005-06-30T00:00:00","2005-05-03T00:00:00","2005-03-22T00:00:00","2005-02-02T00:00:00","2004-12-14T00:00:00","2004-11-10T00:00:00","2004-09-21T00:00:00","2004-08-10T00:00:00","2004-06-30T00:00:00","2004-05-04T00:00:00","2004-03-16T00:00:00","2004-01-28T00:00:00","2003-12-09T00:00:00","2003-10-28T00:00:00","2003-09-16T00:00:00","2003-08-12T00:00:00","2003-06-25T00:00:00","2003-05-06T00:00:00","2003-03-18T00:00:00","2003-01-29T00:00:00","2002-12-10T00:00:00","2002-11-06T00:00:00","2002-09-24T00:00:00","2002-08-13T00:00:00","2002-06-26T00:00:00","2002-05-07T00:00:00","2002-03-19T00:00:00","2002-01-30T00:00:00","2001-12-11T00:00:00","2001-11-06T00:00:00","2001-10-02T00:00:00","2001-09-17T00:00:00","2001-08-21T00:00:00","2001-06-27T00:00:00","2001-05-15T00:00:00","2001-04-18T00:00:00","2001-03-20T00:00:00","2001-01-31T00:00:00","2001-01-03T00:00:00","2000-12-19T00:00:00","2000-11-15T00:00:00","2000-10-03T00:00:00","2000-08-22T00:00:00","2000-06-28T00:00:00","2000-05-16T00:00:00","2000-03-21T00:00:00","2000-02-02T00:00:00"],"y":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,2,4,5,3,3,3,3,3,3,3,2,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"type":"scatter","xaxis":"x","yaxis":"y"},{"name":"Policy Rate","x":["2024-11-07T00:00:00","2024-09-18T00:00:00","2024-07-31T00:00:00","2024-06-12T00:00:00","2024-05-01T00:00:00","2024-03-20T00:00:00","2024-01-31T00:00:00","2023-12-13T00:00:00","2023-11-01T00:00:00","2023-09-20T00:00:00","2023-07-26T00:00:00","2023-06-14T00:00:00","2023-05-03T00:00:00","2023-03-22T00:00:00","2023-02-01T00:00:00","2022-12-14T00:00:00","2022-11-02T00:00:00","2022-09-21T00:00:00","2022-07-27T00:00:00","2022-06-15T00:00:00","2022-05-04T00:00:00","2022-03-16T00:00:00","2022-01-26T00:00:00","2021-12-15T00:00:00","2021-11-03T00:00:00","2021-09-22T00:00:00","2021-07-28T00:00:00","2021-06-16T00:00:00","2021-04-28T00:00:00","2021-03-17T00:00:00","2021-01-27T00:00:00","2020-12-16T00:00:00","2020-11-05T00:00:00","2020-09-16T00:00:00","2020-07-29T00:00:00","2020-06-10T00:00:00","2020-04-29T00:00:00","2020-03-23T00:00:00","2020-03-15T00:00:00","2020-03-03T00:00:00","2020-01-29T00:00:00","2019-12-11T00:00:00","2019-10-30T00:00:00","2019-10-04T00:00:00","2019-09-18T00:00:00","2019-07-31T00:00:00","2019-06-19T00:00:00","2019-05-01T00:00:00","2019-03-20T00:00:00","2019-01-30T00:00:00","2018-12-19T00:00:00","2018-11-08T00:00:00","2018-09-26T00:00:00","2018-08-01T00:00:00","2018-06-13T00:00:00","2018-05-02T00:00:00","2018-03-21T00:00:00","2018-01-31T00:00:00","2017-12-13T00:00:00","2017-11-01T00:00:00","2017-09-20T00:00:00","2017-07-26T00:00:00","2017-06-14T00:00:00","2017-05-03T00:00:00","2017-03-15T00:00:00","2017-02-01T00:00:00","2016-12-14T00:00:00","2016-11-02T00:00:00","2016-09-21T00:00:00","2016-07-27T00:00:00","2016-06-15T00:00:00","2016-04-27T00:00:00","2016-03-16T00:00:00","2016-01-27T00:00:00","2015-12-16T00:00:00","2015-10-28T00:00:00","2015-09-17T00:00:00","2015-07-29T00:00:00","2015-06-17T00:00:00","2015-04-29T00:00:00","2015-03-18T00:00:00","2015-01-28T00:00:00","2014-12-17T00:00:00","2014-10-29T00:00:00","2014-09-17T00:00:00","2014-07-30T00:00:00","2014-06-18T00:00:00","2014-04-30T00:00:00","2014-03-19T00:00:00","2014-01-29T00:00:00","2013-12-18T00:00:00","2013-10-30T00:00:00","2013-09-18T00:00:00","2013-07-31T00:00:00","2013-06-19T00:00:00","2013-05-01T00:00:00","2013-03-20T00:00:00","2013-01-30T00:00:00","2012-12-12T00:00:00","2012-10-24T00:00:00","2012-09-13T00:00:00","2012-08-01T00:00:00","2012-06-20T00:00:00","2012-04-25T00:00:00","2012-03-13T00:00:00","2012-01-25T00:00:00","2011-12-13T00:00:00","2011-11-02T00:00:00","2011-09-21T00:00:00","2011-08-09T00:00:00","2011-06-22T00:00:00","2011-04-27T00:00:00","2011-03-15T00:00:00","2011-01-26T00:00:00","2010-12-14T00:00:00","2010-11-03T00:00:00","2010-09-21T00:00:00","2010-08-10T00:00:00","2010-06-23T00:00:00","2010-05-09T00:00:00","2010-04-28T00:00:00","2010-03-16T00:00:00","2010-01-27T00:00:00","2009-12-15T00:00:00","2009-11-04T00:00:00","2009-09-22T00:00:00","2009-08-11T00:00:00","2009-06-24T00:00:00","2009-04-29T00:00:00","2009-03-17T00:00:00","2009-01-28T00:00:00","2008-12-16T00:00:00","2008-10-29T00:00:00","2008-10-07T00:00:00","2008-09-16T00:00:00","2008-08-08T00:00:00","2008-06-25T00:00:00","2008-04-30T00:00:00","2008-03-18T00:00:00","2008-03-10T00:00:00","2008-01-30T00:00:00","2008-01-21T00:00:00","2007-12-11T00:00:00","2007-10-31T00:00:00","2007-09-18T00:00:00","2007-08-16T00:00:00","2007-08-10T00:00:00","2007-08-07T00:00:00","2007-06-28T00:00:00","2007-05-09T00:00:00","2007-03-21T00:00:00","2007-01-31T00:00:00","2006-12-12T00:00:00","2006-10-25T00:00:00","2006-09-20T00:00:00","2006-08-08T00:00:00","2006-06-29T00:00:00","2006-05-10T00:00:00","2006-03-28T00:00:00","2006-01-31T00:00:00","2005-12-13T00:00:00","2005-11-01T00:00:00","2005-09-20T00:00:00","2005-08-09T00:00:00","2005-06-30T00:00:00","2005-05-03T00:00:00","2005-03-22T00:00:00","2005-02-02T00:00:00","2004-12-14T00:00:00","2004-11-10T00:00:00","2004-09-21T00:00:00","2004-08-10T00:00:00","2004-06-30T00:00:00","2004-05-04T00:00:00","2004-03-16T00:00:00","2004-01-28T00:00:00","2003-12-09T00:00:00","2003-10-28T00:00:00","2003-09-16T00:00:00","2003-08-12T00:00:00","2003-06-25T00:00:00","2003-05-06T00:00:00","2003-03-18T00:00:00","2003-01-29T00:00:00","2002-12-10T00:00:00","2002-11-06T00:00:00","2002-09-24T00:00:00","2002-08-13T00:00:00","2002-06-26T00:00:00","2002-05-07T00:00:00","2002-03-19T00:00:00","2002-01-30T00:00:00","2001-12-11T00:00:00","2001-11-06T00:00:00","2001-10-02T00:00:00","2001-09-17T00:00:00","2001-08-21T00:00:00","2001-06-27T00:00:00","2001-05-15T00:00:00","2001-04-18T00:00:00","2001-03-20T00:00:00","2001-01-31T00:00:00","2001-01-03T00:00:00","2000-12-19T00:00:00","2000-11-15T00:00:00","2000-10-03T00:00:00","2000-08-22T00:00:00","2000-06-28T00:00:00","2000-05-16T00:00:00","2000-03-21T00:00:00","2000-02-02T00:00:00"],"y":[4.75,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.0,5.0,4.75,4.5,4.25,3.75,3.0,2.25,1.5,0.75,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.5,1.5,1.5,1.75,1.75,2.0,2.25,2.25,2.25,2.25,2.25,2.0,2.0,1.75,1.75,1.5,1.5,1.25,1.25,1.0,1.0,1.0,1.0,0.75,0.75,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,2.0,2.0,2.0,2.0,2.25,3.0,3.0,4.25,4.25,4.5,4.75,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.0,4.75,4.5,4.25,4.0,3.75,3.5,3.25,3.0,2.75,2.5,2.25,2.0,1.75,1.5,1.25,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.25,1.25,1.25,1.25,1.25,1.75,1.75,1.75,1.75,1.75,1.75,1.75,2.0,2.5,3.0,3.5,3.75,4.0,4.5,5.0,5.5,6.0,6.5,6.5,6.5,6.5,6.5,6.5,6.0,5.75],"type":"scatter","xaxis":"x","yaxis":"y2"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.94],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Frequency of keyword 'unemployment"}},"yaxis2":{"anchor":"x","overlaying":"y","side":"right","title":{"text":"Policy Rate"}},"title":{"text":"Unemployment Appearing Frequency vs. Policy Rate (statement information)"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('7aae5b9d-862e-4eea-b815-b63bfe1d4fae');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


#### 3.3.1.3 Data visualization from minutes


```python
fig3 = make_subplots(specs=[[{"secondary_y": True}]])

fig3.add_trace(
    go.Scatter(
        x=merged_minutes["Date"],
        y=merged_minutes["unemployment"],
        name="Frequency of keyword 'unemployment'",
    ),
    secondary_y=False,
)

fig3.add_trace(
    go.Scatter(
        x=merged_minutes["Date"],
        y=merged_minutes["Policy Rate"],
        name="Policy Rate",
    ),
    secondary_y=True,
)

fig3.update_layout(title_text=f"Unemployment Appearing Frequency vs. Policy Rate (minutes information)")

fig3.update_xaxes(title_text="Date")

fig3.update_yaxes(title_text="Frequency of keyword 'unemployment", secondary_y=False)
fig3.update_yaxes(title_text="Policy Rate", secondary_y=True)

fig3.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="548cd248-9c49-4289-b56f-c401929131c1" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("548cd248-9c49-4289-b56f-c401929131c1")) {                    Plotly.newPlot(                        "548cd248-9c49-4289-b56f-c401929131c1",                        [{"name":"Frequency of keyword 'unemployment'","x":["2024-11-07T00:00:00","2024-09-18T00:00:00","2024-07-31T00:00:00","2024-06-12T00:00:00","2024-05-01T00:00:00","2024-03-20T00:00:00","2024-01-31T00:00:00","2023-12-13T00:00:00","2023-11-01T00:00:00","2023-09-20T00:00:00","2023-07-26T00:00:00","2023-06-14T00:00:00","2023-05-03T00:00:00","2023-03-22T00:00:00","2023-02-01T00:00:00","2022-12-14T00:00:00","2022-11-02T00:00:00","2022-09-21T00:00:00","2022-07-27T00:00:00","2022-06-15T00:00:00","2022-05-04T00:00:00","2022-03-16T00:00:00","2022-01-26T00:00:00","2021-12-15T00:00:00","2021-11-03T00:00:00","2021-09-22T00:00:00","2021-07-28T00:00:00","2021-06-16T00:00:00","2021-04-28T00:00:00","2021-03-17T00:00:00","2021-01-27T00:00:00","2020-12-16T00:00:00","2020-11-05T00:00:00","2020-09-16T00:00:00","2020-07-29T00:00:00","2020-06-10T00:00:00","2020-04-29T00:00:00","2020-03-15T00:00:00","2020-03-03T00:00:00","2020-01-29T00:00:00","2019-12-11T00:00:00","2019-10-30T00:00:00","2019-10-04T00:00:00","2019-09-18T00:00:00","2019-07-31T00:00:00","2019-06-19T00:00:00","2019-05-01T00:00:00","2019-03-20T00:00:00","2019-01-30T00:00:00","2018-12-19T00:00:00","2018-11-08T00:00:00","2018-09-26T00:00:00","2018-08-01T00:00:00","2018-06-13T00:00:00","2018-05-02T00:00:00","2018-03-21T00:00:00","2018-01-31T00:00:00","2017-12-13T00:00:00","2017-11-01T00:00:00","2017-09-20T00:00:00","2017-07-26T00:00:00","2017-06-14T00:00:00","2017-05-03T00:00:00","2017-03-15T00:00:00","2017-02-01T00:00:00","2016-12-14T00:00:00","2016-11-02T00:00:00","2016-09-21T00:00:00","2016-07-27T00:00:00","2016-06-15T00:00:00","2016-04-27T00:00:00","2016-03-16T00:00:00","2016-01-27T00:00:00","2015-12-16T00:00:00","2015-10-28T00:00:00","2015-09-17T00:00:00","2015-07-29T00:00:00","2015-06-17T00:00:00","2015-04-29T00:00:00","2015-03-18T00:00:00","2015-01-28T00:00:00","2014-12-17T00:00:00","2014-10-29T00:00:00","2014-09-17T00:00:00","2014-07-30T00:00:00","2014-06-18T00:00:00","2014-04-30T00:00:00","2014-03-19T00:00:00","2014-03-04T00:00:00","2014-01-29T00:00:00","2013-12-18T00:00:00","2013-10-30T00:00:00","2013-10-16T00:00:00","2013-09-18T00:00:00","2013-07-31T00:00:00","2013-06-19T00:00:00","2013-05-01T00:00:00","2013-03-20T00:00:00","2013-01-30T00:00:00","2012-12-12T00:00:00","2012-10-24T00:00:00","2012-09-13T00:00:00","2012-08-01T00:00:00","2012-06-20T00:00:00","2012-04-25T00:00:00","2012-03-13T00:00:00","2012-01-25T00:00:00","2011-12-13T00:00:00","2011-11-28T00:00:00","2011-11-02T00:00:00","2011-09-21T00:00:00","2011-08-09T00:00:00","2011-08-01T00:00:00","2011-06-22T00:00:00","2011-04-27T00:00:00","2011-03-15T00:00:00","2011-01-26T00:00:00","2010-12-14T00:00:00","2010-11-03T00:00:00","2010-10-15T00:00:00","2010-09-21T00:00:00","2010-08-10T00:00:00","2010-06-23T00:00:00","2010-05-09T00:00:00","2010-04-28T00:00:00","2010-03-16T00:00:00","2010-01-27T00:00:00","2009-12-15T00:00:00","2009-11-04T00:00:00","2009-09-22T00:00:00","2009-08-11T00:00:00","2009-06-24T00:00:00","2009-06-03T00:00:00","2009-04-29T00:00:00","2009-03-17T00:00:00","2009-02-07T00:00:00","2009-01-28T00:00:00","2009-01-16T00:00:00","2008-12-16T00:00:00","2008-10-29T00:00:00","2008-10-07T00:00:00","2008-09-29T00:00:00","2008-09-16T00:00:00","2008-08-08T00:00:00","2008-07-24T00:00:00","2008-06-25T00:00:00","2008-04-30T00:00:00","2008-03-18T00:00:00","2008-03-10T00:00:00","2008-01-30T00:00:00","2008-01-21T00:00:00","2008-01-09T00:00:00","2007-12-11T00:00:00","2007-12-06T00:00:00","2007-10-31T00:00:00","2007-09-18T00:00:00","2007-08-16T00:00:00","2007-08-10T00:00:00","2007-08-07T00:00:00","2007-06-28T00:00:00","2007-05-09T00:00:00","2007-03-21T00:00:00","2007-01-31T00:00:00","2006-12-12T00:00:00","2006-10-25T00:00:00","2006-09-20T00:00:00","2006-08-08T00:00:00","2006-06-29T00:00:00","2006-05-10T00:00:00","2006-03-28T00:00:00","2006-01-31T00:00:00","2005-12-13T00:00:00","2005-11-01T00:00:00","2005-09-20T00:00:00","2005-08-09T00:00:00","2005-06-30T00:00:00","2005-05-03T00:00:00","2005-03-22T00:00:00","2005-02-02T00:00:00","2004-12-14T00:00:00","2004-11-10T00:00:00","2004-09-21T00:00:00","2004-08-10T00:00:00","2004-06-30T00:00:00","2004-05-04T00:00:00","2004-03-16T00:00:00","2004-01-28T00:00:00","2003-12-09T00:00:00","2003-10-28T00:00:00","2003-09-16T00:00:00","2003-09-15T00:00:00","2003-08-12T00:00:00","2003-06-25T00:00:00","2003-05-06T00:00:00","2003-04-16T00:00:00","2003-04-08T00:00:00","2003-04-01T00:00:00","2003-03-25T00:00:00","2003-03-18T00:00:00","2003-01-29T00:00:00","2002-12-10T00:00:00","2002-11-06T00:00:00","2002-09-24T00:00:00","2002-08-13T00:00:00","2002-06-26T00:00:00","2002-05-07T00:00:00","2002-03-19T00:00:00","2002-01-30T00:00:00","2001-12-11T00:00:00","2001-11-06T00:00:00","2001-10-02T00:00:00","2001-09-17T00:00:00","2001-09-13T00:00:00","2001-08-21T00:00:00","2001-06-27T00:00:00","2001-05-15T00:00:00","2001-04-18T00:00:00","2001-04-11T00:00:00","2001-03-20T00:00:00","2001-01-31T00:00:00","2001-01-03T00:00:00","2000-12-19T00:00:00","2000-11-15T00:00:00","2000-10-03T00:00:00","2000-08-22T00:00:00","2000-06-28T00:00:00","2000-05-16T00:00:00","2000-03-21T00:00:00","2000-02-02T00:00:00"],"y":[8,18,13,11,8,8,9,10,7,9,11,11,11,12,13,12,6,13,11,10,7,7,8,10,6,11,7,9,8,8,9,11,9,8,9,17,10,6,6,13,15,11,11,15,16,12,14,17,15,12,12,13,13,15,14,17,16,13,12,16,12,19,19,15,19,20,13,26,10,14,8,9,10,10,10,11,8,10,9,10,12,12,12,19,20,22,19,28,28,25,29,19,19,21,19,19,18,17,19,19,14,17,10,20,23,18,24,16,16,16,13,13,13,15,11,9,16,9,16,16,9,3,10,10,8,4,14,10,7,8,6,13,13,7,4,4,1,1,2,6,6,6,1,1,1,4,4,0,0,2,2,2,0,0,3,2,2,2,1,3,3,1,2,2,1,0,1,2,1,1,2,1,2,2,2,4,2,2,8,3,2,3,3,2,1,0,3,4,1,1,1,1,3,2,2,2,2,2,2,3,1,3,3,2,3,2,4,6,1,3,2,2,2,2,5,4,2,2,2,2,2,2,2,1,2,2,1,2,3],"type":"scatter","xaxis":"x","yaxis":"y"},{"name":"Policy Rate","x":["2024-11-07T00:00:00","2024-09-18T00:00:00","2024-07-31T00:00:00","2024-06-12T00:00:00","2024-05-01T00:00:00","2024-03-20T00:00:00","2024-01-31T00:00:00","2023-12-13T00:00:00","2023-11-01T00:00:00","2023-09-20T00:00:00","2023-07-26T00:00:00","2023-06-14T00:00:00","2023-05-03T00:00:00","2023-03-22T00:00:00","2023-02-01T00:00:00","2022-12-14T00:00:00","2022-11-02T00:00:00","2022-09-21T00:00:00","2022-07-27T00:00:00","2022-06-15T00:00:00","2022-05-04T00:00:00","2022-03-16T00:00:00","2022-01-26T00:00:00","2021-12-15T00:00:00","2021-11-03T00:00:00","2021-09-22T00:00:00","2021-07-28T00:00:00","2021-06-16T00:00:00","2021-04-28T00:00:00","2021-03-17T00:00:00","2021-01-27T00:00:00","2020-12-16T00:00:00","2020-11-05T00:00:00","2020-09-16T00:00:00","2020-07-29T00:00:00","2020-06-10T00:00:00","2020-04-29T00:00:00","2020-03-15T00:00:00","2020-03-03T00:00:00","2020-01-29T00:00:00","2019-12-11T00:00:00","2019-10-30T00:00:00","2019-10-04T00:00:00","2019-09-18T00:00:00","2019-07-31T00:00:00","2019-06-19T00:00:00","2019-05-01T00:00:00","2019-03-20T00:00:00","2019-01-30T00:00:00","2018-12-19T00:00:00","2018-11-08T00:00:00","2018-09-26T00:00:00","2018-08-01T00:00:00","2018-06-13T00:00:00","2018-05-02T00:00:00","2018-03-21T00:00:00","2018-01-31T00:00:00","2017-12-13T00:00:00","2017-11-01T00:00:00","2017-09-20T00:00:00","2017-07-26T00:00:00","2017-06-14T00:00:00","2017-05-03T00:00:00","2017-03-15T00:00:00","2017-02-01T00:00:00","2016-12-14T00:00:00","2016-11-02T00:00:00","2016-09-21T00:00:00","2016-07-27T00:00:00","2016-06-15T00:00:00","2016-04-27T00:00:00","2016-03-16T00:00:00","2016-01-27T00:00:00","2015-12-16T00:00:00","2015-10-28T00:00:00","2015-09-17T00:00:00","2015-07-29T00:00:00","2015-06-17T00:00:00","2015-04-29T00:00:00","2015-03-18T00:00:00","2015-01-28T00:00:00","2014-12-17T00:00:00","2014-10-29T00:00:00","2014-09-17T00:00:00","2014-07-30T00:00:00","2014-06-18T00:00:00","2014-04-30T00:00:00","2014-03-19T00:00:00","2014-03-04T00:00:00","2014-01-29T00:00:00","2013-12-18T00:00:00","2013-10-30T00:00:00","2013-10-16T00:00:00","2013-09-18T00:00:00","2013-07-31T00:00:00","2013-06-19T00:00:00","2013-05-01T00:00:00","2013-03-20T00:00:00","2013-01-30T00:00:00","2012-12-12T00:00:00","2012-10-24T00:00:00","2012-09-13T00:00:00","2012-08-01T00:00:00","2012-06-20T00:00:00","2012-04-25T00:00:00","2012-03-13T00:00:00","2012-01-25T00:00:00","2011-12-13T00:00:00","2011-11-28T00:00:00","2011-11-02T00:00:00","2011-09-21T00:00:00","2011-08-09T00:00:00","2011-08-01T00:00:00","2011-06-22T00:00:00","2011-04-27T00:00:00","2011-03-15T00:00:00","2011-01-26T00:00:00","2010-12-14T00:00:00","2010-11-03T00:00:00","2010-10-15T00:00:00","2010-09-21T00:00:00","2010-08-10T00:00:00","2010-06-23T00:00:00","2010-05-09T00:00:00","2010-04-28T00:00:00","2010-03-16T00:00:00","2010-01-27T00:00:00","2009-12-15T00:00:00","2009-11-04T00:00:00","2009-09-22T00:00:00","2009-08-11T00:00:00","2009-06-24T00:00:00","2009-06-03T00:00:00","2009-04-29T00:00:00","2009-03-17T00:00:00","2009-02-07T00:00:00","2009-01-28T00:00:00","2009-01-16T00:00:00","2008-12-16T00:00:00","2008-10-29T00:00:00","2008-10-07T00:00:00","2008-09-29T00:00:00","2008-09-16T00:00:00","2008-08-08T00:00:00","2008-07-24T00:00:00","2008-06-25T00:00:00","2008-04-30T00:00:00","2008-03-18T00:00:00","2008-03-10T00:00:00","2008-01-30T00:00:00","2008-01-21T00:00:00","2008-01-09T00:00:00","2007-12-11T00:00:00","2007-12-06T00:00:00","2007-10-31T00:00:00","2007-09-18T00:00:00","2007-08-16T00:00:00","2007-08-10T00:00:00","2007-08-07T00:00:00","2007-06-28T00:00:00","2007-05-09T00:00:00","2007-03-21T00:00:00","2007-01-31T00:00:00","2006-12-12T00:00:00","2006-10-25T00:00:00","2006-09-20T00:00:00","2006-08-08T00:00:00","2006-06-29T00:00:00","2006-05-10T00:00:00","2006-03-28T00:00:00","2006-01-31T00:00:00","2005-12-13T00:00:00","2005-11-01T00:00:00","2005-09-20T00:00:00","2005-08-09T00:00:00","2005-06-30T00:00:00","2005-05-03T00:00:00","2005-03-22T00:00:00","2005-02-02T00:00:00","2004-12-14T00:00:00","2004-11-10T00:00:00","2004-09-21T00:00:00","2004-08-10T00:00:00","2004-06-30T00:00:00","2004-05-04T00:00:00","2004-03-16T00:00:00","2004-01-28T00:00:00","2003-12-09T00:00:00","2003-10-28T00:00:00","2003-09-16T00:00:00","2003-09-15T00:00:00","2003-08-12T00:00:00","2003-06-25T00:00:00","2003-05-06T00:00:00","2003-04-16T00:00:00","2003-04-08T00:00:00","2003-04-01T00:00:00","2003-03-25T00:00:00","2003-03-18T00:00:00","2003-01-29T00:00:00","2002-12-10T00:00:00","2002-11-06T00:00:00","2002-09-24T00:00:00","2002-08-13T00:00:00","2002-06-26T00:00:00","2002-05-07T00:00:00","2002-03-19T00:00:00","2002-01-30T00:00:00","2001-12-11T00:00:00","2001-11-06T00:00:00","2001-10-02T00:00:00","2001-09-17T00:00:00","2001-09-13T00:00:00","2001-08-21T00:00:00","2001-06-27T00:00:00","2001-05-15T00:00:00","2001-04-18T00:00:00","2001-04-11T00:00:00","2001-03-20T00:00:00","2001-01-31T00:00:00","2001-01-03T00:00:00","2000-12-19T00:00:00","2000-11-15T00:00:00","2000-10-03T00:00:00","2000-08-22T00:00:00","2000-06-28T00:00:00","2000-05-16T00:00:00","2000-03-21T00:00:00","2000-02-02T00:00:00"],"y":[4.75,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.0,5.0,4.75,4.5,4.25,3.75,3.0,2.25,1.5,0.75,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.5,1.5,1.5,1.75,1.75,2.0,2.25,2.25,2.25,2.25,2.25,2.0,2.0,1.75,1.75,1.5,1.5,1.25,1.25,1.0,1.0,1.0,1.0,0.75,0.75,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.25,3.0,3.0,4.25,4.25,4.25,4.5,4.5,4.75,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.0,4.75,4.5,4.25,4.0,3.75,3.5,3.25,3.0,2.75,2.5,2.25,2.0,1.75,1.5,1.25,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.75,1.75,1.75,1.75,1.75,1.75,1.75,2.0,2.5,3.0,3.5,3.5,3.75,4.0,4.5,5.0,5.0,5.5,6.0,6.5,6.5,6.5,6.5,6.5,6.5,6.0,5.75],"type":"scatter","xaxis":"x","yaxis":"y2"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.94],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Frequency of keyword 'unemployment"}},"yaxis2":{"anchor":"x","overlaying":"y","side":"right","title":{"text":"Policy Rate"}},"title":{"text":"Unemployment Appearing Frequency vs. Policy Rate (minutes information)"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('548cd248-9c49-4289-b56f-c401929131c1');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```python
fig4 = make_subplots(specs=[[{"secondary_y": True}]])

fig4.add_trace(
    go.Scatter(
        x=merged_minutes["Date"],
        y=merged_minutes["inflation"],
        name="Frequency of keyword 'inflation'",
    ),
    secondary_y=False,
)

fig4.add_trace(
    go.Scatter(
        x=merged_minutes["Date"],
        y=merged_minutes["Policy Rate"],
        name="Policy Rate",
    ),
    secondary_y=True,
)

fig4.update_layout(title_text=f"Inflation Appearing Frequency vs. Policy Rate (minutes information)")

fig4.update_xaxes(title_text="Date")

fig4.update_yaxes(title_text="Frequency of keyword 'inflation", secondary_y=False)
fig4.update_yaxes(title_text="Policy Rate", secondary_y=True)

fig4.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="968d54d3-74ba-4d0b-9573-8fbd5d2703d6" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("968d54d3-74ba-4d0b-9573-8fbd5d2703d6")) {                    Plotly.newPlot(                        "968d54d3-74ba-4d0b-9573-8fbd5d2703d6",                        [{"name":"Frequency of keyword 'inflation'","x":["2024-11-07T00:00:00","2024-09-18T00:00:00","2024-07-31T00:00:00","2024-06-12T00:00:00","2024-05-01T00:00:00","2024-03-20T00:00:00","2024-01-31T00:00:00","2023-12-13T00:00:00","2023-11-01T00:00:00","2023-09-20T00:00:00","2023-07-26T00:00:00","2023-06-14T00:00:00","2023-05-03T00:00:00","2023-03-22T00:00:00","2023-02-01T00:00:00","2022-12-14T00:00:00","2022-11-02T00:00:00","2022-09-21T00:00:00","2022-07-27T00:00:00","2022-06-15T00:00:00","2022-05-04T00:00:00","2022-03-16T00:00:00","2022-01-26T00:00:00","2021-12-15T00:00:00","2021-11-03T00:00:00","2021-09-22T00:00:00","2021-07-28T00:00:00","2021-06-16T00:00:00","2021-04-28T00:00:00","2021-03-17T00:00:00","2021-01-27T00:00:00","2020-12-16T00:00:00","2020-11-05T00:00:00","2020-09-16T00:00:00","2020-07-29T00:00:00","2020-06-10T00:00:00","2020-04-29T00:00:00","2020-03-15T00:00:00","2020-03-03T00:00:00","2020-01-29T00:00:00","2019-12-11T00:00:00","2019-10-30T00:00:00","2019-10-04T00:00:00","2019-09-18T00:00:00","2019-07-31T00:00:00","2019-06-19T00:00:00","2019-05-01T00:00:00","2019-03-20T00:00:00","2019-01-30T00:00:00","2018-12-19T00:00:00","2018-11-08T00:00:00","2018-09-26T00:00:00","2018-08-01T00:00:00","2018-06-13T00:00:00","2018-05-02T00:00:00","2018-03-21T00:00:00","2018-01-31T00:00:00","2017-12-13T00:00:00","2017-11-01T00:00:00","2017-09-20T00:00:00","2017-07-26T00:00:00","2017-06-14T00:00:00","2017-05-03T00:00:00","2017-03-15T00:00:00","2017-02-01T00:00:00","2016-12-14T00:00:00","2016-11-02T00:00:00","2016-09-21T00:00:00","2016-07-27T00:00:00","2016-06-15T00:00:00","2016-04-27T00:00:00","2016-03-16T00:00:00","2016-01-27T00:00:00","2015-12-16T00:00:00","2015-10-28T00:00:00","2015-09-17T00:00:00","2015-07-29T00:00:00","2015-06-17T00:00:00","2015-04-29T00:00:00","2015-03-18T00:00:00","2015-01-28T00:00:00","2014-12-17T00:00:00","2014-10-29T00:00:00","2014-09-17T00:00:00","2014-07-30T00:00:00","2014-06-18T00:00:00","2014-04-30T00:00:00","2014-03-19T00:00:00","2014-03-04T00:00:00","2014-01-29T00:00:00","2013-12-18T00:00:00","2013-10-30T00:00:00","2013-10-16T00:00:00","2013-09-18T00:00:00","2013-07-31T00:00:00","2013-06-19T00:00:00","2013-05-01T00:00:00","2013-03-20T00:00:00","2013-01-30T00:00:00","2012-12-12T00:00:00","2012-10-24T00:00:00","2012-09-13T00:00:00","2012-08-01T00:00:00","2012-06-20T00:00:00","2012-04-25T00:00:00","2012-03-13T00:00:00","2012-01-25T00:00:00","2011-12-13T00:00:00","2011-11-28T00:00:00","2011-11-02T00:00:00","2011-09-21T00:00:00","2011-08-09T00:00:00","2011-08-01T00:00:00","2011-06-22T00:00:00","2011-04-27T00:00:00","2011-03-15T00:00:00","2011-01-26T00:00:00","2010-12-14T00:00:00","2010-11-03T00:00:00","2010-10-15T00:00:00","2010-09-21T00:00:00","2010-08-10T00:00:00","2010-06-23T00:00:00","2010-05-09T00:00:00","2010-04-28T00:00:00","2010-03-16T00:00:00","2010-01-27T00:00:00","2009-12-15T00:00:00","2009-11-04T00:00:00","2009-09-22T00:00:00","2009-08-11T00:00:00","2009-06-24T00:00:00","2009-06-03T00:00:00","2009-04-29T00:00:00","2009-03-17T00:00:00","2009-02-07T00:00:00","2009-01-28T00:00:00","2009-01-16T00:00:00","2008-12-16T00:00:00","2008-10-29T00:00:00","2008-10-07T00:00:00","2008-09-29T00:00:00","2008-09-16T00:00:00","2008-08-08T00:00:00","2008-07-24T00:00:00","2008-06-25T00:00:00","2008-04-30T00:00:00","2008-03-18T00:00:00","2008-03-10T00:00:00","2008-01-30T00:00:00","2008-01-21T00:00:00","2008-01-09T00:00:00","2007-12-11T00:00:00","2007-12-06T00:00:00","2007-10-31T00:00:00","2007-09-18T00:00:00","2007-08-16T00:00:00","2007-08-10T00:00:00","2007-08-07T00:00:00","2007-06-28T00:00:00","2007-05-09T00:00:00","2007-03-21T00:00:00","2007-01-31T00:00:00","2006-12-12T00:00:00","2006-10-25T00:00:00","2006-09-20T00:00:00","2006-08-08T00:00:00","2006-06-29T00:00:00","2006-05-10T00:00:00","2006-03-28T00:00:00","2006-01-31T00:00:00","2005-12-13T00:00:00","2005-11-01T00:00:00","2005-09-20T00:00:00","2005-08-09T00:00:00","2005-06-30T00:00:00","2005-05-03T00:00:00","2005-03-22T00:00:00","2005-02-02T00:00:00","2004-12-14T00:00:00","2004-11-10T00:00:00","2004-09-21T00:00:00","2004-08-10T00:00:00","2004-06-30T00:00:00","2004-05-04T00:00:00","2004-03-16T00:00:00","2004-01-28T00:00:00","2003-12-09T00:00:00","2003-10-28T00:00:00","2003-09-16T00:00:00","2003-09-15T00:00:00","2003-08-12T00:00:00","2003-06-25T00:00:00","2003-05-06T00:00:00","2003-04-16T00:00:00","2003-04-08T00:00:00","2003-04-01T00:00:00","2003-03-25T00:00:00","2003-03-18T00:00:00","2003-01-29T00:00:00","2002-12-10T00:00:00","2002-11-06T00:00:00","2002-09-24T00:00:00","2002-08-13T00:00:00","2002-06-26T00:00:00","2002-05-07T00:00:00","2002-03-19T00:00:00","2002-01-30T00:00:00","2001-12-11T00:00:00","2001-11-06T00:00:00","2001-10-02T00:00:00","2001-09-17T00:00:00","2001-09-13T00:00:00","2001-08-21T00:00:00","2001-06-27T00:00:00","2001-05-15T00:00:00","2001-04-18T00:00:00","2001-04-11T00:00:00","2001-03-20T00:00:00","2001-01-31T00:00:00","2001-01-03T00:00:00","2000-12-19T00:00:00","2000-11-15T00:00:00","2000-10-03T00:00:00","2000-08-22T00:00:00","2000-06-28T00:00:00","2000-05-16T00:00:00","2000-03-21T00:00:00","2000-02-02T00:00:00"],"y":[46,60,50,63,58,53,75,67,55,63,68,76,64,80,75,87,79,78,85,79,61,72,53,56,51,63,65,70,36,44,41,31,29,46,23,29,21,28,28,89,73,74,74,107,109,81,70,67,75,51,39,49,53,60,72,67,128,83,97,78,70,75,67,73,73,67,67,58,68,66,60,62,81,72,59,76,59,56,57,54,76,66,57,65,50,58,40,45,45,53,55,30,30,31,46,40,35,32,33,35,29,31,33,43,33,38,38,31,31,38,35,28,28,53,64,54,37,25,33,33,30,25,37,37,40,33,38,46,21,22,13,18,18,10,9,9,23,23,32,29,29,29,28,41,41,51,46,36,36,35,35,35,32,32,34,20,20,20,32,43,31,34,34,27,33,25,29,35,32,24,25,25,28,25,23,32,30,33,52,20,12,20,14,23,18,22,21,22,20,18,16,16,22,13,8,8,8,8,8,7,7,6,11,10,18,17,9,11,12,11,10,6,6,6,15,21,8,8,8,13,17,17,25,17,19,18,14,17,30],"type":"scatter","xaxis":"x","yaxis":"y"},{"name":"Policy Rate","x":["2024-11-07T00:00:00","2024-09-18T00:00:00","2024-07-31T00:00:00","2024-06-12T00:00:00","2024-05-01T00:00:00","2024-03-20T00:00:00","2024-01-31T00:00:00","2023-12-13T00:00:00","2023-11-01T00:00:00","2023-09-20T00:00:00","2023-07-26T00:00:00","2023-06-14T00:00:00","2023-05-03T00:00:00","2023-03-22T00:00:00","2023-02-01T00:00:00","2022-12-14T00:00:00","2022-11-02T00:00:00","2022-09-21T00:00:00","2022-07-27T00:00:00","2022-06-15T00:00:00","2022-05-04T00:00:00","2022-03-16T00:00:00","2022-01-26T00:00:00","2021-12-15T00:00:00","2021-11-03T00:00:00","2021-09-22T00:00:00","2021-07-28T00:00:00","2021-06-16T00:00:00","2021-04-28T00:00:00","2021-03-17T00:00:00","2021-01-27T00:00:00","2020-12-16T00:00:00","2020-11-05T00:00:00","2020-09-16T00:00:00","2020-07-29T00:00:00","2020-06-10T00:00:00","2020-04-29T00:00:00","2020-03-15T00:00:00","2020-03-03T00:00:00","2020-01-29T00:00:00","2019-12-11T00:00:00","2019-10-30T00:00:00","2019-10-04T00:00:00","2019-09-18T00:00:00","2019-07-31T00:00:00","2019-06-19T00:00:00","2019-05-01T00:00:00","2019-03-20T00:00:00","2019-01-30T00:00:00","2018-12-19T00:00:00","2018-11-08T00:00:00","2018-09-26T00:00:00","2018-08-01T00:00:00","2018-06-13T00:00:00","2018-05-02T00:00:00","2018-03-21T00:00:00","2018-01-31T00:00:00","2017-12-13T00:00:00","2017-11-01T00:00:00","2017-09-20T00:00:00","2017-07-26T00:00:00","2017-06-14T00:00:00","2017-05-03T00:00:00","2017-03-15T00:00:00","2017-02-01T00:00:00","2016-12-14T00:00:00","2016-11-02T00:00:00","2016-09-21T00:00:00","2016-07-27T00:00:00","2016-06-15T00:00:00","2016-04-27T00:00:00","2016-03-16T00:00:00","2016-01-27T00:00:00","2015-12-16T00:00:00","2015-10-28T00:00:00","2015-09-17T00:00:00","2015-07-29T00:00:00","2015-06-17T00:00:00","2015-04-29T00:00:00","2015-03-18T00:00:00","2015-01-28T00:00:00","2014-12-17T00:00:00","2014-10-29T00:00:00","2014-09-17T00:00:00","2014-07-30T00:00:00","2014-06-18T00:00:00","2014-04-30T00:00:00","2014-03-19T00:00:00","2014-03-04T00:00:00","2014-01-29T00:00:00","2013-12-18T00:00:00","2013-10-30T00:00:00","2013-10-16T00:00:00","2013-09-18T00:00:00","2013-07-31T00:00:00","2013-06-19T00:00:00","2013-05-01T00:00:00","2013-03-20T00:00:00","2013-01-30T00:00:00","2012-12-12T00:00:00","2012-10-24T00:00:00","2012-09-13T00:00:00","2012-08-01T00:00:00","2012-06-20T00:00:00","2012-04-25T00:00:00","2012-03-13T00:00:00","2012-01-25T00:00:00","2011-12-13T00:00:00","2011-11-28T00:00:00","2011-11-02T00:00:00","2011-09-21T00:00:00","2011-08-09T00:00:00","2011-08-01T00:00:00","2011-06-22T00:00:00","2011-04-27T00:00:00","2011-03-15T00:00:00","2011-01-26T00:00:00","2010-12-14T00:00:00","2010-11-03T00:00:00","2010-10-15T00:00:00","2010-09-21T00:00:00","2010-08-10T00:00:00","2010-06-23T00:00:00","2010-05-09T00:00:00","2010-04-28T00:00:00","2010-03-16T00:00:00","2010-01-27T00:00:00","2009-12-15T00:00:00","2009-11-04T00:00:00","2009-09-22T00:00:00","2009-08-11T00:00:00","2009-06-24T00:00:00","2009-06-03T00:00:00","2009-04-29T00:00:00","2009-03-17T00:00:00","2009-02-07T00:00:00","2009-01-28T00:00:00","2009-01-16T00:00:00","2008-12-16T00:00:00","2008-10-29T00:00:00","2008-10-07T00:00:00","2008-09-29T00:00:00","2008-09-16T00:00:00","2008-08-08T00:00:00","2008-07-24T00:00:00","2008-06-25T00:00:00","2008-04-30T00:00:00","2008-03-18T00:00:00","2008-03-10T00:00:00","2008-01-30T00:00:00","2008-01-21T00:00:00","2008-01-09T00:00:00","2007-12-11T00:00:00","2007-12-06T00:00:00","2007-10-31T00:00:00","2007-09-18T00:00:00","2007-08-16T00:00:00","2007-08-10T00:00:00","2007-08-07T00:00:00","2007-06-28T00:00:00","2007-05-09T00:00:00","2007-03-21T00:00:00","2007-01-31T00:00:00","2006-12-12T00:00:00","2006-10-25T00:00:00","2006-09-20T00:00:00","2006-08-08T00:00:00","2006-06-29T00:00:00","2006-05-10T00:00:00","2006-03-28T00:00:00","2006-01-31T00:00:00","2005-12-13T00:00:00","2005-11-01T00:00:00","2005-09-20T00:00:00","2005-08-09T00:00:00","2005-06-30T00:00:00","2005-05-03T00:00:00","2005-03-22T00:00:00","2005-02-02T00:00:00","2004-12-14T00:00:00","2004-11-10T00:00:00","2004-09-21T00:00:00","2004-08-10T00:00:00","2004-06-30T00:00:00","2004-05-04T00:00:00","2004-03-16T00:00:00","2004-01-28T00:00:00","2003-12-09T00:00:00","2003-10-28T00:00:00","2003-09-16T00:00:00","2003-09-15T00:00:00","2003-08-12T00:00:00","2003-06-25T00:00:00","2003-05-06T00:00:00","2003-04-16T00:00:00","2003-04-08T00:00:00","2003-04-01T00:00:00","2003-03-25T00:00:00","2003-03-18T00:00:00","2003-01-29T00:00:00","2002-12-10T00:00:00","2002-11-06T00:00:00","2002-09-24T00:00:00","2002-08-13T00:00:00","2002-06-26T00:00:00","2002-05-07T00:00:00","2002-03-19T00:00:00","2002-01-30T00:00:00","2001-12-11T00:00:00","2001-11-06T00:00:00","2001-10-02T00:00:00","2001-09-17T00:00:00","2001-09-13T00:00:00","2001-08-21T00:00:00","2001-06-27T00:00:00","2001-05-15T00:00:00","2001-04-18T00:00:00","2001-04-11T00:00:00","2001-03-20T00:00:00","2001-01-31T00:00:00","2001-01-03T00:00:00","2000-12-19T00:00:00","2000-11-15T00:00:00","2000-10-03T00:00:00","2000-08-22T00:00:00","2000-06-28T00:00:00","2000-05-16T00:00:00","2000-03-21T00:00:00","2000-02-02T00:00:00"],"y":[4.75,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.0,5.0,4.75,4.5,4.25,3.75,3.0,2.25,1.5,0.75,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.5,1.5,1.5,1.75,1.75,2.0,2.25,2.25,2.25,2.25,2.25,2.0,2.0,1.75,1.75,1.5,1.5,1.25,1.25,1.0,1.0,1.0,1.0,0.75,0.75,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.25,3.0,3.0,4.25,4.25,4.25,4.5,4.5,4.75,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.0,4.75,4.5,4.25,4.0,3.75,3.5,3.25,3.0,2.75,2.5,2.25,2.0,1.75,1.5,1.25,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.75,1.75,1.75,1.75,1.75,1.75,1.75,2.0,2.5,3.0,3.5,3.5,3.75,4.0,4.5,5.0,5.0,5.5,6.0,6.5,6.5,6.5,6.5,6.5,6.5,6.0,5.75],"type":"scatter","xaxis":"x","yaxis":"y2"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.94],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Frequency of keyword 'inflation"}},"yaxis2":{"anchor":"x","overlaying":"y","side":"right","title":{"text":"Policy Rate"}},"title":{"text":"Inflation Appearing Frequency vs. Policy Rate (minutes information)"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('968d54d3-74ba-4d0b-9573-8fbd5d2703d6');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


#### 3.3.1 **Conclusion**
The above graphs illustrate the following finding:
1. Minutes' record of word frequency can expose more information than statements' record does
2. Significant change in the word frequency to describe the economic outlook can foreshadow the change of interest rate
3. Specifically, the increasing use of the word "inflation" is associated with rate rising in the coming FOMC.

**Inflation Appearing Frequency vs. Policy Rate (minutes information)** shows that the increase use of "inflation" of the following periods foreshadow the rate increase in the coming FOMC meetings:
1. April 2003 to May 2005
2. Oct 2013 to Jan 2018
3. April 2020 to March 2023


**Unemployment Appearing Frequency vs. Policy Rate (minutes information)** shows that the use of "unemployment" may not be significant as the keyword "inflation" does in predicting the rate cut or rate rise movements

### 3.3.2 TF-IDF Test

The following part utilize AI to generate the code. The prompt is the following: # prompt: write tf-idf test for dataframe "minutes", only need to run tf-idf operation to the "clean no_stop text". The output are stored in a new dataframe named "tf_idf_score" which contains 3 columns. The first column store the Date, the second column stores the vocab, and the third column stores tf-idf scores. Specifically, I would like "tf_idf_score" to contains the words that have top 10 tf-idf scores in each Date. Please remove the words such as "participants", "reserve", "member", and "rate" when calculating the tf-idf score

This is the prompt after I tried to run the test for many rounds and made several adjustments.


```python

def tf_idf_top_words(df, top_n=10):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Remove specific words
    words_to_remove = ["participants", "reserve", "member", "rate"]

    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform(df['clean no_stop text'])

    # Get feature names (vocabulary)
    feature_names = vectorizer.get_feature_names_out()

    # Create an empty list to store the results
    tf_idf_results = []

    for i in range(len(df)):
        # Get the TF-IDF scores for the current document
        tfidf_scores = tfidf_matrix[i].toarray()[0]

        # Create a dictionary of words and their scores
        word_scores = dict(zip(feature_names, tfidf_scores))

        # Remove unwanted words
        for word in words_to_remove:
          if word in word_scores:
            del word_scores[word]

        # Sort the words by TF-IDF score in descending order and get the top N words
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        for word, score in top_words:
            tf_idf_results.append([df['Date'][i], word, score])

    # Create the output DataFrame
    tf_idf_score = pd.DataFrame(tf_idf_results, columns=['Date', 'vocab', 'tf-idf scores'])
    return tf_idf_score

tf_idf_score = tf_idf_top_words(minutes)
tf_idf_score.sample(5)
```





  <div id="df-5566c9a8-bd26-4824-9501-c09efe4d81d0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>vocab</th>
      <th>tf-idf scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>203</th>
      <td>2022-05-04</td>
      <td>ukraine</td>
      <td>0.149364</td>
    </tr>
    <tr>
      <th>682</th>
      <td>2016-07-27</td>
      <td>inflation</td>
      <td>0.229704</td>
    </tr>
    <tr>
      <th>802</th>
      <td>2015-01-28</td>
      <td>foreign</td>
      <td>0.195249</td>
    </tr>
    <tr>
      <th>248</th>
      <td>2021-11-03</td>
      <td>economic</td>
      <td>0.114443</td>
    </tr>
    <tr>
      <th>1854</th>
      <td>2004-03-16</td>
      <td>committee</td>
      <td>0.172509</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5566c9a8-bd26-4824-9501-c09efe4d81d0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5566c9a8-bd26-4824-9501-c09efe4d81d0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5566c9a8-bd26-4824-9501-c09efe4d81d0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-6a45e549-8af6-40fa-a179-be4916319004">
  <button class="colab-df-quickchart" onclick="quickchart('df-6a45e549-8af6-40fa-a179-be4916319004')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-6a45e549-8af6-40fa-a179-be4916319004 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




#### 3.3.2.1 Create Dataframe to store tf-idf scores for data visualization

**Function to sift the top 10 ti-idf words in each documents** <br>

**Note**: in this step, I will remove the words including ["participants", "reserve", "member", "rate", "committee", "federal"] to exclude the often appearing nouns in the documents


```python
def tf_idf_top_words(df, top_n=10):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Remove specific words
    words_to_remove = ["participants", "reserve", "member", "rate", "committee", "federal", "member"] # List of words to ignore

    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform(df['clean no_stop text'])

    # Get feature names (vocabulary)
    feature_names = vectorizer.get_feature_names_out()

    # Create an empty list to store the results
    tf_idf_results = []

    for i in range(len(df)):
        # Get the TF-IDF scores for the current document
        tfidf_scores = tfidf_matrix[i].toarray()[0]

        # Create a dictionary of words and their scores
        word_scores = dict(zip(feature_names, tfidf_scores))

        # Remove unwanted words  # This is where you remove the words
        for word in words_to_remove:
          if word in word_scores:
            del word_scores[word]

        # Sort the words by TF-IDF score in descending order and get the top N words
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        for word, score in top_words:
            tf_idf_results.append([df['Date'][i], word, score])

    # Create the output DataFrame
    tf_idf_score = pd.DataFrame(tf_idf_results, columns=['Date', 'vocab', 'tf-idf scores'])
    return tf_idf_score
```

**For minutes**


```python
top_minutes = tf_idf_top_words(minutes)
top_minutes.sample(5)
```





  <div id="df-a484ffb0-06e8-4dda-a165-49450212cab0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>vocab</th>
      <th>tf-idf scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>723</th>
      <td>2016-01-27</td>
      <td>shall</td>
      <td>0.205144</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2023-07-26</td>
      <td>percent</td>
      <td>0.132481</td>
    </tr>
    <tr>
      <th>211</th>
      <td>2022-03-16</td>
      <td>inflation</td>
      <td>0.247050</td>
    </tr>
    <tr>
      <th>456</th>
      <td>2019-06-19</td>
      <td>percent</td>
      <td>0.138588</td>
    </tr>
    <tr>
      <th>1517</th>
      <td>2008-01-09</td>
      <td>growth</td>
      <td>0.138349</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-a484ffb0-06e8-4dda-a165-49450212cab0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-a484ffb0-06e8-4dda-a165-49450212cab0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-a484ffb0-06e8-4dda-a165-49450212cab0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a3d408eb-5ff4-48bc-bc56-c29dae5784e6">
  <button class="colab-df-quickchart" onclick="quickchart('df-a3d408eb-5ff4-48bc-bc56-c29dae5784e6')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a3d408eb-5ff4-48bc-bc56-c29dae5784e6 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




**For Statements**


```python
top_statements = tf_idf_top_words(minutes)
top_statements.sample(5)
```





  <div id="df-946d2a8c-3c95-4102-9ef6-f3c82785abc5" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>vocab</th>
      <th>tf-idf scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>566</th>
      <td>2018-01-31</td>
      <td>currency</td>
      <td>0.146771</td>
    </tr>
    <tr>
      <th>2272</th>
      <td>2000-03-21</td>
      <td>acceleration</td>
      <td>0.136539</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>2002-11-06</td>
      <td>economic</td>
      <td>0.235649</td>
    </tr>
    <tr>
      <th>1814</th>
      <td>2004-09-21</td>
      <td>consumer</td>
      <td>0.135098</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>2003-09-16</td>
      <td>inflation</td>
      <td>0.115613</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-946d2a8c-3c95-4102-9ef6-f3c82785abc5')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-946d2a8c-3c95-4102-9ef6-f3c82785abc5 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-946d2a8c-3c95-4102-9ef6-f3c82785abc5');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e12f9d1e-2410-4252-97f9-e7bab16759b9">
  <button class="colab-df-quickchart" onclick="quickchart('df-e12f9d1e-2410-4252-97f9-e7bab16759b9')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e12f9d1e-2410-4252-97f9-e7bab16759b9 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




#### 3.3.2.2 Data Visualization

**For Statements**


```python
pd.options.display.max_rows = 600
# adding a little randomness to break ties in term ranking
top_tfidf_plusRand = top_statements.copy()
top_tfidf_plusRand['tf-idf scores'] = top_tfidf_plusRand['tf-idf scores'] + np.random.rand(top_statements.shape[0])*0.0001

# base for all visualizations, with rank calculation
base = alt.Chart(top_tfidf_plusRand).encode(
    x = 'rank:O',
    y = 'Date:N'
).transform_window(
    rank = "rank()",
    sort = [alt.SortField("tf-idf scores", order="descending")],
    groupby = ["Date"],
)

# heatmap specification, the color follows the number of tfidf
heatmap = base.mark_rect().encode(
    color = 'tf-idf scores:Q'
)


# text labels, white for darker heatmap colors
text = base.mark_text(baseline='middle').encode(
    text = 'vocab:N',
    color = alt.condition(alt.datum.tfidf >= 0.23, alt.value('white'), alt.value('black'))
)

# display the three superimposed visualizations
(heatmap + text).properties(width=600)
```





<style>
  #altair-viz-b88d35bdc7624a9a8ee71b9f2116da5f.vega-embed {
    width: 100%;
    display: flex;
  }

  #altair-viz-b88d35bdc7624a9a8ee71b9f2116da5f.vega-embed details,
  #altair-viz-b88d35bdc7624a9a8ee71b9f2116da5f.vega-embed details summary {
    position: relative;
  }
</style>
<div id="altair-viz-b88d35bdc7624a9a8ee71b9f2116da5f"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-b88d35bdc7624a9a8ee71b9f2116da5f") {
      outputDiv = document.getElementById("altair-viz-b88d35bdc7624a9a8ee71b9f2116da5f");
    }

    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm/vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm/vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm/vega-lite@5.20.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm/vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      let deps = ["vega-embed"];
      require(deps, displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "5.20.1"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 300, "continuousHeight": 300}}, "layer": [{"mark": {"type": "rect"}, "encoding": {"color": {"field": "tf-idf scores", "type": "quantitative"}, "x": {"field": "rank", "type": "ordinal"}, "y": {"field": "Date", "type": "nominal"}}, "transform": [{"window": [{"op": "rank", "field": "", "as": "rank"}], "groupby": ["Date"], "sort": [{"field": "tf-idf scores", "order": "descending"}]}]}, {"mark": {"type": "text", "baseline": "middle"}, "encoding": {"color": {"condition": {"test": "(datum.tfidf >= 0.23)", "value": "white"}, "value": "black"}, "text": {"field": "vocab", "type": "nominal"}, "x": {"field": "rank", "type": "ordinal"}, "y": {"field": "Date", "type": "nominal"}}, "transform": [{"window": [{"op": "rank", "field": "", "as": "rank"}], "groupby": ["Date"], "sort": [{"field": "tf-idf scores", "order": "descending"}]}]}], "data": {"name": "data-f6e22495150ea99ea8bacd314f2f4222"}, "width": 600, "$schema": "https://vega.github.io/schema/vega-lite/v5.20.1.json", "datasets": {"data-f6e22495150ea99ea8bacd314f2f4222": [{"Date": "2024-11-07T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21654284696746742}, {"Date": "2024-11-07T00:00:00", "vocab": "market", "tf-idf scores": 0.20674580169903137}, {"Date": "2024-11-07T00:00:00", "vocab": "remained", "tf-idf scores": 0.18708577993933656}, {"Date": "2024-11-07T00:00:00", "vocab": "labor", "tf-idf scores": 0.1723420480978496}, {"Date": "2024-11-07T00:00:00", "vocab": "continued", "tf-idf scores": 0.14771028018976212}, {"Date": "2024-11-07T00:00:00", "vocab": "policy", "tf-idf scores": 0.14768729482733306}, {"Date": "2024-11-07T00:00:00", "vocab": "risks", "tf-idf scores": 0.13347059993343047}, {"Date": "2024-11-07T00:00:00", "vocab": "economic", "tf-idf scores": 0.1279932861515056}, {"Date": "2024-11-07T00:00:00", "vocab": "rrp", "tf-idf scores": 0.11966462656077029}, {"Date": "2024-11-07T00:00:00", "vocab": "rates", "tf-idf scores": 0.1083293113807149}, {"Date": "2024-09-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.29784493581689814}, {"Date": "2024-09-18T00:00:00", "vocab": "policy", "tf-idf scores": 0.18287910067989055}, {"Date": "2024-09-18T00:00:00", "vocab": "market", "tf-idf scores": 0.15677935787094302}, {"Date": "2024-09-18T00:00:00", "vocab": "remained", "tf-idf scores": 0.15158971984383027}, {"Date": "2024-09-18T00:00:00", "vocab": "labor", "tf-idf scores": 0.13590218133017115}, {"Date": "2024-09-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.1202423855370023}, {"Date": "2024-09-18T00:00:00", "vocab": "risks", "tf-idf scores": 0.11544858372673067}, {"Date": "2024-09-18T00:00:00", "vocab": "continued", "tf-idf scores": 0.10979785639939278}, {"Date": "2024-09-18T00:00:00", "vocab": "july", "tf-idf scores": 0.10957152157478325}, {"Date": "2024-09-18T00:00:00", "vocab": "credit", "tf-idf scores": 0.10867343390488204}, {"Date": "2024-07-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2360264939528417}, {"Date": "2024-07-31T00:00:00", "vocab": "remained", "tf-idf scores": 0.20010086799775667}, {"Date": "2024-07-31T00:00:00", "vocab": "market", "tf-idf scores": 0.16931095336799934}, {"Date": "2024-07-31T00:00:00", "vocab": "noted", "tf-idf scores": 0.16490392361896367}, {"Date": "2024-07-31T00:00:00", "vocab": "continued", "tf-idf scores": 0.14372191467664672}, {"Date": "2024-07-31T00:00:00", "vocab": "labor", "tf-idf scores": 0.1385018058735092}, {"Date": "2024-07-31T00:00:00", "vocab": "risks", "tf-idf scores": 0.11850492621735334}, {"Date": "2024-07-31T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11804533257029079}, {"Date": "2024-07-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.11285234452768542}, {"Date": "2024-07-31T00:00:00", "vocab": "policy", "tf-idf scores": 0.10777439117760168}, {"Date": "2024-06-12T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3075672185744457}, {"Date": "2024-06-12T00:00:00", "vocab": "market", "tf-idf scores": 0.17938764398907767}, {"Date": "2024-06-12T00:00:00", "vocab": "economic", "tf-idf scores": 0.16401347364896687}, {"Date": "2024-06-12T00:00:00", "vocab": "remained", "tf-idf scores": 0.1640521844398691}, {"Date": "2024-06-12T00:00:00", "vocab": "labor", "tf-idf scores": 0.15896576604817947}, {"Date": "2024-06-12T00:00:00", "vocab": "continued", "tf-idf scores": 0.1384225479804543}, {"Date": "2024-06-12T00:00:00", "vocab": "policy", "tf-idf scores": 0.1333492477125114}, {"Date": "2024-06-12T00:00:00", "vocab": "credit", "tf-idf scores": 0.11256016812692134}, {"Date": "2024-06-12T00:00:00", "vocab": "rates", "tf-idf scores": 0.10763572958699769}, {"Date": "2024-06-12T00:00:00", "vocab": "april", "tf-idf scores": 0.10280535356201995}, {"Date": "2024-05-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.24856840119248128}, {"Date": "2024-05-01T00:00:00", "vocab": "cap", "tf-idf scores": 0.19716894908547325}, {"Date": "2024-05-01T00:00:00", "vocab": "remained", "tf-idf scores": 0.1611171027537957}, {"Date": "2024-05-01T00:00:00", "vocab": "redemption", "tf-idf scores": 0.14261393701982691}, {"Date": "2024-05-01T00:00:00", "vocab": "growth", "tf-idf scores": 0.12018515001898723}, {"Date": "2024-05-01T00:00:00", "vocab": "commented", "tf-idf scores": 0.11869299561482734}, {"Date": "2024-05-01T00:00:00", "vocab": "treasury", "tf-idf scores": 0.11842149211271072}, {"Date": "2024-05-01T00:00:00", "vocab": "agency", "tf-idf scores": 0.11664589426401359}, {"Date": "2024-05-01T00:00:00", "vocab": "recent", "tf-idf scores": 0.11508930124094296}, {"Date": "2024-05-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.11045672466677457}, {"Date": "2024-03-20T00:00:00", "vocab": "runoff", "tf-idf scores": 0.2738898280686568}, {"Date": "2024-03-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2199099059521753}, {"Date": "2024-03-20T00:00:00", "vocab": "sheet", "tf-idf scores": 0.1682851533871453}, {"Date": "2024-03-20T00:00:00", "vocab": "balance", "tf-idf scores": 0.14612600921820915}, {"Date": "2024-03-20T00:00:00", "vocab": "policy", "tf-idf scores": 0.14355554008357999}, {"Date": "2024-03-20T00:00:00", "vocab": "remained", "tf-idf scores": 0.1391362704584011}, {"Date": "2024-03-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.12569107767399385}, {"Date": "2024-03-20T00:00:00", "vocab": "january", "tf-idf scores": 0.12099785845531356}, {"Date": "2024-03-20T00:00:00", "vocab": "credit", "tf-idf scores": 0.10888050264722494}, {"Date": "2024-03-20T00:00:00", "vocab": "pace", "tf-idf scores": 0.10772149323968795}, {"Date": "2024-01-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.32294371980906306}, {"Date": "2024-01-31T00:00:00", "vocab": "remained", "tf-idf scores": 0.2079598982907585}, {"Date": "2024-01-31T00:00:00", "vocab": "policy", "tf-idf scores": 0.16820071618671448}, {"Date": "2024-01-31T00:00:00", "vocab": "continued", "tf-idf scores": 0.13280045100167867}, {"Date": "2024-01-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.13275620385942055}, {"Date": "2024-01-31T00:00:00", "vocab": "market", "tf-idf scores": 0.13279501517729117}, {"Date": "2024-01-31T00:00:00", "vocab": "noted", "tf-idf scores": 0.1200393487998707}, {"Date": "2024-01-31T00:00:00", "vocab": "better", "tf-idf scores": 0.10278910856747948}, {"Date": "2024-01-31T00:00:00", "vocab": "sustainably", "tf-idf scores": 0.0973308678791821}, {"Date": "2024-01-31T00:00:00", "vocab": "fourth", "tf-idf scores": 0.09524812490683655}, {"Date": "2023-12-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.32570342786818124}, {"Date": "2023-12-13T00:00:00", "vocab": "policy", "tf-idf scores": 0.18964201377765036}, {"Date": "2023-12-13T00:00:00", "vocab": "market", "tf-idf scores": 0.17022568323915654}, {"Date": "2023-12-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.16536510801448515}, {"Date": "2023-12-13T00:00:00", "vocab": "growth", "tf-idf scores": 0.1465425784528771}, {"Date": "2023-12-13T00:00:00", "vocab": "remained", "tf-idf scores": 0.14591025805701377}, {"Date": "2023-12-13T00:00:00", "vocab": "labor", "tf-idf scores": 0.14096609895216647}, {"Date": "2023-12-13T00:00:00", "vocab": "continued", "tf-idf scores": 0.12639039422488124}, {"Date": "2023-12-13T00:00:00", "vocab": "restrictive", "tf-idf scores": 0.11712755231491633}, {"Date": "2023-12-13T00:00:00", "vocab": "credit", "tf-idf scores": 0.11234363136439755}, {"Date": "2023-11-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.264131738116115}, {"Date": "2023-11-01T00:00:00", "vocab": "market", "tf-idf scores": 0.17941118297246958}, {"Date": "2023-11-01T00:00:00", "vocab": "remained", "tf-idf scores": 0.16946125617932115}, {"Date": "2023-11-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.14948701827186792}, {"Date": "2023-11-01T00:00:00", "vocab": "policy", "tf-idf scores": 0.14448967118525888}, {"Date": "2023-11-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.13957796523545665}, {"Date": "2023-11-01T00:00:00", "vocab": "financial", "tf-idf scores": 0.1357441642923402}, {"Date": "2023-11-01T00:00:00", "vocab": "credit", "tf-idf scores": 0.132424777787882}, {"Date": "2023-11-01T00:00:00", "vocab": "labor", "tf-idf scores": 0.1296101582218545}, {"Date": "2023-11-01T00:00:00", "vocab": "treasury", "tf-idf scores": 0.12825722414502327}, {"Date": "2023-09-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3009873129907744}, {"Date": "2023-09-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.20066420815269165}, {"Date": "2023-09-20T00:00:00", "vocab": "policy", "tf-idf scores": 0.2007457979470377}, {"Date": "2023-09-20T00:00:00", "vocab": "market", "tf-idf scores": 0.15056656100666746}, {"Date": "2023-09-20T00:00:00", "vocab": "remained", "tf-idf scores": 0.13545450188266825}, {"Date": "2023-09-20T00:00:00", "vocab": "july", "tf-idf scores": 0.13390853091483404}, {"Date": "2023-09-20T00:00:00", "vocab": "credit", "tf-idf scores": 0.12167240085692617}, {"Date": "2023-09-20T00:00:00", "vocab": "percent", "tf-idf scores": 0.11737965060992417}, {"Date": "2023-09-20T00:00:00", "vocab": "noted", "tf-idf scores": 0.11588215833589194}, {"Date": "2023-09-20T00:00:00", "vocab": "labor", "tf-idf scores": 0.11037633313952898}, {"Date": "2023-07-26T00:00:00", "vocab": "inflation", "tf-idf scores": 0.32410501605152836}, {"Date": "2023-07-26T00:00:00", "vocab": "remained", "tf-idf scores": 0.2293764003982075}, {"Date": "2023-07-26T00:00:00", "vocab": "policy", "tf-idf scores": 0.17959537322720295}, {"Date": "2023-07-26T00:00:00", "vocab": "economic", "tf-idf scores": 0.17451504523682443}, {"Date": "2023-07-26T00:00:00", "vocab": "market", "tf-idf scores": 0.1645975154238063}, {"Date": "2023-07-26T00:00:00", "vocab": "percent", "tf-idf scores": 0.13253927580548494}, {"Date": "2023-07-26T00:00:00", "vocab": "credit", "tf-idf scores": 0.13249558693976204}, {"Date": "2023-07-26T00:00:00", "vocab": "continued", "tf-idf scores": 0.12965027755704428}, {"Date": "2023-07-26T00:00:00", "vocab": "banks", "tf-idf scores": 0.12041717610564001}, {"Date": "2023-07-26T00:00:00", "vocab": "july", "tf-idf scores": 0.114018548263251}, {"Date": "2023-06-14T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3491411219362749}, {"Date": "2023-06-14T00:00:00", "vocab": "economic", "tf-idf scores": 0.19820229729598032}, {"Date": "2023-06-14T00:00:00", "vocab": "policy", "tf-idf scores": 0.16985225070981788}, {"Date": "2023-06-14T00:00:00", "vocab": "remained", "tf-idf scores": 0.16984675693402165}, {"Date": "2023-06-14T00:00:00", "vocab": "credit", "tf-idf scores": 0.15800807079637724}, {"Date": "2023-06-14T00:00:00", "vocab": "market", "tf-idf scores": 0.15575831826460157}, {"Date": "2023-06-14T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12266436484182427}, {"Date": "2023-06-14T00:00:00", "vocab": "monetary", "tf-idf scores": 0.11801034538752857}, {"Date": "2023-06-14T00:00:00", "vocab": "percent", "tf-idf scores": 0.11539808222180749}, {"Date": "2023-06-14T00:00:00", "vocab": "noted", "tf-idf scores": 0.11373256528384904}, {"Date": "2023-05-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2600461147536939}, {"Date": "2023-05-03T00:00:00", "vocab": "banking", "tf-idf scores": 0.2369965723215076}, {"Date": "2023-05-03T00:00:00", "vocab": "remained", "tf-idf scores": 0.18170504311937205}, {"Date": "2023-05-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.1692378507188055}, {"Date": "2023-05-03T00:00:00", "vocab": "stress", "tf-idf scores": 0.15639839112778633}, {"Date": "2023-05-03T00:00:00", "vocab": "market", "tf-idf scores": 0.15277655948956256}, {"Date": "2023-05-03T00:00:00", "vocab": "credit", "tf-idf scores": 0.1430744523107822}, {"Date": "2023-05-03T00:00:00", "vocab": "banks", "tf-idf scores": 0.11866331054522815}, {"Date": "2023-05-03T00:00:00", "vocab": "noted", "tf-idf scores": 0.1161308610413291}, {"Date": "2023-05-03T00:00:00", "vocab": "policy", "tf-idf scores": 0.11563469573806669}, {"Date": "2023-03-22T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3046554748794227}, {"Date": "2023-03-22T00:00:00", "vocab": "banking", "tf-idf scores": 0.24043714433262395}, {"Date": "2023-03-22T00:00:00", "vocab": "economic", "tf-idf scores": 0.18691576843459343}, {"Date": "2023-03-22T00:00:00", "vocab": "signature", "tf-idf scores": 0.18675496587375962}, {"Date": "2023-03-22T00:00:00", "vocab": "silicon", "tf-idf scores": 0.1765471598589355}, {"Date": "2023-03-22T00:00:00", "vocab": "valley", "tf-idf scores": 0.1765525975205037}, {"Date": "2023-03-22T00:00:00", "vocab": "developments", "tf-idf scores": 0.14180531218572173}, {"Date": "2023-03-22T00:00:00", "vocab": "policy", "tf-idf scores": 0.13809536437388195}, {"Date": "2023-03-22T00:00:00", "vocab": "recent", "tf-idf scores": 0.1381921950504831}, {"Date": "2023-03-22T00:00:00", "vocab": "market", "tf-idf scores": 0.13405349780423753}, {"Date": "2023-02-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2799900826389114}, {"Date": "2023-02-01T00:00:00", "vocab": "remained", "tf-idf scores": 0.19830143271833087}, {"Date": "2023-02-01T00:00:00", "vocab": "market", "tf-idf scores": 0.18661192298600449}, {"Date": "2023-02-01T00:00:00", "vocab": "policy", "tf-idf scores": 0.13220482426161836}, {"Date": "2023-02-01T00:00:00", "vocab": "growth", "tf-idf scores": 0.12494339052767309}, {"Date": "2023-02-01T00:00:00", "vocab": "noted", "tf-idf scores": 0.12112468119535146}, {"Date": "2023-02-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.10885437868812523}, {"Date": "2023-02-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.10107975878444217}, {"Date": "2023-02-01T00:00:00", "vocab": "financial", "tf-idf scores": 0.09805094522024554}, {"Date": "2023-02-01T00:00:00", "vocab": "credit", "tf-idf scores": 0.09435079991353214}, {"Date": "2022-12-14T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3649193513746411}, {"Date": "2022-12-14T00:00:00", "vocab": "remained", "tf-idf scores": 0.18244820000383347}, {"Date": "2022-12-14T00:00:00", "vocab": "economic", "tf-idf scores": 0.14340906209163132}, {"Date": "2022-12-14T00:00:00", "vocab": "restrictive", "tf-idf scores": 0.13952755239564746}, {"Date": "2022-12-14T00:00:00", "vocab": "market", "tf-idf scores": 0.1346480021882286}, {"Date": "2022-12-14T00:00:00", "vocab": "policy", "tf-idf scores": 0.12600243551275228}, {"Date": "2022-12-14T00:00:00", "vocab": "october", "tf-idf scores": 0.12203176316661479}, {"Date": "2022-12-14T00:00:00", "vocab": "credit", "tf-idf scores": 0.11543318069469602}, {"Date": "2022-12-14T00:00:00", "vocab": "war", "tf-idf scores": 0.1148369338746509}, {"Date": "2022-12-14T00:00:00", "vocab": "continued", "tf-idf scores": 0.10860883812305082}, {"Date": "2022-11-02T00:00:00", "vocab": "inflation", "tf-idf scores": 0.31306528276609064}, {"Date": "2022-11-02T00:00:00", "vocab": "market", "tf-idf scores": 0.19922953629224463}, {"Date": "2022-11-02T00:00:00", "vocab": "policy", "tf-idf scores": 0.19921889705158843}, {"Date": "2022-11-02T00:00:00", "vocab": "monetary", "tf-idf scores": 0.17081476174573884}, {"Date": "2022-11-02T00:00:00", "vocab": "economic", "tf-idf scores": 0.15855373497312364}, {"Date": "2022-11-02T00:00:00", "vocab": "remained", "tf-idf scores": 0.1504300785586944}, {"Date": "2022-11-02T00:00:00", "vocab": "financial", "tf-idf scores": 0.14765213484778605}, {"Date": "2022-11-02T00:00:00", "vocab": "noted", "tf-idf scores": 0.11025957208179271}, {"Date": "2022-11-02T00:00:00", "vocab": "restrictive", "tf-idf scores": 0.10884848605609042}, {"Date": "2022-11-02T00:00:00", "vocab": "war", "tf-idf scores": 0.1074869466393674}, {"Date": "2022-09-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3144718261944573}, {"Date": "2022-09-21T00:00:00", "vocab": "policy", "tf-idf scores": 0.20547635315641039}, {"Date": "2022-09-21T00:00:00", "vocab": "remained", "tf-idf scores": 0.159381857023311}, {"Date": "2022-09-21T00:00:00", "vocab": "market", "tf-idf scores": 0.1510283166707327}, {"Date": "2022-09-21T00:00:00", "vocab": "restrictive", "tf-idf scores": 0.1347321199536844}, {"Date": "2022-09-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.13425127931476005}, {"Date": "2022-09-21T00:00:00", "vocab": "july", "tf-idf scores": 0.1278982714771913}, {"Date": "2022-09-21T00:00:00", "vocab": "war", "tf-idf scores": 0.1246839729503096}, {"Date": "2022-09-21T00:00:00", "vocab": "continued", "tf-idf scores": 0.11742943705613551}, {"Date": "2022-09-21T00:00:00", "vocab": "labor", "tf-idf scores": 0.1174152042703947}, {"Date": "2022-07-27T00:00:00", "vocab": "inflation", "tf-idf scores": 0.32251684001452935}, {"Date": "2022-07-27T00:00:00", "vocab": "policy", "tf-idf scores": 0.18265038953930407}, {"Date": "2022-07-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.14770427761002014}, {"Date": "2022-07-27T00:00:00", "vocab": "remained", "tf-idf scores": 0.13607160237980379}, {"Date": "2022-07-27T00:00:00", "vocab": "supply", "tf-idf scores": 0.13195747652505735}, {"Date": "2022-07-27T00:00:00", "vocab": "financial", "tf-idf scores": 0.12936391038744977}, {"Date": "2022-07-27T00:00:00", "vocab": "market", "tf-idf scores": 0.12440253871257963}, {"Date": "2022-07-27T00:00:00", "vocab": "credit", "tf-idf scores": 0.12123377387363493}, {"Date": "2022-07-27T00:00:00", "vocab": "war", "tf-idf scores": 0.11556610728628383}, {"Date": "2022-07-27T00:00:00", "vocab": "growth", "tf-idf scores": 0.1092903502384479}, {"Date": "2022-06-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.32172228450552776}, {"Date": "2022-06-15T00:00:00", "vocab": "invasion", "tf-idf scores": 0.270893217647307}, {"Date": "2022-06-15T00:00:00", "vocab": "policy", "tf-idf scores": 0.17327981310278945}, {"Date": "2022-06-15T00:00:00", "vocab": "supply", "tf-idf scores": 0.15167115760671862}, {"Date": "2022-06-15T00:00:00", "vocab": "ukraine", "tf-idf scores": 0.13385281562713597}, {"Date": "2022-06-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.131979470361135}, {"Date": "2022-06-15T00:00:00", "vocab": "market", "tf-idf scores": 0.13201542031746819}, {"Date": "2022-06-15T00:00:00", "vocab": "remained", "tf-idf scores": 0.12786084166637346}, {"Date": "2022-06-15T00:00:00", "vocab": "percent", "tf-idf scores": 0.12274803146498754}, {"Date": "2022-06-15T00:00:00", "vocab": "credit", "tf-idf scores": 0.11909171293122034}, {"Date": "2022-05-04T00:00:00", "vocab": "invasion", "tf-idf scores": 0.3349548211196677}, {"Date": "2022-05-04T00:00:00", "vocab": "inflation", "tf-idf scores": 0.24864756816741918}, {"Date": "2022-05-04T00:00:00", "vocab": "lockdowns", "tf-idf scores": 0.15426816826050124}, {"Date": "2022-05-04T00:00:00", "vocab": "ukraine", "tf-idf scores": 0.14943195586126073}, {"Date": "2022-05-04T00:00:00", "vocab": "policy", "tf-idf scores": 0.14503330344557178}, {"Date": "2022-05-04T00:00:00", "vocab": "market", "tf-idf scores": 0.1409271360039323}, {"Date": "2022-05-04T00:00:00", "vocab": "supply", "tf-idf scores": 0.12898079226247253}, {"Date": "2022-05-04T00:00:00", "vocab": "continued", "tf-idf scores": 0.12017140516596676}, {"Date": "2022-05-04T00:00:00", "vocab": "remained", "tf-idf scores": 0.11602004888959444}, {"Date": "2022-05-04T00:00:00", "vocab": "economic", "tf-idf scores": 0.10363098724488896}, {"Date": "2022-03-16T00:00:00", "vocab": "invasion", "tf-idf scores": 0.3922737771549157}, {"Date": "2022-03-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2471016979401504}, {"Date": "2022-03-16T00:00:00", "vocab": "russian", "tf-idf scores": 0.21539618260492358}, {"Date": "2022-03-16T00:00:00", "vocab": "ukraine", "tf-idf scores": 0.16545639393874476}, {"Date": "2022-03-16T00:00:00", "vocab": "market", "tf-idf scores": 0.14120201222040382}, {"Date": "2022-03-16T00:00:00", "vocab": "treasury", "tf-idf scores": 0.1211515662704062}, {"Date": "2022-03-16T00:00:00", "vocab": "sheet", "tf-idf scores": 0.11592233619461682}, {"Date": "2022-03-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.10588380214854817}, {"Date": "2022-03-16T00:00:00", "vocab": "supply", "tf-idf scores": 0.10486338518693977}, {"Date": "2022-03-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.10244624097095226}, {"Date": "2022-01-26T00:00:00", "vocab": "selected", "tf-idf scores": 0.20524181707789838}, {"Date": "2022-01-26T00:00:00", "vocab": "omicron", "tf-idf scores": 0.20208904108652392}, {"Date": "2022-01-26T00:00:00", "vocab": "foreign", "tf-idf scores": 0.19878447867738458}, {"Date": "2022-01-26T00:00:00", "vocab": "market", "tf-idf scores": 0.1569659848265694}, {"Date": "2022-01-26T00:00:00", "vocab": "currency", "tf-idf scores": 0.15082132276961815}, {"Date": "2022-01-26T00:00:00", "vocab": "securities", "tf-idf scores": 0.1418935871016963}, {"Date": "2022-01-26T00:00:00", "vocab": "bank", "tf-idf scores": 0.13142936141692863}, {"Date": "2022-01-26T00:00:00", "vocab": "inflation", "tf-idf scores": 0.12819148319004448}, {"Date": "2022-01-26T00:00:00", "vocab": "policy", "tf-idf scores": 0.12557796551684292}, {"Date": "2022-01-26T00:00:00", "vocab": "eligible", "tf-idf scores": 0.11802837491597451}, {"Date": "2021-12-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18424233267322662}, {"Date": "2021-12-15T00:00:00", "vocab": "policy", "tf-idf scores": 0.18072521984967738}, {"Date": "2021-12-15T00:00:00", "vocab": "continued", "tf-idf scores": 0.15293008421420565}, {"Date": "2021-12-15T00:00:00", "vocab": "omicron", "tf-idf scores": 0.1509964361154425}, {"Date": "2021-12-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.1459934270163213}, {"Date": "2021-12-15T00:00:00", "vocab": "normalization", "tf-idf scores": 0.1415784735664173}, {"Date": "2021-12-15T00:00:00", "vocab": "sheet", "tf-idf scores": 0.13584329803302533}, {"Date": "2021-12-15T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.13262964646520228}, {"Date": "2021-12-15T00:00:00", "vocab": "labor", "tf-idf scores": 0.1251296192202319}, {"Date": "2021-12-15T00:00:00", "vocab": "market", "tf-idf scores": 0.12509079222235708}, {"Date": "2021-11-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2201460597599125}, {"Date": "2021-11-03T00:00:00", "vocab": "supply", "tf-idf scores": 0.18674714674323845}, {"Date": "2021-11-03T00:00:00", "vocab": "remained", "tf-idf scores": 0.14975542593495195}, {"Date": "2021-11-03T00:00:00", "vocab": "vaccinations", "tf-idf scores": 0.1423196921814716}, {"Date": "2021-11-03T00:00:00", "vocab": "market", "tf-idf scores": 0.14089862935872283}, {"Date": "2021-11-03T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.12597863527216158}, {"Date": "2021-11-03T00:00:00", "vocab": "continued", "tf-idf scores": 0.11887025310923804}, {"Date": "2021-11-03T00:00:00", "vocab": "asset", "tf-idf scores": 0.11474820789492146}, {"Date": "2021-11-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.11452988964647108}, {"Date": "2021-11-03T00:00:00", "vocab": "treasury", "tf-idf scores": 0.11329391287503915}, {"Date": "2021-09-22T00:00:00", "vocab": "inflation", "tf-idf scores": 0.24494278590468868}, {"Date": "2021-09-22T00:00:00", "vocab": "market", "tf-idf scores": 0.17494231563406146}, {"Date": "2021-09-22T00:00:00", "vocab": "delta", "tf-idf scores": 0.15019680205431454}, {"Date": "2021-09-22T00:00:00", "vocab": "july", "tf-idf scores": 0.14079229839329743}, {"Date": "2021-09-22T00:00:00", "vocab": "variant", "tf-idf scores": 0.139686548408948}, {"Date": "2021-09-22T00:00:00", "vocab": "supply", "tf-idf scores": 0.13745131210321196}, {"Date": "2021-09-22T00:00:00", "vocab": "labor", "tf-idf scores": 0.13218070856464814}, {"Date": "2021-09-22T00:00:00", "vocab": "remained", "tf-idf scores": 0.13217643759397432}, {"Date": "2021-09-22T00:00:00", "vocab": "asset", "tf-idf scores": 0.12948902383044572}, {"Date": "2021-09-22T00:00:00", "vocab": "tapering", "tf-idf scores": 0.1256304150905919}, {"Date": "2021-07-28T00:00:00", "vocab": "asset", "tf-idf scores": 0.2428823738216478}, {"Date": "2021-07-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21479793189518268}, {"Date": "2021-07-28T00:00:00", "vocab": "tapering", "tf-idf scores": 0.2034369646750335}, {"Date": "2021-07-28T00:00:00", "vocab": "purchases", "tf-idf scores": 0.19102563031716785}, {"Date": "2021-07-28T00:00:00", "vocab": "remained", "tf-idf scores": 0.1342198426870393}, {"Date": "2021-07-28T00:00:00", "vocab": "progress", "tf-idf scores": 0.13053589094039764}, {"Date": "2021-07-28T00:00:00", "vocab": "market", "tf-idf scores": 0.12753086068611605}, {"Date": "2021-07-28T00:00:00", "vocab": "financial", "tf-idf scores": 0.12190989282262252}, {"Date": "2021-07-28T00:00:00", "vocab": "continued", "tf-idf scores": 0.11412724974835801}, {"Date": "2021-07-28T00:00:00", "vocab": "labor", "tf-idf scores": 0.10740877935822082}, {"Date": "2021-06-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23804403565162954}, {"Date": "2021-06-16T00:00:00", "vocab": "srf", "tf-idf scores": 0.23115386721257178}, {"Date": "2021-06-16T00:00:00", "vocab": "fima", "tf-idf scores": 0.1808163904340392}, {"Date": "2021-06-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.1421426433161235}, {"Date": "2021-06-16T00:00:00", "vocab": "repo", "tf-idf scores": 0.14188751915874445}, {"Date": "2021-06-16T00:00:00", "vocab": "market", "tf-idf scores": 0.13865779356559169}, {"Date": "2021-06-16T00:00:00", "vocab": "facility", "tf-idf scores": 0.11792355619954138}, {"Date": "2021-06-16T00:00:00", "vocab": "progress", "tf-idf scores": 0.11605627158211879}, {"Date": "2021-06-16T00:00:00", "vocab": "vaccinations", "tf-idf scores": 0.11489233269972886}, {"Date": "2021-06-16T00:00:00", "vocab": "remained", "tf-idf scores": 0.10303547396911866}, {"Date": "2021-04-28T00:00:00", "vocab": "repo", "tf-idf scores": 0.2663400808245706}, {"Date": "2021-04-28T00:00:00", "vocab": "standing", "tf-idf scores": 0.19127507530973709}, {"Date": "2021-04-28T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.17112313304315202}, {"Date": "2021-04-28T00:00:00", "vocab": "continued", "tf-idf scores": 0.14486509281856197}, {"Date": "2021-04-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1407343423440717}, {"Date": "2021-04-28T00:00:00", "vocab": "noted", "tf-idf scores": 0.13721723138486297}, {"Date": "2021-04-28T00:00:00", "vocab": "market", "tf-idf scores": 0.1366582515966862}, {"Date": "2021-04-28T00:00:00", "vocab": "remained", "tf-idf scores": 0.13660356286331818}, {"Date": "2021-04-28T00:00:00", "vocab": "facility", "tf-idf scores": 0.12361980937875797}, {"Date": "2021-04-28T00:00:00", "vocab": "fima", "tf-idf scores": 0.12291725268051786}, {"Date": "2021-03-17T00:00:00", "vocab": "market", "tf-idf scores": 0.18735192753255772}, {"Date": "2021-03-17T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18300938183764662}, {"Date": "2021-03-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.15687416477791352}, {"Date": "2021-03-17T00:00:00", "vocab": "remained", "tf-idf scores": 0.148176663834633}, {"Date": "2021-03-17T00:00:00", "vocab": "continued", "tf-idf scores": 0.14384588830430312}, {"Date": "2021-03-17T00:00:00", "vocab": "rrp", "tf-idf scores": 0.13774359846812714}, {"Date": "2021-03-17T00:00:00", "vocab": "january", "tf-idf scores": 0.13134030670687433}, {"Date": "2021-03-17T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12199952979652266}, {"Date": "2021-03-17T00:00:00", "vocab": "facility", "tf-idf scores": 0.11572278654045204}, {"Date": "2021-03-17T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.11083999909879862}, {"Date": "2021-01-27T00:00:00", "vocab": "foreign", "tf-idf scores": 0.22901048912463579}, {"Date": "2021-01-27T00:00:00", "vocab": "selected", "tf-idf scores": 0.2218007016094212}, {"Date": "2021-01-27T00:00:00", "vocab": "market", "tf-idf scores": 0.16961129331546043}, {"Date": "2021-01-27T00:00:00", "vocab": "currency", "tf-idf scores": 0.16794689428668655}, {"Date": "2021-01-27T00:00:00", "vocab": "bank", "tf-idf scores": 0.16468836482483093}, {"Date": "2021-01-27T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.13484740433065995}, {"Date": "2021-01-27T00:00:00", "vocab": "eligible", "tf-idf scores": 0.1276243908429321}, {"Date": "2021-01-27T00:00:00", "vocab": "securities", "tf-idf scores": 0.12460028711289721}, {"Date": "2021-01-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.11878607673674785}, {"Date": "2021-01-27T00:00:00", "vocab": "transactions", "tf-idf scores": 0.11356224246065895}, {"Date": "2020-12-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.1971572619235748}, {"Date": "2020-12-16T00:00:00", "vocab": "remained", "tf-idf scores": 0.19263294298495776}, {"Date": "2020-12-16T00:00:00", "vocab": "vaccines", "tf-idf scores": 0.16746744748721004}, {"Date": "2020-12-16T00:00:00", "vocab": "vaccine", "tf-idf scores": 0.16650488294697888}, {"Date": "2020-12-16T00:00:00", "vocab": "market", "tf-idf scores": 0.15234339715162376}, {"Date": "2020-12-16T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.1424418388902338}, {"Date": "2020-12-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.1344353374870239}, {"Date": "2020-12-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13443570561580098}, {"Date": "2020-12-16T00:00:00", "vocab": "policy", "tf-idf scores": 0.11651265942379262}, {"Date": "2020-12-16T00:00:00", "vocab": "october", "tf-idf scores": 0.10231989407256091}, {"Date": "2020-11-05T00:00:00", "vocab": "asset", "tf-idf scores": 0.2012913137226923}, {"Date": "2020-11-05T00:00:00", "vocab": "market", "tf-idf scores": 0.17090255376729904}, {"Date": "2020-11-05T00:00:00", "vocab": "remained", "tf-idf scores": 0.1628868984537903}, {"Date": "2020-11-05T00:00:00", "vocab": "economic", "tf-idf scores": 0.15896025971179387}, {"Date": "2020-11-05T00:00:00", "vocab": "purchases", "tf-idf scores": 0.15079128383180937}, {"Date": "2020-11-05T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.13901206851275894}, {"Date": "2020-11-05T00:00:00", "vocab": "noted", "tf-idf scores": 0.12378263914553275}, {"Date": "2020-11-05T00:00:00", "vocab": "continued", "tf-idf scores": 0.12315180830056158}, {"Date": "2020-11-05T00:00:00", "vocab": "financial", "tf-idf scores": 0.12026103109868473}, {"Date": "2020-11-05T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11527711916332209}, {"Date": "2020-09-16T00:00:00", "vocab": "thomas", "tf-idf scores": 0.19981649695952974}, {"Date": "2020-09-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.17494864502927612}, {"Date": "2020-09-16T00:00:00", "vocab": "july", "tf-idf scores": 0.17397358991368264}, {"Date": "2020-09-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.1673536638565303}, {"Date": "2020-09-16T00:00:00", "vocab": "market", "tf-idf scores": 0.16735317501071628}, {"Date": "2020-09-16T00:00:00", "vocab": "guidance", "tf-idf scores": 0.14950841164388415}, {"Date": "2020-09-16T00:00:00", "vocab": "policy", "tf-idf scores": 0.1369603623386428}, {"Date": "2020-09-16T00:00:00", "vocab": "remained", "tf-idf scores": 0.129386402753576}, {"Date": "2020-09-16T00:00:00", "vocab": "consensus", "tf-idf scores": 0.12707424978296036}, {"Date": "2020-09-16T00:00:00", "vocab": "agency", "tf-idf scores": 0.11248248728433292}, {"Date": "2020-07-29T00:00:00", "vocab": "coronavirus", "tf-idf scores": 0.22478395430695658}, {"Date": "2020-07-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.17563991224084374}, {"Date": "2020-07-29T00:00:00", "vocab": "policy", "tf-idf scores": 0.15886112448797327}, {"Date": "2020-07-29T00:00:00", "vocab": "market", "tf-idf scores": 0.154727297426093}, {"Date": "2020-07-29T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.13294388091635082}, {"Date": "2020-07-29T00:00:00", "vocab": "credit", "tf-idf scores": 0.12551850721598964}, {"Date": "2020-07-29T00:00:00", "vocab": "virus", "tf-idf scores": 0.1152051101030794}, {"Date": "2020-07-29T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1087715757476771}, {"Date": "2020-07-29T00:00:00", "vocab": "intermeeting", "tf-idf scores": 0.10497314508243223}, {"Date": "2020-07-29T00:00:00", "vocab": "monetary", "tf-idf scores": 0.10456201235333698}, {"Date": "2020-06-10T00:00:00", "vocab": "yct", "tf-idf scores": 0.27557806935813955}, {"Date": "2020-06-10T00:00:00", "vocab": "coronavirus", "tf-idf scores": 0.21258535326374106}, {"Date": "2020-06-10T00:00:00", "vocab": "policy", "tf-idf scores": 0.17817333958770523}, {"Date": "2020-06-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.16785159329610402}, {"Date": "2020-06-10T00:00:00", "vocab": "market", "tf-idf scores": 0.15423992632808983}, {"Date": "2020-06-10T00:00:00", "vocab": "agency", "tf-idf scores": 0.13985568881709612}, {"Date": "2020-06-10T00:00:00", "vocab": "guidance", "tf-idf scores": 0.13463857861607587}, {"Date": "2020-06-10T00:00:00", "vocab": "forward", "tf-idf scores": 0.10664712463415248}, {"Date": "2020-06-10T00:00:00", "vocab": "support", "tf-idf scores": 0.10653841318333562}, {"Date": "2020-06-10T00:00:00", "vocab": "monetary", "tf-idf scores": 0.1028284690714452}, {"Date": "2020-04-29T00:00:00", "vocab": "coronavirus", "tf-idf scores": 0.3163414887376736}, {"Date": "2020-04-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.1930658111128495}, {"Date": "2020-04-29T00:00:00", "vocab": "outbreak", "tf-idf scores": 0.19030565474525057}, {"Date": "2020-04-29T00:00:00", "vocab": "market", "tf-idf scores": 0.17128322891961678}, {"Date": "2020-04-29T00:00:00", "vocab": "financial", "tf-idf scores": 0.13963718857266236}, {"Date": "2020-04-29T00:00:00", "vocab": "flow", "tf-idf scores": 0.12569308394407955}, {"Date": "2020-04-29T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11663752040039867}, {"Date": "2020-04-29T00:00:00", "vocab": "credit", "tf-idf scores": 0.10940701896048297}, {"Date": "2020-04-29T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.1043285552889532}, {"Date": "2020-04-29T00:00:00", "vocab": "agency", "tf-idf scores": 0.09746418703190446}, {"Date": "2020-03-15T00:00:00", "vocab": "coronavirus", "tf-idf scores": 0.3955558049957066}, {"Date": "2020-03-15T00:00:00", "vocab": "outbreak", "tf-idf scores": 0.22839110370123125}, {"Date": "2020-03-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.22570617029433718}, {"Date": "2020-03-15T00:00:00", "vocab": "market", "tf-idf scores": 0.16070869859657824}, {"Date": "2020-03-15T00:00:00", "vocab": "activity", "tf-idf scores": 0.1109455766748441}, {"Date": "2020-03-15T00:00:00", "vocab": "credit", "tf-idf scores": 0.11047607743293278}, {"Date": "2020-03-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.10712311731988292}, {"Date": "2020-03-15T00:00:00", "vocab": "range", "tf-idf scores": 0.10240396154333192}, {"Date": "2020-03-15T00:00:00", "vocab": "households", "tf-idf scores": 0.09305209803471298}, {"Date": "2020-03-15T00:00:00", "vocab": "noted", "tf-idf scores": 0.09223666041187094}, {"Date": "2020-03-03T00:00:00", "vocab": "coronavirus", "tf-idf scores": 0.3955968047048527}, {"Date": "2020-03-03T00:00:00", "vocab": "outbreak", "tf-idf scores": 0.22837004678557138}, {"Date": "2020-03-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.2257495617165914}, {"Date": "2020-03-03T00:00:00", "vocab": "market", "tf-idf scores": 0.16074682268817228}, {"Date": "2020-03-03T00:00:00", "vocab": "activity", "tf-idf scores": 0.1109403585855851}, {"Date": "2020-03-03T00:00:00", "vocab": "credit", "tf-idf scores": 0.11051112982806136}, {"Date": "2020-03-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.10716559221005521}, {"Date": "2020-03-03T00:00:00", "vocab": "range", "tf-idf scores": 0.1024544244163684}, {"Date": "2020-03-03T00:00:00", "vocab": "households", "tf-idf scores": 0.09303150824673152}, {"Date": "2020-03-03T00:00:00", "vocab": "noted", "tf-idf scores": 0.09222816989502335}, {"Date": "2020-01-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23823296815268918}, {"Date": "2020-01-29T00:00:00", "vocab": "selected", "tf-idf scores": 0.21001777750762016}, {"Date": "2020-01-29T00:00:00", "vocab": "foreign", "tf-idf scores": 0.19279663612135592}, {"Date": "2020-01-29T00:00:00", "vocab": "market", "tf-idf scores": 0.1874562167963509}, {"Date": "2020-01-29T00:00:00", "vocab": "bank", "tf-idf scores": 0.15056235916797225}, {"Date": "2020-01-29T00:00:00", "vocab": "financial", "tf-idf scores": 0.14580141474713865}, {"Date": "2020-01-29T00:00:00", "vocab": "currency", "tf-idf scores": 0.1450154738364559}, {"Date": "2020-01-29T00:00:00", "vocab": "policy", "tf-idf scores": 0.13920121762686646}, {"Date": "2020-01-29T00:00:00", "vocab": "operations", "tf-idf scores": 0.13082071736355058}, {"Date": "2020-01-29T00:00:00", "vocab": "eligible", "tf-idf scores": 0.11365736873701075}, {"Date": "2019-12-11T00:00:00", "vocab": "inflation", "tf-idf scores": 0.29973932733391057}, {"Date": "2019-12-11T00:00:00", "vocab": "representatives", "tf-idf scores": 0.188788175498321}, {"Date": "2019-12-11T00:00:00", "vocab": "market", "tf-idf scores": 0.17657787645129325}, {"Date": "2019-12-11T00:00:00", "vocab": "economic", "tf-idf scores": 0.15602907069310182}, {"Date": "2019-12-11T00:00:00", "vocab": "policy", "tf-idf scores": 0.1437861917889945}, {"Date": "2019-12-11T00:00:00", "vocab": "remained", "tf-idf scores": 0.13964723791622782}, {"Date": "2019-12-11T00:00:00", "vocab": "october", "tf-idf scores": 0.13703223530122238}, {"Date": "2019-12-11T00:00:00", "vocab": "monetary", "tf-idf scores": 0.13557861088378864}, {"Date": "2019-12-11T00:00:00", "vocab": "labor", "tf-idf scores": 0.13147144640649733}, {"Date": "2019-12-11T00:00:00", "vocab": "listens", "tf-idf scores": 0.12446146912401311}, {"Date": "2019-10-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23427240201150776}, {"Date": "2019-10-30T00:00:00", "vocab": "repo", "tf-idf scores": 0.21357055941225783}, {"Date": "2019-10-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.16682805225773711}, {"Date": "2019-10-30T00:00:00", "vocab": "market", "tf-idf scores": 0.14763939034108028}, {"Date": "2019-10-30T00:00:00", "vocab": "remained", "tf-idf scores": 0.147591428617862}, {"Date": "2019-10-30T00:00:00", "vocab": "operations", "tf-idf scores": 0.1440400582230926}, {"Date": "2019-10-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.141240261950732}, {"Date": "2019-10-30T00:00:00", "vocab": "funds", "tf-idf scores": 0.12195749392221758}, {"Date": "2019-10-30T00:00:00", "vocab": "symmetric", "tf-idf scores": 0.11217881054548098}, {"Date": "2019-10-30T00:00:00", "vocab": "financial", "tf-idf scores": 0.10359458424505413}, {"Date": "2019-10-04T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23420355366831527}, {"Date": "2019-10-04T00:00:00", "vocab": "repo", "tf-idf scores": 0.21360004391711077}, {"Date": "2019-10-04T00:00:00", "vocab": "economic", "tf-idf scores": 0.1668772056121777}, {"Date": "2019-10-04T00:00:00", "vocab": "market", "tf-idf scores": 0.14757978335807354}, {"Date": "2019-10-04T00:00:00", "vocab": "remained", "tf-idf scores": 0.14766156153461385}, {"Date": "2019-10-04T00:00:00", "vocab": "operations", "tf-idf scores": 0.1440227969400664}, {"Date": "2019-10-04T00:00:00", "vocab": "policy", "tf-idf scores": 0.1411556530916195}, {"Date": "2019-10-04T00:00:00", "vocab": "funds", "tf-idf scores": 0.12199224409762642}, {"Date": "2019-10-04T00:00:00", "vocab": "symmetric", "tf-idf scores": 0.11216091012310338}, {"Date": "2019-10-04T00:00:00", "vocab": "financial", "tf-idf scores": 0.10355356442013318}, {"Date": "2019-09-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3188454060152669}, {"Date": "2019-09-18T00:00:00", "vocab": "makeup", "tf-idf scores": 0.21684110699846157}, {"Date": "2019-09-18T00:00:00", "vocab": "policy", "tf-idf scores": 0.2023173262047927}, {"Date": "2019-09-18T00:00:00", "vocab": "strategies", "tf-idf scores": 0.16993912653670004}, {"Date": "2019-09-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.1564287426725532}, {"Date": "2019-09-18T00:00:00", "vocab": "percent", "tf-idf scores": 0.13686351891923815}, {"Date": "2019-09-18T00:00:00", "vocab": "market", "tf-idf scores": 0.13187954083827633}, {"Date": "2019-09-18T00:00:00", "vocab": "july", "tf-idf scores": 0.12853803486743084}, {"Date": "2019-09-18T00:00:00", "vocab": "elb", "tf-idf scores": 0.11148238201338191}, {"Date": "2019-09-18T00:00:00", "vocab": "growth", "tf-idf scores": 0.10472175970276204}, {"Date": "2019-07-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.34136525328013495}, {"Date": "2019-07-31T00:00:00", "vocab": "policy", "tf-idf scores": 0.25605252626110087}, {"Date": "2019-07-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.18019470482432476}, {"Date": "2019-07-31T00:00:00", "vocab": "market", "tf-idf scores": 0.1517322813188694}, {"Date": "2019-07-31T00:00:00", "vocab": "elb", "tf-idf scores": 0.1150130673656927}, {"Date": "2019-07-31T00:00:00", "vocab": "monetary", "tf-idf scores": 0.11381443051569919}, {"Date": "2019-07-31T00:00:00", "vocab": "growth", "tf-idf scores": 0.10798596129930005}, {"Date": "2019-07-31T00:00:00", "vocab": "percent", "tf-idf scores": 0.10749360426639235}, {"Date": "2019-07-31T00:00:00", "vocab": "remained", "tf-idf scores": 0.10753802275665036}, {"Date": "2019-07-31T00:00:00", "vocab": "range", "tf-idf scores": 0.09516308615536981}, {"Date": "2019-06-19T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3260899694757695}, {"Date": "2019-06-19T00:00:00", "vocab": "economic", "tf-idf scores": 0.20782578055441953}, {"Date": "2019-06-19T00:00:00", "vocab": "market", "tf-idf scores": 0.179331473051801}, {"Date": "2019-06-19T00:00:00", "vocab": "facility", "tf-idf scores": 0.1487404897662545}, {"Date": "2019-06-19T00:00:00", "vocab": "noted", "tf-idf scores": 0.14326367026588022}, {"Date": "2019-06-19T00:00:00", "vocab": "symmetric", "tf-idf scores": 0.1424497546536848}, {"Date": "2019-06-19T00:00:00", "vocab": "percent", "tf-idf scores": 0.1386555783328589}, {"Date": "2019-06-19T00:00:00", "vocab": "april", "tf-idf scores": 0.13377803571935654}, {"Date": "2019-06-19T00:00:00", "vocab": "growth", "tf-idf scores": 0.10647866792111296}, {"Date": "2019-06-19T00:00:00", "vocab": "repo", "tf-idf scores": 0.09953822096934466}, {"Date": "2019-05-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.24992264450547186}, {"Date": "2019-05-01T00:00:00", "vocab": "portfolio", "tf-idf scores": 0.23442742362626512}, {"Date": "2019-05-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.1964192985033311}, {"Date": "2019-05-01T00:00:00", "vocab": "maturity", "tf-idf scores": 0.1468861969557618}, {"Date": "2019-05-01T00:00:00", "vocab": "market", "tf-idf scores": 0.14287143590590756}, {"Date": "2019-05-01T00:00:00", "vocab": "composition", "tf-idf scores": 0.13861986118834}, {"Date": "2019-05-01T00:00:00", "vocab": "financial", "tf-idf scores": 0.13327927559001468}, {"Date": "2019-05-01T00:00:00", "vocab": "growth", "tf-idf scores": 0.13266609166747825}, {"Date": "2019-05-01T00:00:00", "vocab": "shorter", "tf-idf scores": 0.12804663312357314}, {"Date": "2019-05-01T00:00:00", "vocab": "target", "tf-idf scores": 0.1269315560472797}, {"Date": "2019-03-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2209864539727807}, {"Date": "2019-03-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.19644073306137372}, {"Date": "2019-03-20T00:00:00", "vocab": "reserves", "tf-idf scores": 0.19362327774597976}, {"Date": "2019-03-20T00:00:00", "vocab": "growth", "tf-idf scores": 0.162014239426516}, {"Date": "2019-03-20T00:00:00", "vocab": "market", "tf-idf scores": 0.15781031430398823}, {"Date": "2019-03-20T00:00:00", "vocab": "remained", "tf-idf scores": 0.13679501780735506}, {"Date": "2019-03-20T00:00:00", "vocab": "policy", "tf-idf scores": 0.12981977353412327}, {"Date": "2019-03-20T00:00:00", "vocab": "recent", "tf-idf scores": 0.1297932376289403}, {"Date": "2019-03-20T00:00:00", "vocab": "securities", "tf-idf scores": 0.12280428087636101}, {"Date": "2019-03-20T00:00:00", "vocab": "level", "tf-idf scores": 0.11159915141556245}, {"Date": "2019-01-30T00:00:00", "vocab": "market", "tf-idf scores": 0.21362163192371836}, {"Date": "2019-01-30T00:00:00", "vocab": "selected", "tf-idf scores": 0.20080700940796756}, {"Date": "2019-01-30T00:00:00", "vocab": "foreign", "tf-idf scores": 0.19126818184836347}, {"Date": "2019-01-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18631503714841274}, {"Date": "2019-01-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.15898720681809622}, {"Date": "2019-01-30T00:00:00", "vocab": "financial", "tf-idf scores": 0.13535532137109987}, {"Date": "2019-01-30T00:00:00", "vocab": "currency", "tf-idf scores": 0.13452774872438034}, {"Date": "2019-01-30T00:00:00", "vocab": "bank", "tf-idf scores": 0.12471893743997413}, {"Date": "2019-01-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.12417223770835133}, {"Date": "2019-01-30T00:00:00", "vocab": "securities", "tf-idf scores": 0.109421681774401}, {"Date": "2018-12-19T00:00:00", "vocab": "economic", "tf-idf scores": 0.22828010708671426}, {"Date": "2018-12-19T00:00:00", "vocab": "inflation", "tf-idf scores": 0.20745408982292532}, {"Date": "2018-12-19T00:00:00", "vocab": "market", "tf-idf scores": 0.20747860234152415}, {"Date": "2018-12-19T00:00:00", "vocab": "growth", "tf-idf scores": 0.15427778350979876}, {"Date": "2018-12-19T00:00:00", "vocab": "financial", "tf-idf scores": 0.13399904211197336}, {"Date": "2018-12-19T00:00:00", "vocab": "policy", "tf-idf scores": 0.1286631988754823}, {"Date": "2018-12-19T00:00:00", "vocab": "liabilities", "tf-idf scores": 0.12506678303659127}, {"Date": "2018-12-19T00:00:00", "vocab": "funds", "tf-idf scores": 0.12035494806100326}, {"Date": "2018-12-19T00:00:00", "vocab": "recent", "tf-idf scores": 0.1203913821044152}, {"Date": "2018-12-19T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11211843727992483}, {"Date": "2018-11-08T00:00:00", "vocab": "regime", "tf-idf scores": 0.22114957685634437}, {"Date": "2018-11-08T00:00:00", "vocab": "market", "tf-idf scores": 0.19541506580542728}, {"Date": "2018-11-08T00:00:00", "vocab": "economic", "tf-idf scores": 0.16631439726241376}, {"Date": "2018-11-08T00:00:00", "vocab": "reserves", "tf-idf scores": 0.1640198785379599}, {"Date": "2018-11-08T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15382670182175967}, {"Date": "2018-11-08T00:00:00", "vocab": "obfr", "tf-idf scores": 0.14331970260175134}, {"Date": "2018-11-08T00:00:00", "vocab": "rates", "tf-idf scores": 0.1413818581837415}, {"Date": "2018-11-08T00:00:00", "vocab": "funds", "tf-idf scores": 0.12473894502648425}, {"Date": "2018-11-08T00:00:00", "vocab": "abundant", "tf-idf scores": 0.12036037173585588}, {"Date": "2018-11-08T00:00:00", "vocab": "effr", "tf-idf scores": 0.12041325028959535}, {"Date": "2018-09-26T00:00:00", "vocab": "inflation", "tf-idf scores": 0.25315630392265404}, {"Date": "2018-09-26T00:00:00", "vocab": "economic", "tf-idf scores": 0.19515532858106957}, {"Date": "2018-09-26T00:00:00", "vocab": "market", "tf-idf scores": 0.15827805377556772}, {"Date": "2018-09-26T00:00:00", "vocab": "policy", "tf-idf scores": 0.147671006489943}, {"Date": "2018-09-26T00:00:00", "vocab": "growth", "tf-idf scores": 0.1377283759182233}, {"Date": "2018-09-26T00:00:00", "vocab": "funds", "tf-idf scores": 0.13721169849362938}, {"Date": "2018-09-26T00:00:00", "vocab": "continued", "tf-idf scores": 0.13187907221993037}, {"Date": "2018-09-26T00:00:00", "vocab": "labor", "tf-idf scores": 0.1265819461332897}, {"Date": "2018-09-26T00:00:00", "vocab": "percent", "tf-idf scores": 0.12332503021895512}, {"Date": "2018-09-26T00:00:00", "vocab": "recent", "tf-idf scores": 0.11608843211800449}, {"Date": "2018-08-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.20904868379897404}, {"Date": "2018-08-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.20492662590453467}, {"Date": "2018-08-01T00:00:00", "vocab": "elb", "tf-idf scores": 0.19871794319543273}, {"Date": "2018-08-01T00:00:00", "vocab": "policy", "tf-idf scores": 0.1721534994365072}, {"Date": "2018-08-01T00:00:00", "vocab": "market", "tf-idf scores": 0.16809475804339716}, {"Date": "2018-08-01T00:00:00", "vocab": "june", "tf-idf scores": 0.1339116133104604}, {"Date": "2018-08-01T00:00:00", "vocab": "funds", "tf-idf scores": 0.127139027951055}, {"Date": "2018-08-01T00:00:00", "vocab": "labor", "tf-idf scores": 0.1230243474850259}, {"Date": "2018-08-01T00:00:00", "vocab": "tariff", "tf-idf scores": 0.10353703260985547}, {"Date": "2018-08-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.10247531012365807}, {"Date": "2018-06-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2878668613429842}, {"Date": "2018-06-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.18370838173946943}, {"Date": "2018-06-13T00:00:00", "vocab": "market", "tf-idf scores": 0.16875919370975856}, {"Date": "2018-06-13T00:00:00", "vocab": "percent", "tf-idf scores": 0.1530490159243121}, {"Date": "2018-06-13T00:00:00", "vocab": "funds", "tf-idf scores": 0.1489695660459308}, {"Date": "2018-06-13T00:00:00", "vocab": "recent", "tf-idf scores": 0.1390046625153046}, {"Date": "2018-06-13T00:00:00", "vocab": "labor", "tf-idf scores": 0.13404257083258944}, {"Date": "2018-06-13T00:00:00", "vocab": "growth", "tf-idf scores": 0.1296102813858308}, {"Date": "2018-06-13T00:00:00", "vocab": "policy", "tf-idf scores": 0.129124758568498}, {"Date": "2018-06-13T00:00:00", "vocab": "real", "tf-idf scores": 0.12185626792764667}, {"Date": "2018-05-02T00:00:00", "vocab": "inflation", "tf-idf scores": 0.30047767303431827}, {"Date": "2018-05-02T00:00:00", "vocab": "economic", "tf-idf scores": 0.18784427426222422}, {"Date": "2018-05-02T00:00:00", "vocab": "funds", "tf-idf scores": 0.17533990234640118}, {"Date": "2018-05-02T00:00:00", "vocab": "market", "tf-idf scores": 0.16690607999106502}, {"Date": "2018-05-02T00:00:00", "vocab": "labor", "tf-idf scores": 0.14189011409155072}, {"Date": "2018-05-02T00:00:00", "vocab": "recent", "tf-idf scores": 0.14192367833389957}, {"Date": "2018-05-02T00:00:00", "vocab": "march", "tf-idf scores": 0.14064831952092383}, {"Date": "2018-05-02T00:00:00", "vocab": "symmetric", "tf-idf scores": 0.13367625995512067}, {"Date": "2018-05-02T00:00:00", "vocab": "percent", "tf-idf scores": 0.1331156703757618}, {"Date": "2018-05-02T00:00:00", "vocab": "policy", "tf-idf scores": 0.1294071684355243}, {"Date": "2018-03-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3032649586836472}, {"Date": "2018-03-21T00:00:00", "vocab": "market", "tf-idf scores": 0.22056283675871316}, {"Date": "2018-03-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.2113814931599211}, {"Date": "2018-03-21T00:00:00", "vocab": "funds", "tf-idf scores": 0.14241892568126513}, {"Date": "2018-03-21T00:00:00", "vocab": "recent", "tf-idf scores": 0.14247589072629488}, {"Date": "2018-03-21T00:00:00", "vocab": "conditions", "tf-idf scores": 0.13329940620638658}, {"Date": "2018-03-21T00:00:00", "vocab": "labor", "tf-idf scores": 0.13330454027762387}, {"Date": "2018-03-21T00:00:00", "vocab": "january", "tf-idf scores": 0.13120327119331615}, {"Date": "2018-03-21T00:00:00", "vocab": "percent", "tf-idf scores": 0.11724258641170557}, {"Date": "2018-03-21T00:00:00", "vocab": "continued", "tf-idf scores": 0.1149008594916899}, {"Date": "2018-01-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.34696092354879937}, {"Date": "2018-01-31T00:00:00", "vocab": "selected", "tf-idf scores": 0.21918397630839193}, {"Date": "2018-01-31T00:00:00", "vocab": "market", "tf-idf scores": 0.21415211309813117}, {"Date": "2018-01-31T00:00:00", "vocab": "foreign", "tf-idf scores": 0.20062063696675853}, {"Date": "2018-01-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.17079527774203032}, {"Date": "2018-01-31T00:00:00", "vocab": "bank", "tf-idf scores": 0.14704996959909888}, {"Date": "2018-01-31T00:00:00", "vocab": "currency", "tf-idf scores": 0.14683826291032093}, {"Date": "2018-01-31T00:00:00", "vocab": "eligible", "tf-idf scores": 0.11514987416955404}, {"Date": "2018-01-31T00:00:00", "vocab": "securities", "tf-idf scores": 0.11026779995323444}, {"Date": "2018-01-31T00:00:00", "vocab": "shall", "tf-idf scores": 0.10798729285559902}, {"Date": "2017-12-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3925037518668161}, {"Date": "2017-12-13T00:00:00", "vocab": "market", "tf-idf scores": 0.19628803784544652}, {"Date": "2017-12-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.18671079292492634}, {"Date": "2017-12-13T00:00:00", "vocab": "percent", "tf-idf scores": 0.14759103700322698}, {"Date": "2017-12-13T00:00:00", "vocab": "labor", "tf-idf scores": 0.14362997006922137}, {"Date": "2017-12-13T00:00:00", "vocab": "continued", "tf-idf scores": 0.13399697112211306}, {"Date": "2017-12-13T00:00:00", "vocab": "treasury", "tf-idf scores": 0.12828994393401552}, {"Date": "2017-12-13T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12443441041353792}, {"Date": "2017-12-13T00:00:00", "vocab": "range", "tf-idf scores": 0.11742399366613514}, {"Date": "2017-12-13T00:00:00", "vocab": "remained", "tf-idf scores": 0.11485500101423378}, {"Date": "2017-11-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.42375927828983934}, {"Date": "2017-11-01T00:00:00", "vocab": "market", "tf-idf scores": 0.21410776111043434}, {"Date": "2017-11-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.16170121058760561}, {"Date": "2017-11-01T00:00:00", "vocab": "hurricanes", "tf-idf scores": 0.1574897611064825}, {"Date": "2017-11-01T00:00:00", "vocab": "labor", "tf-idf scores": 0.15732947014150572}, {"Date": "2017-11-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.1442164652185279}, {"Date": "2017-11-01T00:00:00", "vocab": "percent", "tf-idf scores": 0.14393989703288285}, {"Date": "2017-11-01T00:00:00", "vocab": "funds", "tf-idf scores": 0.12235560802673906}, {"Date": "2017-11-01T00:00:00", "vocab": "september", "tf-idf scores": 0.12235364310476478}, {"Date": "2017-11-01T00:00:00", "vocab": "remained", "tf-idf scores": 0.11360838339882602}, {"Date": "2017-09-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3022297062382572}, {"Date": "2017-09-20T00:00:00", "vocab": "market", "tf-idf scores": 0.20019925020745166}, {"Date": "2017-09-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.1844705266747592}, {"Date": "2017-09-20T00:00:00", "vocab": "hurricanes", "tf-idf scores": 0.18397701191782156}, {"Date": "2017-09-20T00:00:00", "vocab": "harvey", "tf-idf scores": 0.18046158947317847}, {"Date": "2017-09-20T00:00:00", "vocab": "july", "tf-idf scores": 0.17953403717310773}, {"Date": "2017-09-20T00:00:00", "vocab": "labor", "tf-idf scores": 0.12956313408074122}, {"Date": "2017-09-20T00:00:00", "vocab": "expected", "tf-idf scores": 0.12568418056617667}, {"Date": "2017-09-20T00:00:00", "vocab": "funds", "tf-idf scores": 0.12171884338032868}, {"Date": "2017-09-20T00:00:00", "vocab": "storms", "tf-idf scores": 0.10483804601853812}, {"Date": "2017-07-26T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3287554707536113}, {"Date": "2017-07-26T00:00:00", "vocab": "market", "tf-idf scores": 0.20665641060547923}, {"Date": "2017-07-26T00:00:00", "vocab": "financial", "tf-idf scores": 0.16580579445315888}, {"Date": "2017-07-26T00:00:00", "vocab": "june", "tf-idf scores": 0.1534267849832695}, {"Date": "2017-07-26T00:00:00", "vocab": "economic", "tf-idf scores": 0.15033041893097973}, {"Date": "2017-07-26T00:00:00", "vocab": "remained", "tf-idf scores": 0.1221666384805342}, {"Date": "2017-07-26T00:00:00", "vocab": "percent", "tf-idf scores": 0.1198684415140175}, {"Date": "2017-07-26T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11739028654909604}, {"Date": "2017-07-26T00:00:00", "vocab": "continued", "tf-idf scores": 0.1127467835106939}, {"Date": "2017-07-26T00:00:00", "vocab": "policy", "tf-idf scores": 0.11273114188633274}, {"Date": "2017-06-14T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3247497821137366}, {"Date": "2017-06-14T00:00:00", "vocab": "market", "tf-idf scores": 0.21651473228360119}, {"Date": "2017-06-14T00:00:00", "vocab": "economic", "tf-idf scores": 0.1775994890664961}, {"Date": "2017-06-14T00:00:00", "vocab": "percent", "tf-idf scores": 0.14728349374359137}, {"Date": "2017-06-14T00:00:00", "vocab": "policy", "tf-idf scores": 0.14720754076015927}, {"Date": "2017-06-14T00:00:00", "vocab": "funds", "tf-idf scores": 0.13858359748216378}, {"Date": "2017-06-14T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12992002427545823}, {"Date": "2017-06-14T00:00:00", "vocab": "continued", "tf-idf scores": 0.12990402623944644}, {"Date": "2017-06-14T00:00:00", "vocab": "labor", "tf-idf scores": 0.12994451870870902}, {"Date": "2017-06-14T00:00:00", "vocab": "normalization", "tf-idf scores": 0.12135542415564773}, {"Date": "2017-05-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.293499387808632}, {"Date": "2017-05-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.19117676346020268}, {"Date": "2017-05-03T00:00:00", "vocab": "march", "tf-idf scores": 0.17981293183512878}, {"Date": "2017-05-03T00:00:00", "vocab": "market", "tf-idf scores": 0.17345962328724482}, {"Date": "2017-05-03T00:00:00", "vocab": "growth", "tf-idf scores": 0.15181764923368612}, {"Date": "2017-05-03T00:00:00", "vocab": "continued", "tf-idf scores": 0.1467147181144826}, {"Date": "2017-05-03T00:00:00", "vocab": "percent", "tf-idf scores": 0.1465593674966349}, {"Date": "2017-05-03T00:00:00", "vocab": "remained", "tf-idf scores": 0.12455006959970527}, {"Date": "2017-05-03T00:00:00", "vocab": "real", "tf-idf scores": 0.11864400612837298}, {"Date": "2017-05-03T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11567247256080734}, {"Date": "2017-03-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.287583632546859}, {"Date": "2017-03-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.21169412372807364}, {"Date": "2017-03-15T00:00:00", "vocab": "market", "tf-idf scores": 0.18781461333913074}, {"Date": "2017-03-15T00:00:00", "vocab": "continued", "tf-idf scores": 0.16779244653911052}, {"Date": "2017-03-15T00:00:00", "vocab": "policy", "tf-idf scores": 0.15982832724520118}, {"Date": "2017-03-15T00:00:00", "vocab": "percent", "tf-idf scores": 0.15290480707953355}, {"Date": "2017-03-15T00:00:00", "vocab": "reinvestments", "tf-idf scores": 0.14645789885974117}, {"Date": "2017-03-15T00:00:00", "vocab": "recent", "tf-idf scores": 0.13186533033360917}, {"Date": "2017-03-15T00:00:00", "vocab": "labor", "tf-idf scores": 0.12781529836291758}, {"Date": "2017-03-15T00:00:00", "vocab": "funds", "tf-idf scores": 0.11983931243629213}, {"Date": "2017-02-01T00:00:00", "vocab": "selected", "tf-idf scores": 0.22417724107310952}, {"Date": "2017-02-01T00:00:00", "vocab": "foreign", "tf-idf scores": 0.2200243633406206}, {"Date": "2017-02-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1971215378825428}, {"Date": "2017-02-01T00:00:00", "vocab": "market", "tf-idf scores": 0.18285420316531875}, {"Date": "2017-02-01T00:00:00", "vocab": "currency", "tf-idf scores": 0.174681242866295}, {"Date": "2017-02-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.15997880536038447}, {"Date": "2017-02-01T00:00:00", "vocab": "bank", "tf-idf scores": 0.1520763572507829}, {"Date": "2017-02-01T00:00:00", "vocab": "paragraph", "tf-idf scores": 0.12469903368202268}, {"Date": "2017-02-01T00:00:00", "vocab": "eligible", "tf-idf scores": 0.12137057632241698}, {"Date": "2017-02-01T00:00:00", "vocab": "shall", "tf-idf scores": 0.11372324340603623}, {"Date": "2016-12-14T00:00:00", "vocab": "inflation", "tf-idf scores": 0.285416847638315}, {"Date": "2016-12-14T00:00:00", "vocab": "market", "tf-idf scores": 0.24155022754802388}, {"Date": "2016-12-14T00:00:00", "vocab": "economic", "tf-idf scores": 0.21954760595314354}, {"Date": "2016-12-14T00:00:00", "vocab": "labor", "tf-idf scores": 0.16690957335935644}, {"Date": "2016-12-14T00:00:00", "vocab": "percent", "tf-idf scores": 0.1540833669732144}, {"Date": "2016-12-14T00:00:00", "vocab": "prices", "tf-idf scores": 0.13232411857205387}, {"Date": "2016-12-14T00:00:00", "vocab": "policy", "tf-idf scores": 0.13176419210828766}, {"Date": "2016-12-14T00:00:00", "vocab": "funds", "tf-idf scores": 0.11862564426720681}, {"Date": "2016-12-14T00:00:00", "vocab": "conditions", "tf-idf scores": 0.10978682721116138}, {"Date": "2016-12-14T00:00:00", "vocab": "recent", "tf-idf scores": 0.10544168030210932}, {"Date": "2016-11-02T00:00:00", "vocab": "inflation", "tf-idf scores": 0.27401270171910036}, {"Date": "2016-11-02T00:00:00", "vocab": "market", "tf-idf scores": 0.18127465101761978}, {"Date": "2016-11-02T00:00:00", "vocab": "continued", "tf-idf scores": 0.17288895098164023}, {"Date": "2016-11-02T00:00:00", "vocab": "economic", "tf-idf scores": 0.16860074870749725}, {"Date": "2016-11-02T00:00:00", "vocab": "policy", "tf-idf scores": 0.16024845418905032}, {"Date": "2016-11-02T00:00:00", "vocab": "remained", "tf-idf scores": 0.13918387385270722}, {"Date": "2016-11-02T00:00:00", "vocab": "labor", "tf-idf scores": 0.1307550653640337}, {"Date": "2016-11-02T00:00:00", "vocab": "growth", "tf-idf scores": 0.12277171264790353}, {"Date": "2016-11-02T00:00:00", "vocab": "monetary", "tf-idf scores": 0.11384359987257343}, {"Date": "2016-11-02T00:00:00", "vocab": "implementation", "tf-idf scores": 0.11064413203367783}, {"Date": "2016-09-21T00:00:00", "vocab": "foreign", "tf-idf scores": 0.22602192099301283}, {"Date": "2016-09-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21472496859524087}, {"Date": "2016-09-21T00:00:00", "vocab": "currency", "tf-idf scores": 0.19086434022336116}, {"Date": "2016-09-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.18837319426732047}, {"Date": "2016-09-21T00:00:00", "vocab": "market", "tf-idf scores": 0.18834805854989645}, {"Date": "2016-09-21T00:00:00", "vocab": "continued", "tf-idf scores": 0.14697853690319407}, {"Date": "2016-09-21T00:00:00", "vocab": "labor", "tf-idf scores": 0.1356696558044067}, {"Date": "2016-09-21T00:00:00", "vocab": "policy", "tf-idf scores": 0.13187169192699671}, {"Date": "2016-09-21T00:00:00", "vocab": "recent", "tf-idf scores": 0.11681341957402304}, {"Date": "2016-09-21T00:00:00", "vocab": "standing", "tf-idf scores": 0.10637996501836273}, {"Date": "2016-07-27T00:00:00", "vocab": "market", "tf-idf scores": 0.2580725141074205}, {"Date": "2016-07-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.22977540545744524}, {"Date": "2016-07-27T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2297315206594353}, {"Date": "2016-07-27T00:00:00", "vocab": "brexit", "tf-idf scores": 0.19430623634398442}, {"Date": "2016-07-27T00:00:00", "vocab": "labor", "tf-idf scores": 0.18738992463968773}, {"Date": "2016-07-27T00:00:00", "vocab": "financial", "tf-idf scores": 0.16755140311415817}, {"Date": "2016-07-27T00:00:00", "vocab": "june", "tf-idf scores": 0.1594354885751396}, {"Date": "2016-07-27T00:00:00", "vocab": "prices", "tf-idf scores": 0.13494493968129564}, {"Date": "2016-07-27T00:00:00", "vocab": "policy", "tf-idf scores": 0.13076884299974542}, {"Date": "2016-07-27T00:00:00", "vocab": "continued", "tf-idf scores": 0.1237408874367091}, {"Date": "2016-06-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.26443957559888714}, {"Date": "2016-06-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.2602447758639667}, {"Date": "2016-06-15T00:00:00", "vocab": "market", "tf-idf scores": 0.2107499209875082}, {"Date": "2016-06-15T00:00:00", "vocab": "labor", "tf-idf scores": 0.2065881743311624}, {"Date": "2016-06-15T00:00:00", "vocab": "april", "tf-idf scores": 0.16572439068234476}, {"Date": "2016-06-15T00:00:00", "vocab": "continued", "tf-idf scores": 0.13636229036464784}, {"Date": "2016-06-15T00:00:00", "vocab": "referendum", "tf-idf scores": 0.12611351128914242}, {"Date": "2016-06-15T00:00:00", "vocab": "percent", "tf-idf scores": 0.1229687962473733}, {"Date": "2016-06-15T00:00:00", "vocab": "growth", "tf-idf scores": 0.12038019989122244}, {"Date": "2016-06-15T00:00:00", "vocab": "recent", "tf-idf scores": 0.11987364831017577}, {"Date": "2016-04-27T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22838760886962203}, {"Date": "2016-04-27T00:00:00", "vocab": "financial", "tf-idf scores": 0.2187181694631358}, {"Date": "2016-04-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.2090860082924087}, {"Date": "2016-04-27T00:00:00", "vocab": "market", "tf-idf scores": 0.19743596068953104}, {"Date": "2016-04-27T00:00:00", "vocab": "continued", "tf-idf scores": 0.17809864796554953}, {"Date": "2016-04-27T00:00:00", "vocab": "growth", "tf-idf scores": 0.1633010893691117}, {"Date": "2016-04-27T00:00:00", "vocab": "labor", "tf-idf scores": 0.15097590864526397}, {"Date": "2016-04-27T00:00:00", "vocab": "prices", "tf-idf scores": 0.1439012184888867}, {"Date": "2016-04-27T00:00:00", "vocab": "recent", "tf-idf scores": 0.14330706282972608}, {"Date": "2016-04-27T00:00:00", "vocab": "march", "tf-idf scores": 0.12396009117907257}, {"Date": "2016-03-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.27418952799849}, {"Date": "2016-03-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.24273249862716526}, {"Date": "2016-03-16T00:00:00", "vocab": "market", "tf-idf scores": 0.19782680680242826}, {"Date": "2016-03-16T00:00:00", "vocab": "labor", "tf-idf scores": 0.1528909352817943}, {"Date": "2016-03-16T00:00:00", "vocab": "recent", "tf-idf scores": 0.14830873808960462}, {"Date": "2016-03-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.13548754864870496}, {"Date": "2016-03-16T00:00:00", "vocab": "rrps", "tf-idf scores": 0.13341381311461098}, {"Date": "2016-03-16T00:00:00", "vocab": "financial", "tf-idf scores": 0.13148972448930354}, {"Date": "2016-03-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.12592306960943137}, {"Date": "2016-03-16T00:00:00", "vocab": "global", "tf-idf scores": 0.12521985142681588}, {"Date": "2016-01-27T00:00:00", "vocab": "foreign", "tf-idf scores": 0.24336609101275358}, {"Date": "2016-01-27T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23197398921672777}, {"Date": "2016-01-27T00:00:00", "vocab": "market", "tf-idf scores": 0.2061672393864317}, {"Date": "2016-01-27T00:00:00", "vocab": "shall", "tf-idf scores": 0.20518265379179357}, {"Date": "2016-01-27T00:00:00", "vocab": "currency", "tf-idf scores": 0.18509585695970512}, {"Date": "2016-01-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.1746787094552695}, {"Date": "2016-01-27T00:00:00", "vocab": "financial", "tf-idf scores": 0.12713171808338408}, {"Date": "2016-01-27T00:00:00", "vocab": "selected", "tf-idf scores": 0.12257970855657783}, {"Date": "2016-01-27T00:00:00", "vocab": "chairman", "tf-idf scores": 0.1216496297979054}, {"Date": "2016-01-27T00:00:00", "vocab": "eligible", "tf-idf scores": 0.12163680153203078}, {"Date": "2015-12-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3425581413990569}, {"Date": "2015-12-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.23645967875246932}, {"Date": "2015-12-16T00:00:00", "vocab": "market", "tf-idf scores": 0.21228340538259552}, {"Date": "2015-12-16T00:00:00", "vocab": "labor", "tf-idf scores": 0.1785799045883905}, {"Date": "2015-12-16T00:00:00", "vocab": "prices", "tf-idf scores": 0.17451179045148624}, {"Date": "2015-12-16T00:00:00", "vocab": "october", "tf-idf scores": 0.13558676425296873}, {"Date": "2015-12-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.1255040191024052}, {"Date": "2015-12-16T00:00:00", "vocab": "energy", "tf-idf scores": 0.11685647573547413}, {"Date": "2015-12-16T00:00:00", "vocab": "activity", "tf-idf scores": 0.11095740533393532}, {"Date": "2015-12-16T00:00:00", "vocab": "policy", "tf-idf scores": 0.10615298529647046}, {"Date": "2015-10-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2618906764036615}, {"Date": "2015-10-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.2264016115732469}, {"Date": "2015-10-28T00:00:00", "vocab": "market", "tf-idf scores": 0.20862228051671017}, {"Date": "2015-10-28T00:00:00", "vocab": "labor", "tf-idf scores": 0.20423050108516158}, {"Date": "2015-10-28T00:00:00", "vocab": "real", "tf-idf scores": 0.1421765713704538}, {"Date": "2015-10-28T00:00:00", "vocab": "policy", "tf-idf scores": 0.13315718789100142}, {"Date": "2015-10-28T00:00:00", "vocab": "financial", "tf-idf scores": 0.1298810458066442}, {"Date": "2015-10-28T00:00:00", "vocab": "continued", "tf-idf scores": 0.11987114552116732}, {"Date": "2015-10-28T00:00:00", "vocab": "prices", "tf-idf scores": 0.11147947434240009}, {"Date": "2015-10-28T00:00:00", "vocab": "range", "tf-idf scores": 0.1088842134504179}, {"Date": "2015-09-17T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3401513741804015}, {"Date": "2015-09-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.2857540614953714}, {"Date": "2015-09-17T00:00:00", "vocab": "market", "tf-idf scores": 0.2222863241966893}, {"Date": "2015-09-17T00:00:00", "vocab": "labor", "tf-idf scores": 0.16781631333338037}, {"Date": "2015-09-17T00:00:00", "vocab": "prices", "tf-idf scores": 0.1549179016694827}, {"Date": "2015-09-17T00:00:00", "vocab": "july", "tf-idf scores": 0.12970288154022414}, {"Date": "2015-09-17T00:00:00", "vocab": "activity", "tf-idf scores": 0.1224739574159336}, {"Date": "2015-09-17T00:00:00", "vocab": "policy", "tf-idf scores": 0.12245541041034927}, {"Date": "2015-09-17T00:00:00", "vocab": "reinvestments", "tf-idf scores": 0.11640870686251555}, {"Date": "2015-09-17T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11339866922582051}, {"Date": "2015-07-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.26119693954009343}, {"Date": "2015-07-29T00:00:00", "vocab": "market", "tf-idf scores": 0.23020613784163194}, {"Date": "2015-07-29T00:00:00", "vocab": "labor", "tf-idf scores": 0.20363431942385488}, {"Date": "2015-07-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.19034509582107303}, {"Date": "2015-07-29T00:00:00", "vocab": "reinvestments", "tf-idf scores": 0.16230123074295066}, {"Date": "2015-07-29T00:00:00", "vocab": "continued", "tf-idf scores": 0.14170802731255108}, {"Date": "2015-07-29T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12839126948782825}, {"Date": "2015-07-29T00:00:00", "vocab": "prices", "tf-idf scores": 0.12454702807682093}, {"Date": "2015-07-29T00:00:00", "vocab": "policy", "tf-idf scores": 0.11511406619830408}, {"Date": "2015-07-29T00:00:00", "vocab": "range", "tf-idf scores": 0.10368907772635469}, {"Date": "2015-06-17T00:00:00", "vocab": "inflation", "tf-idf scores": 0.25888829556939924}, {"Date": "2015-06-17T00:00:00", "vocab": "market", "tf-idf scores": 0.24966428270786964}, {"Date": "2015-06-17T00:00:00", "vocab": "labor", "tf-idf scores": 0.22184708059581018}, {"Date": "2015-06-17T00:00:00", "vocab": "real", "tf-idf scores": 0.1579355366369613}, {"Date": "2015-06-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.15717398826837678}, {"Date": "2015-06-17T00:00:00", "vocab": "policy", "tf-idf scores": 0.1525358739159798}, {"Date": "2015-06-17T00:00:00", "vocab": "april", "tf-idf scores": 0.15170847610993668}, {"Date": "2015-06-17T00:00:00", "vocab": "prices", "tf-idf scores": 0.13927604892925685}, {"Date": "2015-06-17T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1156217904448978}, {"Date": "2015-06-17T00:00:00", "vocab": "continued", "tf-idf scores": 0.11560180182043162}, {"Date": "2015-04-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.26122300742099347}, {"Date": "2015-04-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.2292369049705965}, {"Date": "2015-04-29T00:00:00", "vocab": "market", "tf-idf scores": 0.2246257280824019}, {"Date": "2015-04-29T00:00:00", "vocab": "prices", "tf-idf scores": 0.17031600167757668}, {"Date": "2015-04-29T00:00:00", "vocab": "labor", "tf-idf scores": 0.16501139353387567}, {"Date": "2015-04-29T00:00:00", "vocab": "growth", "tf-idf scores": 0.15658465092602772}, {"Date": "2015-04-29T00:00:00", "vocab": "remained", "tf-idf scores": 0.15128809455476075}, {"Date": "2015-04-29T00:00:00", "vocab": "policy", "tf-idf scores": 0.13293004837201908}, {"Date": "2015-04-29T00:00:00", "vocab": "real", "tf-idf scores": 0.13214657838900015}, {"Date": "2015-04-29T00:00:00", "vocab": "transitory", "tf-idf scores": 0.11818933613508004}, {"Date": "2015-03-18T00:00:00", "vocab": "rrp", "tf-idf scores": 0.2944633491117621}, {"Date": "2015-03-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22560138271541502}, {"Date": "2015-03-18T00:00:00", "vocab": "market", "tf-idf scores": 0.19213954490283083}, {"Date": "2015-03-18T00:00:00", "vocab": "term", "tf-idf scores": 0.14945754239767198}, {"Date": "2015-03-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.14619097092067893}, {"Date": "2015-03-18T00:00:00", "vocab": "labor", "tf-idf scores": 0.14619895758847304}, {"Date": "2015-03-18T00:00:00", "vocab": "policy", "tf-idf scores": 0.14624555941990375}, {"Date": "2015-03-18T00:00:00", "vocab": "january", "tf-idf scores": 0.12593918329545062}, {"Date": "2015-03-18T00:00:00", "vocab": "normalization", "tf-idf scores": 0.11700032011785311}, {"Date": "2015-03-18T00:00:00", "vocab": "liftoff", "tf-idf scores": 0.11331452775952366}, {"Date": "2015-01-28T00:00:00", "vocab": "rrp", "tf-idf scores": 0.20085160528774548}, {"Date": "2015-01-28T00:00:00", "vocab": "foreign", "tf-idf scores": 0.19534731276689465}, {"Date": "2015-01-28T00:00:00", "vocab": "shall", "tf-idf scores": 0.19267572693693935}, {"Date": "2015-01-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18528532319878416}, {"Date": "2015-01-28T00:00:00", "vocab": "market", "tf-idf scores": 0.18277274964730963}, {"Date": "2015-01-28T00:00:00", "vocab": "policy", "tf-idf scores": 0.15527184928547338}, {"Date": "2015-01-28T00:00:00", "vocab": "currency", "tf-idf scores": 0.15312975977104856}, {"Date": "2015-01-28T00:00:00", "vocab": "operations", "tf-idf scores": 0.1388154158725326}, {"Date": "2015-01-28T00:00:00", "vocab": "selected", "tf-idf scores": 0.11905609437665565}, {"Date": "2015-01-28T00:00:00", "vocab": "bank", "tf-idf scores": 0.11819601347255804}, {"Date": "2014-12-17T00:00:00", "vocab": "inflation", "tf-idf scores": 0.32019153557784286}, {"Date": "2014-12-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.21346926299447486}, {"Date": "2014-12-17T00:00:00", "vocab": "market", "tf-idf scores": 0.20376250063301324}, {"Date": "2014-12-17T00:00:00", "vocab": "prices", "tf-idf scores": 0.15597944482945497}, {"Date": "2014-12-17T00:00:00", "vocab": "labor", "tf-idf scores": 0.1552985343746649}, {"Date": "2014-12-17T00:00:00", "vocab": "october", "tf-idf scores": 0.15336902258465077}, {"Date": "2014-12-17T00:00:00", "vocab": "policy", "tf-idf scores": 0.14073240175500562}, {"Date": "2014-12-17T00:00:00", "vocab": "real", "tf-idf scores": 0.11913286074404075}, {"Date": "2014-12-17T00:00:00", "vocab": "continued", "tf-idf scores": 0.11647672124898102}, {"Date": "2014-12-17T00:00:00", "vocab": "oil", "tf-idf scores": 0.10367107848083164}, {"Date": "2014-10-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2670437014663354}, {"Date": "2014-10-29T00:00:00", "vocab": "market", "tf-idf scores": 0.20980193786931614}, {"Date": "2014-10-29T00:00:00", "vocab": "rrp", "tf-idf scores": 0.19707312406402414}, {"Date": "2014-10-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.1669592397648398}, {"Date": "2014-10-29T00:00:00", "vocab": "policy", "tf-idf scores": 0.12880257507573922}, {"Date": "2014-10-29T00:00:00", "vocab": "september", "tf-idf scores": 0.12611090192581517}, {"Date": "2014-10-29T00:00:00", "vocab": "continued", "tf-idf scores": 0.11925165490130124}, {"Date": "2014-10-29T00:00:00", "vocab": "financial", "tf-idf scores": 0.11550217325934102}, {"Date": "2014-10-29T00:00:00", "vocab": "labor", "tf-idf scores": 0.11449446101036213}, {"Date": "2014-10-29T00:00:00", "vocab": "preannounced", "tf-idf scores": 0.11082036024296625}, {"Date": "2014-09-17T00:00:00", "vocab": "inflation", "tf-idf scores": 0.25984578159801736}, {"Date": "2014-09-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.20307129920300368}, {"Date": "2014-09-17T00:00:00", "vocab": "market", "tf-idf scores": 0.1826989077712892}, {"Date": "2014-09-17T00:00:00", "vocab": "labor", "tf-idf scores": 0.16645186646834872}, {"Date": "2014-09-17T00:00:00", "vocab": "policy", "tf-idf scores": 0.14620380563875343}, {"Date": "2014-09-17T00:00:00", "vocab": "guidance", "tf-idf scores": 0.13439008046752143}, {"Date": "2014-09-17T00:00:00", "vocab": "funds", "tf-idf scores": 0.13402928386456162}, {"Date": "2014-09-17T00:00:00", "vocab": "july", "tf-idf scores": 0.12377816098968851}, {"Date": "2014-09-17T00:00:00", "vocab": "normalization", "tf-idf scores": 0.10340900729495583}, {"Date": "2014-09-17T00:00:00", "vocab": "continued", "tf-idf scores": 0.1015387576530299}, {"Date": "2014-07-30T00:00:00", "vocab": "market", "tf-idf scores": 0.2522484143402162}, {"Date": "2014-07-30T00:00:00", "vocab": "labor", "tf-idf scores": 0.23009111076745756}, {"Date": "2014-07-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22124868708708453}, {"Date": "2014-07-30T00:00:00", "vocab": "second", "tf-idf scores": 0.17734959670147626}, {"Date": "2014-07-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.15491993306891633}, {"Date": "2014-07-30T00:00:00", "vocab": "continued", "tf-idf scores": 0.1505312126003209}, {"Date": "2014-07-30T00:00:00", "vocab": "rrp", "tf-idf scores": 0.13986963004334027}, {"Date": "2014-07-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.11952208171758902}, {"Date": "2014-07-30T00:00:00", "vocab": "normalization", "tf-idf scores": 0.11271053802020736}, {"Date": "2014-07-30T00:00:00", "vocab": "remained", "tf-idf scores": 0.10619712793566716}, {"Date": "2014-06-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.25385901092733326}, {"Date": "2014-06-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.21010539102964249}, {"Date": "2014-06-18T00:00:00", "vocab": "policy", "tf-idf scores": 0.1969904519105206}, {"Date": "2014-06-18T00:00:00", "vocab": "market", "tf-idf scores": 0.18383884669626818}, {"Date": "2014-06-18T00:00:00", "vocab": "april", "tf-idf scores": 0.1436269510188937}, {"Date": "2014-06-18T00:00:00", "vocab": "rrp", "tf-idf scores": 0.13836023499531225}, {"Date": "2014-06-18T00:00:00", "vocab": "normalization", "tf-idf scores": 0.12255992824852384}, {"Date": "2014-06-18T00:00:00", "vocab": "labor", "tf-idf scores": 0.12257936027417196}, {"Date": "2014-06-18T00:00:00", "vocab": "real", "tf-idf scores": 0.1168222731895388}, {"Date": "2014-06-18T00:00:00", "vocab": "remained", "tf-idf scores": 0.11380015559281192}, {"Date": "2014-04-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23492826613866716}, {"Date": "2014-04-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.1997212470303339}, {"Date": "2014-04-30T00:00:00", "vocab": "market", "tf-idf scores": 0.19383499146298416}, {"Date": "2014-04-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.146853054012849}, {"Date": "2014-04-30T00:00:00", "vocab": "continued", "tf-idf scores": 0.12920215361227652}, {"Date": "2014-04-30T00:00:00", "vocab": "labor", "tf-idf scores": 0.12917324653202183}, {"Date": "2014-04-30T00:00:00", "vocab": "financial", "tf-idf scores": 0.12445415914325753}, {"Date": "2014-04-30T00:00:00", "vocab": "real", "tf-idf scores": 0.11913757754101952}, {"Date": "2014-04-30T00:00:00", "vocab": "remained", "tf-idf scores": 0.11743969576664243}, {"Date": "2014-04-30T00:00:00", "vocab": "unemployment", "tf-idf scores": 0.11457123280273328}, {"Date": "2014-03-19T00:00:00", "vocab": "economic", "tf-idf scores": 0.2378639174851015}, {"Date": "2014-03-19T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2141007469463816}, {"Date": "2014-03-19T00:00:00", "vocab": "growth", "tf-idf scores": 0.18638697118569617}, {"Date": "2014-03-19T00:00:00", "vocab": "market", "tf-idf scores": 0.18082983665924315}, {"Date": "2014-03-19T00:00:00", "vocab": "policy", "tf-idf scores": 0.1380387855594669}, {"Date": "2014-03-19T00:00:00", "vocab": "unemployment", "tf-idf scores": 0.1367590453654081}, {"Date": "2014-03-19T00:00:00", "vocab": "labor", "tf-idf scores": 0.13321629325508422}, {"Date": "2014-03-19T00:00:00", "vocab": "pace", "tf-idf scores": 0.12847252157255784}, {"Date": "2014-03-19T00:00:00", "vocab": "winter", "tf-idf scores": 0.1273293411317849}, {"Date": "2014-03-19T00:00:00", "vocab": "january", "tf-idf scores": 0.12078687986403085}, {"Date": "2014-03-04T00:00:00", "vocab": "economic", "tf-idf scores": 0.2378410913900952}, {"Date": "2014-03-04T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2140602735388086}, {"Date": "2014-03-04T00:00:00", "vocab": "growth", "tf-idf scores": 0.18632749526860123}, {"Date": "2014-03-04T00:00:00", "vocab": "market", "tf-idf scores": 0.18084542903564219}, {"Date": "2014-03-04T00:00:00", "vocab": "policy", "tf-idf scores": 0.1379951735403874}, {"Date": "2014-03-04T00:00:00", "vocab": "unemployment", "tf-idf scores": 0.136773478300223}, {"Date": "2014-03-04T00:00:00", "vocab": "labor", "tf-idf scores": 0.133225327385756}, {"Date": "2014-03-04T00:00:00", "vocab": "pace", "tf-idf scores": 0.1284597172208973}, {"Date": "2014-03-04T00:00:00", "vocab": "winter", "tf-idf scores": 0.1272768091437335}, {"Date": "2014-03-04T00:00:00", "vocab": "january", "tf-idf scores": 0.12073070422323447}, {"Date": "2014-01-29T00:00:00", "vocab": "market", "tf-idf scores": 0.28622618923898224}, {"Date": "2014-01-29T00:00:00", "vocab": "shall", "tf-idf scores": 0.2430319118604554}, {"Date": "2014-01-29T00:00:00", "vocab": "foreign", "tf-idf scores": 0.2061226378972721}, {"Date": "2014-01-29T00:00:00", "vocab": "currency", "tf-idf scores": 0.18003663862277136}, {"Date": "2014-01-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15167831284407965}, {"Date": "2014-01-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.1373934938561043}, {"Date": "2014-01-29T00:00:00", "vocab": "open", "tf-idf scores": 0.13510622531373337}, {"Date": "2014-01-29T00:00:00", "vocab": "bank", "tf-idf scores": 0.12937988973596293}, {"Date": "2014-01-29T00:00:00", "vocab": "securities", "tf-idf scores": 0.1228765107551834}, {"Date": "2014-01-29T00:00:00", "vocab": "policy", "tf-idf scores": 0.12023879648032347}, {"Date": "2013-12-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22230494860190578}, {"Date": "2013-12-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.20206357890627563}, {"Date": "2013-12-18T00:00:00", "vocab": "market", "tf-idf scores": 0.1778465499581087}, {"Date": "2013-12-18T00:00:00", "vocab": "financial", "tf-idf scores": 0.1549117340254292}, {"Date": "2013-12-18T00:00:00", "vocab": "asset", "tf-idf scores": 0.15208233489137593}, {"Date": "2013-12-18T00:00:00", "vocab": "labor", "tf-idf scores": 0.14550709150579985}, {"Date": "2013-12-18T00:00:00", "vocab": "threshold", "tf-idf scores": 0.14119731566192995}, {"Date": "2013-12-18T00:00:00", "vocab": "marginal", "tf-idf scores": 0.14096639680083678}, {"Date": "2013-12-18T00:00:00", "vocab": "purchases", "tf-idf scores": 0.13902757868352328}, {"Date": "2013-12-18T00:00:00", "vocab": "policy", "tf-idf scores": 0.13747384550471245}, {"Date": "2013-10-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.21078539159725906}, {"Date": "2013-10-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.18565197329334743}, {"Date": "2013-10-30T00:00:00", "vocab": "market", "tf-idf scores": 0.17062099134609415}, {"Date": "2013-10-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15053206689655457}, {"Date": "2013-10-30T00:00:00", "vocab": "asset", "tf-idf scores": 0.13803898283901767}, {"Date": "2013-10-30T00:00:00", "vocab": "labor", "tf-idf scores": 0.13548735384653893}, {"Date": "2013-10-30T00:00:00", "vocab": "pace", "tf-idf scores": 0.13052690272844203}, {"Date": "2013-10-30T00:00:00", "vocab": "september", "tf-idf scores": 0.12492892953830945}, {"Date": "2013-10-30T00:00:00", "vocab": "continued", "tf-idf scores": 0.11547305124693294}, {"Date": "2013-10-30T00:00:00", "vocab": "funds", "tf-idf scores": 0.11546386408084212}, {"Date": "2013-10-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.21079269892841782}, {"Date": "2013-10-16T00:00:00", "vocab": "policy", "tf-idf scores": 0.18570775185964455}, {"Date": "2013-10-16T00:00:00", "vocab": "market", "tf-idf scores": 0.1705889084861131}, {"Date": "2013-10-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15050891886823609}, {"Date": "2013-10-16T00:00:00", "vocab": "asset", "tf-idf scores": 0.13800663648823705}, {"Date": "2013-10-16T00:00:00", "vocab": "labor", "tf-idf scores": 0.13545678397509517}, {"Date": "2013-10-16T00:00:00", "vocab": "pace", "tf-idf scores": 0.13046387261786824}, {"Date": "2013-10-16T00:00:00", "vocab": "september", "tf-idf scores": 0.12487058526534123}, {"Date": "2013-10-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.1153872929776516}, {"Date": "2013-10-16T00:00:00", "vocab": "funds", "tf-idf scores": 0.11543614954384333}, {"Date": "2013-09-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.21859650712367282}, {"Date": "2013-09-18T00:00:00", "vocab": "asset", "tf-idf scores": 0.1807977878152076}, {"Date": "2013-09-18T00:00:00", "vocab": "market", "tf-idf scores": 0.17841873280448356}, {"Date": "2013-09-18T00:00:00", "vocab": "july", "tf-idf scores": 0.1700269094910751}, {"Date": "2013-09-18T00:00:00", "vocab": "pace", "tf-idf scores": 0.1561274464785427}, {"Date": "2013-09-18T00:00:00", "vocab": "purchases", "tf-idf scores": 0.15342710865637119}, {"Date": "2013-09-18T00:00:00", "vocab": "financial", "tf-idf scores": 0.15296639558344763}, {"Date": "2013-09-18T00:00:00", "vocab": "policy", "tf-idf scores": 0.15165725658362653}, {"Date": "2013-09-18T00:00:00", "vocab": "conditions", "tf-idf scores": 0.14275256762723423}, {"Date": "2013-09-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13827085699182617}, {"Date": "2013-07-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22935983347951125}, {"Date": "2013-07-31T00:00:00", "vocab": "policy", "tf-idf scores": 0.21406219797740217}, {"Date": "2013-07-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.20386155314455356}, {"Date": "2013-07-31T00:00:00", "vocab": "market", "tf-idf scores": 0.16822027902378475}, {"Date": "2013-07-31T00:00:00", "vocab": "pace", "tf-idf scores": 0.12748093705883165}, {"Date": "2013-07-31T00:00:00", "vocab": "contingent", "tf-idf scores": 0.12663778590871555}, {"Date": "2013-07-31T00:00:00", "vocab": "recent", "tf-idf scores": 0.12231550507267087}, {"Date": "2013-07-31T00:00:00", "vocab": "june", "tf-idf scores": 0.11892718671512285}, {"Date": "2013-07-31T00:00:00", "vocab": "asset", "tf-idf scores": 0.11811693206997792}, {"Date": "2013-07-31T00:00:00", "vocab": "second", "tf-idf scores": 0.116717359591211}, {"Date": "2013-06-19T00:00:00", "vocab": "economic", "tf-idf scores": 0.26532950167071834}, {"Date": "2013-06-19T00:00:00", "vocab": "market", "tf-idf scores": 0.1966889895657927}, {"Date": "2013-06-19T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18296031459825407}, {"Date": "2013-06-19T00:00:00", "vocab": "asset", "tf-idf scores": 0.1787682016325241}, {"Date": "2013-06-19T00:00:00", "vocab": "labor", "tf-idf scores": 0.1418760570234028}, {"Date": "2013-06-19T00:00:00", "vocab": "policy", "tf-idf scores": 0.13730493311290778}, {"Date": "2013-06-19T00:00:00", "vocab": "recent", "tf-idf scores": 0.12358470469879594}, {"Date": "2013-06-19T00:00:00", "vocab": "april", "tf-idf scores": 0.1167942715539615}, {"Date": "2013-06-19T00:00:00", "vocab": "purchases", "tf-idf scores": 0.11390697040595313}, {"Date": "2013-06-19T00:00:00", "vocab": "activity", "tf-idf scores": 0.10976892075142274}, {"Date": "2013-05-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.20419178158483775}, {"Date": "2013-05-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.19258361628979004}, {"Date": "2013-05-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.1575754960583246}, {"Date": "2013-05-01T00:00:00", "vocab": "market", "tf-idf scores": 0.15170736646436325}, {"Date": "2013-05-01T00:00:00", "vocab": "pace", "tf-idf scores": 0.15173606359135913}, {"Date": "2013-05-01T00:00:00", "vocab": "policy", "tf-idf scores": 0.14585891790461858}, {"Date": "2013-05-01T00:00:00", "vocab": "purchases", "tf-idf scores": 0.1314358511188605}, {"Date": "2013-05-01T00:00:00", "vocab": "march", "tf-idf scores": 0.12782469531176205}, {"Date": "2013-05-01T00:00:00", "vocab": "asset", "tf-idf scores": 0.11830763031037735}, {"Date": "2013-05-01T00:00:00", "vocab": "recent", "tf-idf scores": 0.11667140067491714}, {"Date": "2013-03-20T00:00:00", "vocab": "purchases", "tf-idf scores": 0.23324108201337115}, {"Date": "2013-03-20T00:00:00", "vocab": "asset", "tf-idf scores": 0.21353342024860542}, {"Date": "2013-03-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.19667261889059637}, {"Date": "2013-03-20T00:00:00", "vocab": "financial", "tf-idf scores": 0.16370168974700355}, {"Date": "2013-03-20T00:00:00", "vocab": "policy", "tf-idf scores": 0.15737567349154055}, {"Date": "2013-03-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1475232904144688}, {"Date": "2013-03-20T00:00:00", "vocab": "market", "tf-idf scores": 0.13770907316585798}, {"Date": "2013-03-20T00:00:00", "vocab": "monetary", "tf-idf scores": 0.11308510001425177}, {"Date": "2013-03-20T00:00:00", "vocab": "pace", "tf-idf scores": 0.11311825079578897}, {"Date": "2013-03-20T00:00:00", "vocab": "january", "tf-idf scores": 0.10924335332145879}, {"Date": "2013-01-30T00:00:00", "vocab": "shall", "tf-idf scores": 0.26613554986859206}, {"Date": "2013-01-30T00:00:00", "vocab": "market", "tf-idf scores": 0.19726765854664857}, {"Date": "2013-01-30T00:00:00", "vocab": "foreign", "tf-idf scores": 0.19055664673314812}, {"Date": "2013-01-30T00:00:00", "vocab": "open", "tf-idf scores": 0.14780300460812654}, {"Date": "2013-01-30T00:00:00", "vocab": "securities", "tf-idf scores": 0.1435515282797216}, {"Date": "2013-01-30T00:00:00", "vocab": "currency", "tf-idf scores": 0.1343058983494411}, {"Date": "2013-01-30T00:00:00", "vocab": "bank", "tf-idf scores": 0.13433955329214914}, {"Date": "2013-01-30T00:00:00", "vocab": "fourth", "tf-idf scores": 0.13426181579980687}, {"Date": "2013-01-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.1304452854527899}, {"Date": "2013-01-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.12369814073476505}, {"Date": "2012-12-12T00:00:00", "vocab": "thresholds", "tf-idf scores": 0.2360769627415479}, {"Date": "2012-12-12T00:00:00", "vocab": "economic", "tf-idf scores": 0.21394643538663502}, {"Date": "2012-12-12T00:00:00", "vocab": "inflation", "tf-idf scores": 0.16637978308671858}, {"Date": "2012-12-12T00:00:00", "vocab": "purchases", "tf-idf scores": 0.15222749368939145}, {"Date": "2012-12-12T00:00:00", "vocab": "financial", "tf-idf scores": 0.14386653167382957}, {"Date": "2012-12-12T00:00:00", "vocab": "october", "tf-idf scores": 0.13359073349719255}, {"Date": "2012-12-12T00:00:00", "vocab": "market", "tf-idf scores": 0.13314961383860782}, {"Date": "2012-12-12T00:00:00", "vocab": "policy", "tf-idf scores": 0.13315917822402892}, {"Date": "2012-12-12T00:00:00", "vocab": "remained", "tf-idf scores": 0.13309222118952152}, {"Date": "2012-12-12T00:00:00", "vocab": "securities", "tf-idf scores": 0.12893310504933092}, {"Date": "2012-10-24T00:00:00", "vocab": "thresholds", "tf-idf scores": 0.24723454057374084}, {"Date": "2012-10-24T00:00:00", "vocab": "economic", "tf-idf scores": 0.20364596458807616}, {"Date": "2012-10-24T00:00:00", "vocab": "september", "tf-idf scores": 0.16673001097028398}, {"Date": "2012-10-24T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15541492768227494}, {"Date": "2012-10-24T00:00:00", "vocab": "policy", "tf-idf scores": 0.15543964427011825}, {"Date": "2012-10-24T00:00:00", "vocab": "financial", "tf-idf scores": 0.13521928930633126}, {"Date": "2012-10-24T00:00:00", "vocab": "recent", "tf-idf scores": 0.12867326372004906}, {"Date": "2012-10-24T00:00:00", "vocab": "remained", "tf-idf scores": 0.12326895214777325}, {"Date": "2012-10-24T00:00:00", "vocab": "quantitative", "tf-idf scores": 0.12219827706356316}, {"Date": "2012-10-24T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11790444524493061}, {"Date": "2012-09-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.23700230839881453}, {"Date": "2012-09-13T00:00:00", "vocab": "policy", "tf-idf scores": 0.1827148989632465}, {"Date": "2012-09-13T00:00:00", "vocab": "purchases", "tf-idf scores": 0.15815894963345828}, {"Date": "2012-09-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15313423926660616}, {"Date": "2012-09-13T00:00:00", "vocab": "august", "tf-idf scores": 0.148899637782916}, {"Date": "2012-09-13T00:00:00", "vocab": "continued", "tf-idf scores": 0.14323368108700038}, {"Date": "2012-09-13T00:00:00", "vocab": "financial", "tf-idf scores": 0.13948109866404904}, {"Date": "2012-09-13T00:00:00", "vocab": "pace", "tf-idf scores": 0.11857593902416352}, {"Date": "2012-09-13T00:00:00", "vocab": "additional", "tf-idf scores": 0.11825344008997206}, {"Date": "2012-09-13T00:00:00", "vocab": "agency", "tf-idf scores": 0.11125504075300989}, {"Date": "2012-08-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.27155890322520093}, {"Date": "2012-08-01T00:00:00", "vocab": "rules", "tf-idf scores": 0.18247960686165277}, {"Date": "2012-08-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1723620655919817}, {"Date": "2012-08-01T00:00:00", "vocab": "policy", "tf-idf scores": 0.16711221717791505}, {"Date": "2012-08-01T00:00:00", "vocab": "second", "tf-idf scores": 0.14350660816976935}, {"Date": "2012-08-01T00:00:00", "vocab": "prices", "tf-idf scores": 0.12596566905744344}, {"Date": "2012-08-01T00:00:00", "vocab": "june", "tf-idf scores": 0.12190380662778336}, {"Date": "2012-08-01T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12012173228311357}, {"Date": "2012-08-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.12011495348899479}, {"Date": "2012-08-01T00:00:00", "vocab": "recent", "tf-idf scores": 0.12019766856460393}, {"Date": "2012-06-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.24151758948197732}, {"Date": "2012-06-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.20771381676419765}, {"Date": "2012-06-20T00:00:00", "vocab": "april", "tf-idf scores": 0.20251909146828667}, {"Date": "2012-06-20T00:00:00", "vocab": "policy", "tf-idf scores": 0.16426787673514034}, {"Date": "2012-06-20T00:00:00", "vocab": "securities", "tf-idf scores": 0.13642176998104616}, {"Date": "2012-06-20T00:00:00", "vocab": "growth", "tf-idf scores": 0.13588248814943224}, {"Date": "2012-06-20T00:00:00", "vocab": "recent", "tf-idf scores": 0.13039998366327665}, {"Date": "2012-06-20T00:00:00", "vocab": "prices", "tf-idf scores": 0.12130783719834744}, {"Date": "2012-06-20T00:00:00", "vocab": "continued", "tf-idf scores": 0.11592161181342037}, {"Date": "2012-06-20T00:00:00", "vocab": "treasury", "tf-idf scores": 0.10359777468599322}, {"Date": "2012-04-25T00:00:00", "vocab": "economic", "tf-idf scores": 0.26675993397277603}, {"Date": "2012-04-25T00:00:00", "vocab": "inflation", "tf-idf scores": 0.17260128885572049}, {"Date": "2012-04-25T00:00:00", "vocab": "recent", "tf-idf scores": 0.15172160543583346}, {"Date": "2012-04-25T00:00:00", "vocab": "continued", "tf-idf scores": 0.14651743930203567}, {"Date": "2012-04-25T00:00:00", "vocab": "policy", "tf-idf scores": 0.14647040726811478}, {"Date": "2012-04-25T00:00:00", "vocab": "rules", "tf-idf scores": 0.13715286182141634}, {"Date": "2012-04-25T00:00:00", "vocab": "march", "tf-idf scores": 0.13219934311654702}, {"Date": "2012-04-25T00:00:00", "vocab": "prices", "tf-idf scores": 0.13134561362925995}, {"Date": "2012-04-25T00:00:00", "vocab": "unemployment", "tf-idf scores": 0.12355093118781406}, {"Date": "2012-04-25T00:00:00", "vocab": "monetary", "tf-idf scores": 0.12032805254180694}, {"Date": "2012-03-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21856273402342596}, {"Date": "2012-03-13T00:00:00", "vocab": "recent", "tf-idf scores": 0.21274887570442005}, {"Date": "2012-03-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.20702994130067806}, {"Date": "2012-03-13T00:00:00", "vocab": "continued", "tf-idf scores": 0.1898199728724206}, {"Date": "2012-03-13T00:00:00", "vocab": "january", "tf-idf scores": 0.18243305153006575}, {"Date": "2012-03-13T00:00:00", "vocab": "remained", "tf-idf scores": 0.15532973115256646}, {"Date": "2012-03-13T00:00:00", "vocab": "gasoline", "tf-idf scores": 0.1394687413924586}, {"Date": "2012-03-13T00:00:00", "vocab": "prices", "tf-idf scores": 0.13284290878585836}, {"Date": "2012-03-13T00:00:00", "vocab": "conditions", "tf-idf scores": 0.13234446834771782}, {"Date": "2012-03-13T00:00:00", "vocab": "market", "tf-idf scores": 0.12651110698089035}, {"Date": "2012-01-25T00:00:00", "vocab": "shall", "tf-idf scores": 0.2136754929107749}, {"Date": "2012-01-25T00:00:00", "vocab": "market", "tf-idf scores": 0.20302875484095162}, {"Date": "2012-01-25T00:00:00", "vocab": "foreign", "tf-idf scores": 0.18203409066966295}, {"Date": "2012-01-25T00:00:00", "vocab": "economic", "tf-idf scores": 0.1540172252533708}, {"Date": "2012-01-25T00:00:00", "vocab": "open", "tf-idf scores": 0.13715430964880457}, {"Date": "2012-01-25T00:00:00", "vocab": "currency", "tf-idf scores": 0.13451624791297287}, {"Date": "2012-01-25T00:00:00", "vocab": "bank", "tf-idf scores": 0.1336082492596476}, {"Date": "2012-01-25T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13308217146012377}, {"Date": "2012-01-25T00:00:00", "vocab": "securities", "tf-idf scores": 0.1266025300673226}, {"Date": "2012-01-25T00:00:00", "vocab": "recent", "tf-idf scores": 0.12601558426187692}, {"Date": "2011-12-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.2215684468061553}, {"Date": "2011-12-13T00:00:00", "vocab": "swap", "tf-idf scores": 0.15477686527232207}, {"Date": "2011-12-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14143204290490452}, {"Date": "2011-12-13T00:00:00", "vocab": "market", "tf-idf scores": 0.14142739889205216}, {"Date": "2011-12-13T00:00:00", "vocab": "european", "tf-idf scores": 0.13874965840123013}, {"Date": "2011-12-13T00:00:00", "vocab": "financial", "tf-idf scores": 0.1284159368603356}, {"Date": "2011-12-13T00:00:00", "vocab": "november", "tf-idf scores": 0.1255008159899755}, {"Date": "2011-12-13T00:00:00", "vocab": "remained", "tf-idf scores": 0.12261154556814537}, {"Date": "2011-12-13T00:00:00", "vocab": "policy", "tf-idf scores": 0.1179512380567166}, {"Date": "2011-12-13T00:00:00", "vocab": "recent", "tf-idf scores": 0.11790159054300588}, {"Date": "2011-11-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.2216139132370269}, {"Date": "2011-11-28T00:00:00", "vocab": "swap", "tf-idf scores": 0.15484581225142177}, {"Date": "2011-11-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14144191346855298}, {"Date": "2011-11-28T00:00:00", "vocab": "market", "tf-idf scores": 0.14142544494752354}, {"Date": "2011-11-28T00:00:00", "vocab": "european", "tf-idf scores": 0.13869837084789866}, {"Date": "2011-11-28T00:00:00", "vocab": "financial", "tf-idf scores": 0.12841543471854347}, {"Date": "2011-11-28T00:00:00", "vocab": "november", "tf-idf scores": 0.12549476683185662}, {"Date": "2011-11-28T00:00:00", "vocab": "remained", "tf-idf scores": 0.12264419311786702}, {"Date": "2011-11-28T00:00:00", "vocab": "policy", "tf-idf scores": 0.11791925767182344}, {"Date": "2011-11-28T00:00:00", "vocab": "recent", "tf-idf scores": 0.1178753131911741}, {"Date": "2011-11-02T00:00:00", "vocab": "economic", "tf-idf scores": 0.20380600696866638}, {"Date": "2011-11-02T00:00:00", "vocab": "inflation", "tf-idf scores": 0.19362610081770565}, {"Date": "2011-11-02T00:00:00", "vocab": "september", "tf-idf scores": 0.15055245361034397}, {"Date": "2011-11-02T00:00:00", "vocab": "remained", "tf-idf scores": 0.14772537407735342}, {"Date": "2011-11-02T00:00:00", "vocab": "securities", "tf-idf scores": 0.13812536121969832}, {"Date": "2011-11-02T00:00:00", "vocab": "continued", "tf-idf scores": 0.13755151679672345}, {"Date": "2011-11-02T00:00:00", "vocab": "policy", "tf-idf scores": 0.13243893093716724}, {"Date": "2011-11-02T00:00:00", "vocab": "prices", "tf-idf scores": 0.1228448529000848}, {"Date": "2011-11-02T00:00:00", "vocab": "growth", "tf-idf scores": 0.117755825553263}, {"Date": "2011-11-02T00:00:00", "vocab": "financial", "tf-idf scores": 0.1131336779161292}, {"Date": "2011-09-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.20331342686814038}, {"Date": "2011-09-21T00:00:00", "vocab": "securities", "tf-idf scores": 0.17871629832722724}, {"Date": "2011-09-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15814172195656173}, {"Date": "2011-09-21T00:00:00", "vocab": "ior", "tf-idf scores": 0.15578363313133017}, {"Date": "2011-09-21T00:00:00", "vocab": "august", "tf-idf scores": 0.13622608380300094}, {"Date": "2011-09-21T00:00:00", "vocab": "policy", "tf-idf scores": 0.13103249896880784}, {"Date": "2011-09-21T00:00:00", "vocab": "treasury", "tf-idf scores": 0.1308264468326308}, {"Date": "2011-09-21T00:00:00", "vocab": "financial", "tf-idf scores": 0.12309116807899054}, {"Date": "2011-09-21T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12201772823868005}, {"Date": "2011-09-21T00:00:00", "vocab": "maturity", "tf-idf scores": 0.12084221199847925}, {"Date": "2011-08-09T00:00:00", "vocab": "economic", "tf-idf scores": 0.2552333121749093}, {"Date": "2011-08-09T00:00:00", "vocab": "inflation", "tf-idf scores": 0.17430247433542767}, {"Date": "2011-08-09T00:00:00", "vocab": "prices", "tf-idf scores": 0.16261382609787875}, {"Date": "2011-08-09T00:00:00", "vocab": "recent", "tf-idf scores": 0.15562096262214606}, {"Date": "2011-08-09T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13077824107087171}, {"Date": "2011-08-09T00:00:00", "vocab": "second", "tf-idf scores": 0.12836551914325325}, {"Date": "2011-08-09T00:00:00", "vocab": "remained", "tf-idf scores": 0.12453738694134954}, {"Date": "2011-08-09T00:00:00", "vocab": "market", "tf-idf scores": 0.11835350381076631}, {"Date": "2011-08-09T00:00:00", "vocab": "real", "tf-idf scores": 0.11298617453069007}, {"Date": "2011-08-09T00:00:00", "vocab": "continued", "tf-idf scores": 0.11206641909681377}, {"Date": "2011-08-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.25529985613984923}, {"Date": "2011-08-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.17436821005015243}, {"Date": "2011-08-01T00:00:00", "vocab": "prices", "tf-idf scores": 0.16257574277526954}, {"Date": "2011-08-01T00:00:00", "vocab": "recent", "tf-idf scores": 0.15570765844732978}, {"Date": "2011-08-01T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13081690709622393}, {"Date": "2011-08-01T00:00:00", "vocab": "second", "tf-idf scores": 0.1282833136747179}, {"Date": "2011-08-01T00:00:00", "vocab": "remained", "tf-idf scores": 0.12457283810230653}, {"Date": "2011-08-01T00:00:00", "vocab": "market", "tf-idf scores": 0.11836828136452626}, {"Date": "2011-08-01T00:00:00", "vocab": "real", "tf-idf scores": 0.11299392184542456}, {"Date": "2011-08-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.11207988493102065}, {"Date": "2011-06-22T00:00:00", "vocab": "inflation", "tf-idf scores": 0.24355189870829982}, {"Date": "2011-06-22T00:00:00", "vocab": "april", "tf-idf scores": 0.23903933367725919}, {"Date": "2011-06-22T00:00:00", "vocab": "economic", "tf-idf scores": 0.19672104067293733}, {"Date": "2011-06-22T00:00:00", "vocab": "dsge", "tf-idf scores": 0.18831070278960055}, {"Date": "2011-06-22T00:00:00", "vocab": "prices", "tf-idf scores": 0.14586794675557896}, {"Date": "2011-06-22T00:00:00", "vocab": "recent", "tf-idf scores": 0.13583826752112813}, {"Date": "2011-06-22T00:00:00", "vocab": "pace", "tf-idf scores": 0.13117893640348352}, {"Date": "2011-06-22T00:00:00", "vocab": "remained", "tf-idf scores": 0.13111999886226738}, {"Date": "2011-06-22T00:00:00", "vocab": "market", "tf-idf scores": 0.12644777117773037}, {"Date": "2011-06-22T00:00:00", "vocab": "models", "tf-idf scores": 0.12151245069693882}, {"Date": "2011-04-27T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2852749661352559}, {"Date": "2011-04-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.19608189137775547}, {"Date": "2011-04-27T00:00:00", "vocab": "securities", "tf-idf scores": 0.14098153232354843}, {"Date": "2011-04-27T00:00:00", "vocab": "continued", "tf-idf scores": 0.13823564808180927}, {"Date": "2011-04-27T00:00:00", "vocab": "policy", "tf-idf scores": 0.13817466961982225}, {"Date": "2011-04-27T00:00:00", "vocab": "remained", "tf-idf scores": 0.13824157548763877}, {"Date": "2011-04-27T00:00:00", "vocab": "february", "tf-idf scores": 0.12667802660305372}, {"Date": "2011-04-27T00:00:00", "vocab": "prices", "tf-idf scores": 0.12539616415645824}, {"Date": "2011-04-27T00:00:00", "vocab": "pace", "tf-idf scores": 0.12041569453920323}, {"Date": "2011-04-27T00:00:00", "vocab": "commodity", "tf-idf scores": 0.11622421185213487}, {"Date": "2011-03-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.30197451769637434}, {"Date": "2011-03-15T00:00:00", "vocab": "january", "tf-idf scores": 0.20403322606431123}, {"Date": "2011-03-15T00:00:00", "vocab": "prices", "tf-idf scores": 0.1741571034833683}, {"Date": "2011-03-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.1678101454539437}, {"Date": "2011-03-15T00:00:00", "vocab": "remained", "tf-idf scores": 0.14539854453076836}, {"Date": "2011-03-15T00:00:00", "vocab": "market", "tf-idf scores": 0.1397765219629633}, {"Date": "2011-03-15T00:00:00", "vocab": "labor", "tf-idf scores": 0.12300859879859108}, {"Date": "2011-03-15T00:00:00", "vocab": "recent", "tf-idf scores": 0.11748313185252868}, {"Date": "2011-03-15T00:00:00", "vocab": "fourth", "tf-idf scores": 0.11227420424303537}, {"Date": "2011-03-15T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11184843886545771}, {"Date": "2011-01-26T00:00:00", "vocab": "shall", "tf-idf scores": 0.21714879419211397}, {"Date": "2011-01-26T00:00:00", "vocab": "foreign", "tf-idf scores": 0.216986843126481}, {"Date": "2011-01-26T00:00:00", "vocab": "market", "tf-idf scores": 0.19565426963606605}, {"Date": "2011-01-26T00:00:00", "vocab": "currency", "tf-idf scores": 0.1801782743903711}, {"Date": "2011-01-26T00:00:00", "vocab": "economic", "tf-idf scores": 0.1600452016095036}, {"Date": "2011-01-26T00:00:00", "vocab": "securities", "tf-idf scores": 0.14874454809941756}, {"Date": "2011-01-26T00:00:00", "vocab": "open", "tf-idf scores": 0.1393177177921797}, {"Date": "2011-01-26T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1280698597451148}, {"Date": "2011-01-26T00:00:00", "vocab": "remained", "tf-idf scores": 0.12099491978261366}, {"Date": "2011-01-26T00:00:00", "vocab": "bank", "tf-idf scores": 0.11790079130552919}, {"Date": "2010-12-14T00:00:00", "vocab": "november", "tf-idf scores": 0.24210439758115287}, {"Date": "2010-12-14T00:00:00", "vocab": "continued", "tf-idf scores": 0.17206542937402722}, {"Date": "2010-12-14T00:00:00", "vocab": "economic", "tf-idf scores": 0.1720805275339246}, {"Date": "2010-12-14T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14835649706565968}, {"Date": "2010-12-14T00:00:00", "vocab": "remained", "tf-idf scores": 0.14828573322724095}, {"Date": "2010-12-14T00:00:00", "vocab": "october", "tf-idf scores": 0.13544922721889907}, {"Date": "2010-12-14T00:00:00", "vocab": "market", "tf-idf scores": 0.13047905853929925}, {"Date": "2010-12-14T00:00:00", "vocab": "securities", "tf-idf scores": 0.1273977530260378}, {"Date": "2010-12-14T00:00:00", "vocab": "prices", "tf-idf scores": 0.12515096245988688}, {"Date": "2010-12-14T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12458461782423036}, {"Date": "2010-11-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.17761800496489358}, {"Date": "2010-11-03T00:00:00", "vocab": "remained", "tf-idf scores": 0.1776309408818188}, {"Date": "2010-11-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.15396014688547818}, {"Date": "2010-11-03T00:00:00", "vocab": "securities", "tf-idf scores": 0.14718204689127848}, {"Date": "2010-11-03T00:00:00", "vocab": "continued", "tf-idf scores": 0.12435139432447304}, {"Date": "2010-11-03T00:00:00", "vocab": "september", "tf-idf scores": 0.11973719550924163}, {"Date": "2010-11-03T00:00:00", "vocab": "price", "tf-idf scores": 0.11837379766239187}, {"Date": "2010-11-03T00:00:00", "vocab": "increase", "tf-idf scores": 0.10892241702429789}, {"Date": "2010-11-03T00:00:00", "vocab": "levels", "tf-idf scores": 0.10795652667049709}, {"Date": "2010-11-03T00:00:00", "vocab": "generally", "tf-idf scores": 0.10700791836328737}, {"Date": "2010-10-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.17761899165521125}, {"Date": "2010-10-15T00:00:00", "vocab": "remained", "tf-idf scores": 0.1776206673524342}, {"Date": "2010-10-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.1539629303896382}, {"Date": "2010-10-15T00:00:00", "vocab": "securities", "tf-idf scores": 0.1471126256937377}, {"Date": "2010-10-15T00:00:00", "vocab": "continued", "tf-idf scores": 0.12432434298983328}, {"Date": "2010-10-15T00:00:00", "vocab": "september", "tf-idf scores": 0.11971328718228941}, {"Date": "2010-10-15T00:00:00", "vocab": "price", "tf-idf scores": 0.11839005954157597}, {"Date": "2010-10-15T00:00:00", "vocab": "increase", "tf-idf scores": 0.10893205044992287}, {"Date": "2010-10-15T00:00:00", "vocab": "levels", "tf-idf scores": 0.10798640871429271}, {"Date": "2010-10-15T00:00:00", "vocab": "generally", "tf-idf scores": 0.1069951412510251}, {"Date": "2010-09-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.2284548629202028}, {"Date": "2010-09-21T00:00:00", "vocab": "july", "tf-idf scores": 0.21767905819432923}, {"Date": "2010-09-21T00:00:00", "vocab": "august", "tf-idf scores": 0.20328512514844568}, {"Date": "2010-09-21T00:00:00", "vocab": "remained", "tf-idf scores": 0.1903549233728151}, {"Date": "2010-09-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18398198946361644}, {"Date": "2010-09-21T00:00:00", "vocab": "continued", "tf-idf scores": 0.13327646243316732}, {"Date": "2010-09-21T00:00:00", "vocab": "prices", "tf-idf scores": 0.12111738040491242}, {"Date": "2010-09-21T00:00:00", "vocab": "intermeeting", "tf-idf scores": 0.11469292848094223}, {"Date": "2010-09-21T00:00:00", "vocab": "financial", "tf-idf scores": 0.10879776997481792}, {"Date": "2010-09-21T00:00:00", "vocab": "real", "tf-idf scores": 0.10838668830544923}, {"Date": "2010-08-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.22402057002026016}, {"Date": "2010-08-10T00:00:00", "vocab": "june", "tf-idf scores": 0.18757086419542326}, {"Date": "2010-08-10T00:00:00", "vocab": "remained", "tf-idf scores": 0.14930886357742734}, {"Date": "2010-08-10T00:00:00", "vocab": "second", "tf-idf scores": 0.14463251017088047}, {"Date": "2010-08-10T00:00:00", "vocab": "recent", "tf-idf scores": 0.13786730390709748}, {"Date": "2010-08-10T00:00:00", "vocab": "data", "tf-idf scores": 0.1332639485550264}, {"Date": "2010-08-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.1327276938685343}, {"Date": "2010-08-10T00:00:00", "vocab": "continued", "tf-idf scores": 0.1321560962477521}, {"Date": "2010-08-10T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13208424320098108}, {"Date": "2010-08-10T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1263376541646975}, {"Date": "2010-06-23T00:00:00", "vocab": "april", "tf-idf scores": 0.22683708072566958}, {"Date": "2010-06-23T00:00:00", "vocab": "economic", "tf-idf scores": 0.18672813920510237}, {"Date": "2010-06-23T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18668818991606312}, {"Date": "2010-06-23T00:00:00", "vocab": "financial", "tf-idf scores": 0.16736343963154351}, {"Date": "2010-06-23T00:00:00", "vocab": "continued", "tf-idf scores": 0.14006498682060495}, {"Date": "2010-06-23T00:00:00", "vocab": "prices", "tf-idf scores": 0.13022174624903496}, {"Date": "2010-06-23T00:00:00", "vocab": "recent", "tf-idf scores": 0.1296114488563364}, {"Date": "2010-06-23T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1244487634133661}, {"Date": "2010-06-23T00:00:00", "vocab": "securities", "tf-idf scores": 0.11130206701934302}, {"Date": "2010-06-23T00:00:00", "vocab": "european", "tf-idf scores": 0.11021799146778376}, {"Date": "2010-05-09T00:00:00", "vocab": "april", "tf-idf scores": 0.22689384222198505}, {"Date": "2010-05-09T00:00:00", "vocab": "economic", "tf-idf scores": 0.1866749924195188}, {"Date": "2010-05-09T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18667049104513908}, {"Date": "2010-05-09T00:00:00", "vocab": "financial", "tf-idf scores": 0.16737786777964764}, {"Date": "2010-05-09T00:00:00", "vocab": "continued", "tf-idf scores": 0.14006518537440452}, {"Date": "2010-05-09T00:00:00", "vocab": "prices", "tf-idf scores": 0.13025528691685498}, {"Date": "2010-05-09T00:00:00", "vocab": "recent", "tf-idf scores": 0.12970027227375397}, {"Date": "2010-05-09T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12443510947382755}, {"Date": "2010-05-09T00:00:00", "vocab": "securities", "tf-idf scores": 0.11131166125115403}, {"Date": "2010-05-09T00:00:00", "vocab": "european", "tf-idf scores": 0.11019944711884483}, {"Date": "2010-04-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21496025684365602}, {"Date": "2010-04-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.2042802343018899}, {"Date": "2010-04-28T00:00:00", "vocab": "continued", "tf-idf scores": 0.1343895919174752}, {"Date": "2010-04-28T00:00:00", "vocab": "sales", "tf-idf scores": 0.1320045821455773}, {"Date": "2010-04-28T00:00:00", "vocab": "market", "tf-idf scores": 0.1290510577539279}, {"Date": "2010-04-28T00:00:00", "vocab": "recovery", "tf-idf scores": 0.12631024450023032}, {"Date": "2010-04-28T00:00:00", "vocab": "recent", "tf-idf scores": 0.12358689417268097}, {"Date": "2010-04-28T00:00:00", "vocab": "strategy", "tf-idf scores": 0.11320262589665131}, {"Date": "2010-04-28T00:00:00", "vocab": "remained", "tf-idf scores": 0.11293763445973765}, {"Date": "2010-04-28T00:00:00", "vocab": "credit", "tf-idf scores": 0.11171618325978402}, {"Date": "2010-03-16T00:00:00", "vocab": "january", "tf-idf scores": 0.2124398685480488}, {"Date": "2010-03-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1921415215594444}, {"Date": "2010-03-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.1688644440761292}, {"Date": "2010-03-16T00:00:00", "vocab": "market", "tf-idf scores": 0.14556112137222024}, {"Date": "2010-03-16T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1280707801582042}, {"Date": "2010-03-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.1281312006453008}, {"Date": "2010-03-16T00:00:00", "vocab": "remained", "tf-idf scores": 0.11645154672068221}, {"Date": "2010-03-16T00:00:00", "vocab": "credit", "tf-idf scores": 0.11430091058956479}, {"Date": "2010-03-16T00:00:00", "vocab": "recent", "tf-idf scores": 0.11069743957539561}, {"Date": "2010-03-16T00:00:00", "vocab": "agency", "tf-idf scores": 0.10662343732485692}, {"Date": "2010-01-27T00:00:00", "vocab": "market", "tf-idf scores": 0.2027559401734201}, {"Date": "2010-01-27T00:00:00", "vocab": "shall", "tf-idf scores": 0.19647052472104273}, {"Date": "2010-01-27T00:00:00", "vocab": "foreign", "tf-idf scores": 0.19307492367589452}, {"Date": "2010-01-27T00:00:00", "vocab": "securities", "tf-idf scores": 0.16368695152277773}, {"Date": "2010-01-27T00:00:00", "vocab": "currency", "tf-idf scores": 0.14618303303471608}, {"Date": "2010-01-27T00:00:00", "vocab": "open", "tf-idf scores": 0.12612832987738784}, {"Date": "2010-01-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.12557149363257883}, {"Date": "2010-01-27T00:00:00", "vocab": "inflation", "tf-idf scores": 0.12231894574617123}, {"Date": "2010-01-27T00:00:00", "vocab": "ioer", "tf-idf scores": 0.1185608920541839}, {"Date": "2010-01-27T00:00:00", "vocab": "bank", "tf-idf scores": 0.10673058205709432}, {"Date": "2009-12-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22383593357542328}, {"Date": "2009-12-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.21898501669602344}, {"Date": "2009-12-15T00:00:00", "vocab": "november", "tf-idf scores": 0.15542288394327658}, {"Date": "2009-12-15T00:00:00", "vocab": "continued", "tf-idf scores": 0.1363161151958252}, {"Date": "2009-12-15T00:00:00", "vocab": "market", "tf-idf scores": 0.13630535070841468}, {"Date": "2009-12-15T00:00:00", "vocab": "credit", "tf-idf scores": 0.12369102857816289}, {"Date": "2009-12-15T00:00:00", "vocab": "recent", "tf-idf scores": 0.11678158306496658}, {"Date": "2009-12-15T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11198380437903843}, {"Date": "2009-12-15T00:00:00", "vocab": "remained", "tf-idf scores": 0.11192636459191266}, {"Date": "2009-12-15T00:00:00", "vocab": "increased", "tf-idf scores": 0.1075296389263796}, {"Date": "2009-11-04T00:00:00", "vocab": "continued", "tf-idf scores": 0.18985968807009285}, {"Date": "2009-11-04T00:00:00", "vocab": "economic", "tf-idf scores": 0.17087536694863445}, {"Date": "2009-11-04T00:00:00", "vocab": "agency", "tf-idf scores": 0.14255527875607896}, {"Date": "2009-11-04T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13294986175602097}, {"Date": "2009-11-04T00:00:00", "vocab": "credit", "tf-idf scores": 0.13158812247071572}, {"Date": "2009-11-04T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12659869841127813}, {"Date": "2009-11-04T00:00:00", "vocab": "market", "tf-idf scores": 0.12659953794500517}, {"Date": "2009-11-04T00:00:00", "vocab": "recovery", "tf-idf scores": 0.12086120483487789}, {"Date": "2009-11-04T00:00:00", "vocab": "remained", "tf-idf scores": 0.12024183964034944}, {"Date": "2009-11-04T00:00:00", "vocab": "tools", "tf-idf scores": 0.1157685628760912}, {"Date": "2009-09-22T00:00:00", "vocab": "economic", "tf-idf scores": 0.17559319129667897}, {"Date": "2009-09-22T00:00:00", "vocab": "market", "tf-idf scores": 0.1635029578745215}, {"Date": "2009-09-22T00:00:00", "vocab": "august", "tf-idf scores": 0.1482945095395386}, {"Date": "2009-09-22T00:00:00", "vocab": "continued", "tf-idf scores": 0.13325948869865492}, {"Date": "2009-09-22T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13321060932608647}, {"Date": "2009-09-22T00:00:00", "vocab": "credit", "tf-idf scores": 0.1328849517716223}, {"Date": "2009-09-22T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12718363707805025}, {"Date": "2009-09-22T00:00:00", "vocab": "remained", "tf-idf scores": 0.12109905968168977}, {"Date": "2009-09-22T00:00:00", "vocab": "financial", "tf-idf scores": 0.11607725317598884}, {"Date": "2009-09-22T00:00:00", "vocab": "prices", "tf-idf scores": 0.11556245643558266}, {"Date": "2009-08-11T00:00:00", "vocab": "talf", "tf-idf scores": 0.1692688097997699}, {"Date": "2009-08-11T00:00:00", "vocab": "credit", "tf-idf scores": 0.16893830376063018}, {"Date": "2009-08-11T00:00:00", "vocab": "second", "tf-idf scores": 0.15987978979890857}, {"Date": "2009-08-11T00:00:00", "vocab": "economic", "tf-idf scores": 0.14627496016326183}, {"Date": "2009-08-11T00:00:00", "vocab": "continued", "tf-idf scores": 0.1396562251886956}, {"Date": "2009-08-11T00:00:00", "vocab": "july", "tf-idf scores": 0.126760855616156}, {"Date": "2009-08-11T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12640014577537007}, {"Date": "2009-08-11T00:00:00", "vocab": "market", "tf-idf scores": 0.12640404713823097}, {"Date": "2009-08-11T00:00:00", "vocab": "markets", "tf-idf scores": 0.11969147635543496}, {"Date": "2009-08-11T00:00:00", "vocab": "remained", "tf-idf scores": 0.11975663680211432}, {"Date": "2009-06-24T00:00:00", "vocab": "market", "tf-idf scores": 0.23809432849550882}, {"Date": "2009-06-24T00:00:00", "vocab": "april", "tf-idf scores": 0.1542769879376954}, {"Date": "2009-06-24T00:00:00", "vocab": "economic", "tf-idf scores": 0.14282925747149267}, {"Date": "2009-06-24T00:00:00", "vocab": "financial", "tf-idf scores": 0.13880702961375227}, {"Date": "2009-06-24T00:00:00", "vocab": "likely", "tf-idf scores": 0.13760626092046846}, {"Date": "2009-06-24T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12694189134149778}, {"Date": "2009-06-24T00:00:00", "vocab": "securities", "tf-idf scores": 0.12554647388251605}, {"Date": "2009-06-24T00:00:00", "vocab": "programs", "tf-idf scores": 0.12223969963678567}, {"Date": "2009-06-24T00:00:00", "vocab": "remained", "tf-idf scores": 0.12173628987047863}, {"Date": "2009-06-24T00:00:00", "vocab": "credit", "tf-idf scores": 0.11611946771589798}, {"Date": "2009-06-03T00:00:00", "vocab": "market", "tf-idf scores": 0.238077273082483}, {"Date": "2009-06-03T00:00:00", "vocab": "april", "tf-idf scores": 0.15430005035038377}, {"Date": "2009-06-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.14283983979432655}, {"Date": "2009-06-03T00:00:00", "vocab": "financial", "tf-idf scores": 0.1387252718168668}, {"Date": "2009-06-03T00:00:00", "vocab": "likely", "tf-idf scores": 0.13760343999257538}, {"Date": "2009-06-03T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12693789869126448}, {"Date": "2009-06-03T00:00:00", "vocab": "securities", "tf-idf scores": 0.1255656099278124}, {"Date": "2009-06-03T00:00:00", "vocab": "programs", "tf-idf scores": 0.12227110009317342}, {"Date": "2009-06-03T00:00:00", "vocab": "remained", "tf-idf scores": 0.12167055096810543}, {"Date": "2009-06-03T00:00:00", "vocab": "credit", "tf-idf scores": 0.11613287498296816}, {"Date": "2009-04-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.2353146896570232}, {"Date": "2009-04-29T00:00:00", "vocab": "financial", "tf-idf scores": 0.21095804558053732}, {"Date": "2009-04-29T00:00:00", "vocab": "credit", "tf-idf scores": 0.16607699269614087}, {"Date": "2009-04-29T00:00:00", "vocab": "march", "tf-idf scores": 0.154191530953067}, {"Date": "2009-04-29T00:00:00", "vocab": "market", "tf-idf scores": 0.13732400544210455}, {"Date": "2009-04-29T00:00:00", "vocab": "securities", "tf-idf scores": 0.13298264658737916}, {"Date": "2009-04-29T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1307118226159777}, {"Date": "2009-04-29T00:00:00", "vocab": "bank", "tf-idf scores": 0.10502317771810017}, {"Date": "2009-04-29T00:00:00", "vocab": "markets", "tf-idf scores": 0.10463900836209576}, {"Date": "2009-04-29T00:00:00", "vocab": "remained", "tf-idf scores": 0.09804457002087322}, {"Date": "2009-03-17T00:00:00", "vocab": "talf", "tf-idf scores": 0.24586573669760756}, {"Date": "2009-03-17T00:00:00", "vocab": "financial", "tf-idf scores": 0.15595365001476894}, {"Date": "2009-03-17T00:00:00", "vocab": "january", "tf-idf scores": 0.14307153876764495}, {"Date": "2009-03-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.13528621101997987}, {"Date": "2009-03-17T00:00:00", "vocab": "market", "tf-idf scores": 0.1353272993233773}, {"Date": "2009-03-17T00:00:00", "vocab": "purchase", "tf-idf scores": 0.1316194482863882}, {"Date": "2009-03-17T00:00:00", "vocab": "bank", "tf-idf scores": 0.1294219814990954}, {"Date": "2009-03-17T00:00:00", "vocab": "purchases", "tf-idf scores": 0.12222944027632017}, {"Date": "2009-03-17T00:00:00", "vocab": "fourth", "tf-idf scores": 0.12010139791154369}, {"Date": "2009-03-17T00:00:00", "vocab": "billion", "tf-idf scores": 0.11624245895753636}, {"Date": "2009-02-07T00:00:00", "vocab": "talf", "tf-idf scores": 0.24584302950645545}, {"Date": "2009-02-07T00:00:00", "vocab": "financial", "tf-idf scores": 0.15595590688549993}, {"Date": "2009-02-07T00:00:00", "vocab": "january", "tf-idf scores": 0.1430793567258962}, {"Date": "2009-02-07T00:00:00", "vocab": "economic", "tf-idf scores": 0.1352510987926593}, {"Date": "2009-02-07T00:00:00", "vocab": "market", "tf-idf scores": 0.13532283975755674}, {"Date": "2009-02-07T00:00:00", "vocab": "purchase", "tf-idf scores": 0.13156814441404266}, {"Date": "2009-02-07T00:00:00", "vocab": "bank", "tf-idf scores": 0.1294186966731703}, {"Date": "2009-02-07T00:00:00", "vocab": "purchases", "tf-idf scores": 0.12224065854903346}, {"Date": "2009-02-07T00:00:00", "vocab": "fourth", "tf-idf scores": 0.12014427085175758}, {"Date": "2009-02-07T00:00:00", "vocab": "billion", "tf-idf scores": 0.11625349276151499}, {"Date": "2009-01-28T00:00:00", "vocab": "market", "tf-idf scores": 0.24588987762531325}, {"Date": "2009-01-28T00:00:00", "vocab": "shall", "tf-idf scores": 0.21442586789058118}, {"Date": "2009-01-28T00:00:00", "vocab": "foreign", "tf-idf scores": 0.20016416355921354}, {"Date": "2009-01-28T00:00:00", "vocab": "currency", "tf-idf scores": 0.17790822617369442}, {"Date": "2009-01-28T00:00:00", "vocab": "securities", "tf-idf scores": 0.15478298840501448}, {"Date": "2009-01-28T00:00:00", "vocab": "open", "tf-idf scores": 0.14107544120212795}, {"Date": "2009-01-28T00:00:00", "vocab": "credit", "tf-idf scores": 0.12976685274191394}, {"Date": "2009-01-28T00:00:00", "vocab": "financial", "tf-idf scores": 0.12755970907719993}, {"Date": "2009-01-28T00:00:00", "vocab": "programs", "tf-idf scores": 0.12625827063403489}, {"Date": "2009-01-28T00:00:00", "vocab": "chairman", "tf-idf scores": 0.11604091704674901}, {"Date": "2009-01-16T00:00:00", "vocab": "market", "tf-idf scores": 0.2458479051895818}, {"Date": "2009-01-16T00:00:00", "vocab": "shall", "tf-idf scores": 0.21438997543828955}, {"Date": "2009-01-16T00:00:00", "vocab": "foreign", "tf-idf scores": 0.200199516380293}, {"Date": "2009-01-16T00:00:00", "vocab": "currency", "tf-idf scores": 0.177943280907517}, {"Date": "2009-01-16T00:00:00", "vocab": "securities", "tf-idf scores": 0.15473665725429242}, {"Date": "2009-01-16T00:00:00", "vocab": "open", "tf-idf scores": 0.14109889926906508}, {"Date": "2009-01-16T00:00:00", "vocab": "credit", "tf-idf scores": 0.12977222336283753}, {"Date": "2009-01-16T00:00:00", "vocab": "financial", "tf-idf scores": 0.12760225557631333}, {"Date": "2009-01-16T00:00:00", "vocab": "programs", "tf-idf scores": 0.12619013775847357}, {"Date": "2009-01-16T00:00:00", "vocab": "chairman", "tf-idf scores": 0.11613637791852494}, {"Date": "2008-12-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.21310214317998522}, {"Date": "2008-12-16T00:00:00", "vocab": "market", "tf-idf scores": 0.1752478245165122}, {"Date": "2008-12-16T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1562474131000848}, {"Date": "2008-12-16T00:00:00", "vocab": "financial", "tf-idf scores": 0.15282022285626412}, {"Date": "2008-12-16T00:00:00", "vocab": "credit", "tf-idf scores": 0.136757922154756}, {"Date": "2008-12-16T00:00:00", "vocab": "prices", "tf-idf scores": 0.13313612034236796}, {"Date": "2008-12-16T00:00:00", "vocab": "drops", "tf-idf scores": 0.13099840358945722}, {"Date": "2008-12-16T00:00:00", "vocab": "decline", "tf-idf scores": 0.12102223600056396}, {"Date": "2008-12-16T00:00:00", "vocab": "funds", "tf-idf scores": 0.1184253697615709}, {"Date": "2008-12-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11841502229085506}, {"Date": "2008-10-29T00:00:00", "vocab": "financial", "tf-idf scores": 0.22680032905479408}, {"Date": "2008-10-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.22021325980077228}, {"Date": "2008-10-29T00:00:00", "vocab": "market", "tf-idf scores": 0.21085375770866585}, {"Date": "2008-10-29T00:00:00", "vocab": "credit", "tf-idf scores": 0.13532908539114466}, {"Date": "2008-10-29T00:00:00", "vocab": "september", "tf-idf scores": 0.1238641487686643}, {"Date": "2008-10-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1218419618841182}, {"Date": "2008-10-29T00:00:00", "vocab": "october", "tf-idf scores": 0.11524330486652956}, {"Date": "2008-10-29T00:00:00", "vocab": "liquidity", "tf-idf scores": 0.11057097941164924}, {"Date": "2008-10-29T00:00:00", "vocab": "growth", "tf-idf scores": 0.10351948457210688}, {"Date": "2008-10-29T00:00:00", "vocab": "conditions", "tf-idf scores": 0.10311838795062121}, {"Date": "2008-10-07T00:00:00", "vocab": "financial", "tf-idf scores": 0.22680151330128212}, {"Date": "2008-10-07T00:00:00", "vocab": "economic", "tf-idf scores": 0.22022727981753787}, {"Date": "2008-10-07T00:00:00", "vocab": "market", "tf-idf scores": 0.21082470547785803}, {"Date": "2008-10-07T00:00:00", "vocab": "credit", "tf-idf scores": 0.1352522999158978}, {"Date": "2008-10-07T00:00:00", "vocab": "september", "tf-idf scores": 0.1239198175258887}, {"Date": "2008-10-07T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1218809240148687}, {"Date": "2008-10-07T00:00:00", "vocab": "october", "tf-idf scores": 0.11522412991514518}, {"Date": "2008-10-07T00:00:00", "vocab": "liquidity", "tf-idf scores": 0.1106590349116056}, {"Date": "2008-10-07T00:00:00", "vocab": "growth", "tf-idf scores": 0.10352025601039094}, {"Date": "2008-10-07T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1031222149766012}, {"Date": "2008-09-29T00:00:00", "vocab": "financial", "tf-idf scores": 0.2268933235001001}, {"Date": "2008-09-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.22019314742331547}, {"Date": "2008-09-29T00:00:00", "vocab": "market", "tf-idf scores": 0.21084741626043754}, {"Date": "2008-09-29T00:00:00", "vocab": "credit", "tf-idf scores": 0.13529702648760464}, {"Date": "2008-09-29T00:00:00", "vocab": "september", "tf-idf scores": 0.12393395892858614}, {"Date": "2008-09-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.12186061047522206}, {"Date": "2008-09-29T00:00:00", "vocab": "october", "tf-idf scores": 0.1152021469094383}, {"Date": "2008-09-29T00:00:00", "vocab": "liquidity", "tf-idf scores": 0.1106194663180134}, {"Date": "2008-09-29T00:00:00", "vocab": "growth", "tf-idf scores": 0.10351608232788503}, {"Date": "2008-09-29T00:00:00", "vocab": "conditions", "tf-idf scores": 0.10312913399050003}, {"Date": "2008-09-16T00:00:00", "vocab": "financial", "tf-idf scores": 0.23564289419936538}, {"Date": "2008-09-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.22705475918049525}, {"Date": "2008-09-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21102790730405632}, {"Date": "2008-09-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.20349492413876266}, {"Date": "2008-09-16T00:00:00", "vocab": "august", "tf-idf scores": 0.1988354359984984}, {"Date": "2008-09-16T00:00:00", "vocab": "prices", "tf-idf scores": 0.1741330508576955}, {"Date": "2008-09-16T00:00:00", "vocab": "market", "tf-idf scores": 0.12817043735145656}, {"Date": "2008-09-16T00:00:00", "vocab": "strains", "tf-idf scores": 0.12638085644042138}, {"Date": "2008-09-16T00:00:00", "vocab": "credit", "tf-idf scores": 0.12186815355144373}, {"Date": "2008-09-16T00:00:00", "vocab": "real", "tf-idf scores": 0.12068374058454631}, {"Date": "2008-08-08T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2920221719084552}, {"Date": "2008-08-08T00:00:00", "vocab": "second", "tf-idf scores": 0.17932870129609918}, {"Date": "2008-08-08T00:00:00", "vocab": "prices", "tf-idf scores": 0.17883015551249049}, {"Date": "2008-08-08T00:00:00", "vocab": "growth", "tf-idf scores": 0.17166950744715684}, {"Date": "2008-08-08T00:00:00", "vocab": "financial", "tf-idf scores": 0.16529756826318331}, {"Date": "2008-08-08T00:00:00", "vocab": "june", "tf-idf scores": 0.13294822183343438}, {"Date": "2008-08-08T00:00:00", "vocab": "continued", "tf-idf scores": 0.12819501533277053}, {"Date": "2008-08-08T00:00:00", "vocab": "economic", "tf-idf scores": 0.1282242011469047}, {"Date": "2008-08-08T00:00:00", "vocab": "market", "tf-idf scores": 0.12107571828180952}, {"Date": "2008-08-08T00:00:00", "vocab": "remained", "tf-idf scores": 0.12108518237534972}, {"Date": "2008-07-24T00:00:00", "vocab": "inflation", "tf-idf scores": 0.29201360541657057}, {"Date": "2008-07-24T00:00:00", "vocab": "second", "tf-idf scores": 0.1793681330614253}, {"Date": "2008-07-24T00:00:00", "vocab": "prices", "tf-idf scores": 0.17881747095618805}, {"Date": "2008-07-24T00:00:00", "vocab": "growth", "tf-idf scores": 0.1717018762818257}, {"Date": "2008-07-24T00:00:00", "vocab": "financial", "tf-idf scores": 0.16527608569698785}, {"Date": "2008-07-24T00:00:00", "vocab": "june", "tf-idf scores": 0.13296007876696264}, {"Date": "2008-07-24T00:00:00", "vocab": "continued", "tf-idf scores": 0.1281926783224082}, {"Date": "2008-07-24T00:00:00", "vocab": "economic", "tf-idf scores": 0.12821038944716745}, {"Date": "2008-07-24T00:00:00", "vocab": "market", "tf-idf scores": 0.121100862751}, {"Date": "2008-07-24T00:00:00", "vocab": "remained", "tf-idf scores": 0.12106010648648095}, {"Date": "2008-06-25T00:00:00", "vocab": "inflation", "tf-idf scores": 0.28884431954255263}, {"Date": "2008-06-25T00:00:00", "vocab": "april", "tf-idf scores": 0.26331355737193657}, {"Date": "2008-06-25T00:00:00", "vocab": "economic", "tf-idf scores": 0.18054769252134348}, {"Date": "2008-06-25T00:00:00", "vocab": "credit", "tf-idf scores": 0.17378652994077698}, {"Date": "2008-06-25T00:00:00", "vocab": "financial", "tf-idf scores": 0.16393989327439001}, {"Date": "2008-06-25T00:00:00", "vocab": "growth", "tf-idf scores": 0.1450652309294795}, {"Date": "2008-06-25T00:00:00", "vocab": "prices", "tf-idf scores": 0.14507248753884788}, {"Date": "2008-06-25T00:00:00", "vocab": "recent", "tf-idf scores": 0.138490560825333}, {"Date": "2008-06-25T00:00:00", "vocab": "remained", "tf-idf scores": 0.13846393624186823}, {"Date": "2008-06-25T00:00:00", "vocab": "market", "tf-idf scores": 0.12638444516180306}, {"Date": "2008-04-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2487987773387902}, {"Date": "2008-04-30T00:00:00", "vocab": "march", "tf-idf scores": 0.20957601730190004}, {"Date": "2008-04-30T00:00:00", "vocab": "financial", "tf-idf scores": 0.19639805793515835}, {"Date": "2008-04-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.17311367938092465}, {"Date": "2008-04-30T00:00:00", "vocab": "growth", "tf-idf scores": 0.15756078303033533}, {"Date": "2008-04-30T00:00:00", "vocab": "prices", "tf-idf scores": 0.1520747524529758}, {"Date": "2008-04-30T00:00:00", "vocab": "remained", "tf-idf scores": 0.1405947347428768}, {"Date": "2008-04-30T00:00:00", "vocab": "credit", "tf-idf scores": 0.13740652876645426}, {"Date": "2008-04-30T00:00:00", "vocab": "markets", "tf-idf scores": 0.13518426861509375}, {"Date": "2008-04-30T00:00:00", "vocab": "recent", "tf-idf scores": 0.12977345471778828}, {"Date": "2008-03-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2360690402221032}, {"Date": "2008-03-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.20241244902518074}, {"Date": "2008-03-18T00:00:00", "vocab": "prices", "tf-idf scores": 0.19643061125333647}, {"Date": "2008-03-18T00:00:00", "vocab": "january", "tf-idf scores": 0.1604595197513949}, {"Date": "2008-03-18T00:00:00", "vocab": "financial", "tf-idf scores": 0.15649271402078657}, {"Date": "2008-03-18T00:00:00", "vocab": "credit", "tf-idf scores": 0.14805833211440358}, {"Date": "2008-03-18T00:00:00", "vocab": "real", "tf-idf scores": 0.13682245744109392}, {"Date": "2008-03-18T00:00:00", "vocab": "market", "tf-idf scores": 0.12818077802169237}, {"Date": "2008-03-18T00:00:00", "vocab": "growth", "tf-idf scores": 0.12201707579321125}, {"Date": "2008-03-18T00:00:00", "vocab": "markets", "tf-idf scores": 0.12145302800207242}, {"Date": "2008-03-10T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23612441477342305}, {"Date": "2008-03-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.20233731744015043}, {"Date": "2008-03-10T00:00:00", "vocab": "prices", "tf-idf scores": 0.19645806213774722}, {"Date": "2008-03-10T00:00:00", "vocab": "january", "tf-idf scores": 0.16048725224425575}, {"Date": "2008-03-10T00:00:00", "vocab": "financial", "tf-idf scores": 0.15652456407816914}, {"Date": "2008-03-10T00:00:00", "vocab": "credit", "tf-idf scores": 0.14807756923313917}, {"Date": "2008-03-10T00:00:00", "vocab": "real", "tf-idf scores": 0.13678867623646956}, {"Date": "2008-03-10T00:00:00", "vocab": "market", "tf-idf scores": 0.12817220974029936}, {"Date": "2008-03-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.12193770465955102}, {"Date": "2008-03-10T00:00:00", "vocab": "markets", "tf-idf scores": 0.12148158005654673}, {"Date": "2008-01-30T00:00:00", "vocab": "shall", "tf-idf scores": 0.2507306183698131}, {"Date": "2008-01-30T00:00:00", "vocab": "foreign", "tf-idf scores": 0.22829644841159513}, {"Date": "2008-01-30T00:00:00", "vocab": "currency", "tf-idf scores": 0.18568061907063446}, {"Date": "2008-01-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.17319912258863468}, {"Date": "2008-01-30T00:00:00", "vocab": "market", "tf-idf scores": 0.16530579431529677}, {"Date": "2008-01-30T00:00:00", "vocab": "fourth", "tf-idf scores": 0.1524688858931739}, {"Date": "2008-01-30T00:00:00", "vocab": "december", "tf-idf scores": 0.1479552930162217}, {"Date": "2008-01-30T00:00:00", "vocab": "growth", "tf-idf scores": 0.13836033557085972}, {"Date": "2008-01-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1377931067090675}, {"Date": "2008-01-30T00:00:00", "vocab": "open", "tf-idf scores": 0.1344652436664117}, {"Date": "2008-01-21T00:00:00", "vocab": "shall", "tf-idf scores": 0.25069047539295564}, {"Date": "2008-01-21T00:00:00", "vocab": "foreign", "tf-idf scores": 0.2282729048774888}, {"Date": "2008-01-21T00:00:00", "vocab": "currency", "tf-idf scores": 0.1856494612910833}, {"Date": "2008-01-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.1731990699769455}, {"Date": "2008-01-21T00:00:00", "vocab": "market", "tf-idf scores": 0.16530671756252444}, {"Date": "2008-01-21T00:00:00", "vocab": "fourth", "tf-idf scores": 0.15248115103992227}, {"Date": "2008-01-21T00:00:00", "vocab": "december", "tf-idf scores": 0.1480119353295544}, {"Date": "2008-01-21T00:00:00", "vocab": "growth", "tf-idf scores": 0.13837286641123997}, {"Date": "2008-01-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13776397054484035}, {"Date": "2008-01-21T00:00:00", "vocab": "open", "tf-idf scores": 0.13444678131555182}, {"Date": "2008-01-09T00:00:00", "vocab": "shall", "tf-idf scores": 0.2507112336472935}, {"Date": "2008-01-09T00:00:00", "vocab": "foreign", "tf-idf scores": 0.22834572651936316}, {"Date": "2008-01-09T00:00:00", "vocab": "currency", "tf-idf scores": 0.18566747856873642}, {"Date": "2008-01-09T00:00:00", "vocab": "economic", "tf-idf scores": 0.17323641984209112}, {"Date": "2008-01-09T00:00:00", "vocab": "market", "tf-idf scores": 0.16532423899380838}, {"Date": "2008-01-09T00:00:00", "vocab": "fourth", "tf-idf scores": 0.15245271376143027}, {"Date": "2008-01-09T00:00:00", "vocab": "december", "tf-idf scores": 0.1479944646444813}, {"Date": "2008-01-09T00:00:00", "vocab": "growth", "tf-idf scores": 0.13840146185993113}, {"Date": "2008-01-09T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13784365088024098}, {"Date": "2008-01-09T00:00:00", "vocab": "open", "tf-idf scores": 0.13445273753165365}, {"Date": "2007-12-11T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2214058307515546}, {"Date": "2007-12-11T00:00:00", "vocab": "growth", "tf-idf scores": 0.20852956896780062}, {"Date": "2007-12-11T00:00:00", "vocab": "financial", "tf-idf scores": 0.18850053864905977}, {"Date": "2007-12-11T00:00:00", "vocab": "october", "tf-idf scores": 0.17015741086826905}, {"Date": "2007-12-11T00:00:00", "vocab": "prices", "tf-idf scores": 0.15289712976805622}, {"Date": "2007-12-11T00:00:00", "vocab": "real", "tf-idf scores": 0.14770280678687686}, {"Date": "2007-12-11T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1314945294732115}, {"Date": "2007-12-11T00:00:00", "vocab": "credit", "tf-idf scores": 0.1278629365773155}, {"Date": "2007-12-11T00:00:00", "vocab": "economic", "tf-idf scores": 0.11763861832982223}, {"Date": "2007-12-11T00:00:00", "vocab": "core", "tf-idf scores": 0.11183566699955645}, {"Date": "2007-12-06T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2214427343878316}, {"Date": "2007-12-06T00:00:00", "vocab": "growth", "tf-idf scores": 0.20854528571287048}, {"Date": "2007-12-06T00:00:00", "vocab": "financial", "tf-idf scores": 0.18848316996840633}, {"Date": "2007-12-06T00:00:00", "vocab": "october", "tf-idf scores": 0.17008226762887627}, {"Date": "2007-12-06T00:00:00", "vocab": "prices", "tf-idf scores": 0.1529573307980766}, {"Date": "2007-12-06T00:00:00", "vocab": "real", "tf-idf scores": 0.14773200072704645}, {"Date": "2007-12-06T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13150594609485877}, {"Date": "2007-12-06T00:00:00", "vocab": "credit", "tf-idf scores": 0.1278494461949737}, {"Date": "2007-12-06T00:00:00", "vocab": "economic", "tf-idf scores": 0.11766936694077403}, {"Date": "2007-12-06T00:00:00", "vocab": "core", "tf-idf scores": 0.11174578438339049}, {"Date": "2007-10-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.20682371841563868}, {"Date": "2007-10-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.20679389075520446}, {"Date": "2007-10-31T00:00:00", "vocab": "markets", "tf-idf scores": 0.193485033864015}, {"Date": "2007-10-31T00:00:00", "vocab": "august", "tf-idf scores": 0.1760130333943853}, {"Date": "2007-10-31T00:00:00", "vocab": "september", "tf-idf scores": 0.1660184348165536}, {"Date": "2007-10-31T00:00:00", "vocab": "growth", "tf-idf scores": 0.16086271052031476}, {"Date": "2007-10-31T00:00:00", "vocab": "prices", "tf-idf scores": 0.1407490855933301}, {"Date": "2007-10-31T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13349175121403428}, {"Date": "2007-10-31T00:00:00", "vocab": "financial", "tf-idf scores": 0.1279285991523088}, {"Date": "2007-10-31T00:00:00", "vocab": "recent", "tf-idf scores": 0.12011058077942682}, {"Date": "2007-09-18T00:00:00", "vocab": "financial", "tf-idf scores": 0.22430023620750478}, {"Date": "2007-09-18T00:00:00", "vocab": "credit", "tf-idf scores": 0.19257266011533283}, {"Date": "2007-09-18T00:00:00", "vocab": "market", "tf-idf scores": 0.18766381592808007}, {"Date": "2007-09-18T00:00:00", "vocab": "july", "tf-idf scores": 0.18533836410686547}, {"Date": "2007-09-18T00:00:00", "vocab": "recent", "tf-idf scores": 0.173752702184585}, {"Date": "2007-09-18T00:00:00", "vocab": "august", "tf-idf scores": 0.17018496976209402}, {"Date": "2007-09-18T00:00:00", "vocab": "growth", "tf-idf scores": 0.13964579607469593}, {"Date": "2007-09-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.1390065836721847}, {"Date": "2007-09-18T00:00:00", "vocab": "markets", "tf-idf scores": 0.1389721895987941}, {"Date": "2007-09-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13207454980580466}, {"Date": "2007-08-16T00:00:00", "vocab": "financial", "tf-idf scores": 0.22433294923039138}, {"Date": "2007-08-16T00:00:00", "vocab": "credit", "tf-idf scores": 0.19259123627869298}, {"Date": "2007-08-16T00:00:00", "vocab": "market", "tf-idf scores": 0.18762182082465464}, {"Date": "2007-08-16T00:00:00", "vocab": "july", "tf-idf scores": 0.18538272280678075}, {"Date": "2007-08-16T00:00:00", "vocab": "recent", "tf-idf scores": 0.1737349054465375}, {"Date": "2007-08-16T00:00:00", "vocab": "august", "tf-idf scores": 0.17021334698167992}, {"Date": "2007-08-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.1395973332438893}, {"Date": "2007-08-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.13899250699221885}, {"Date": "2007-08-16T00:00:00", "vocab": "markets", "tf-idf scores": 0.1390344037718334}, {"Date": "2007-08-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13200854623644767}, {"Date": "2007-08-10T00:00:00", "vocab": "financial", "tf-idf scores": 0.22428449500950645}, {"Date": "2007-08-10T00:00:00", "vocab": "credit", "tf-idf scores": 0.19261931508885258}, {"Date": "2007-08-10T00:00:00", "vocab": "market", "tf-idf scores": 0.18765491010760324}, {"Date": "2007-08-10T00:00:00", "vocab": "july", "tf-idf scores": 0.18534131911016577}, {"Date": "2007-08-10T00:00:00", "vocab": "recent", "tf-idf scores": 0.17376807323322727}, {"Date": "2007-08-10T00:00:00", "vocab": "august", "tf-idf scores": 0.17021447074736695}, {"Date": "2007-08-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.13962570505678068}, {"Date": "2007-08-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.13903997828235018}, {"Date": "2007-08-10T00:00:00", "vocab": "markets", "tf-idf scores": 0.1389874833677185}, {"Date": "2007-08-10T00:00:00", "vocab": "inflation", "tf-idf scores": 0.132069959106083}, {"Date": "2007-08-07T00:00:00", "vocab": "growth", "tf-idf scores": 0.29492896646092615}, {"Date": "2007-08-07T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23490975835608832}, {"Date": "2007-08-07T00:00:00", "vocab": "second", "tf-idf scores": 0.21856444076012127}, {"Date": "2007-08-07T00:00:00", "vocab": "economic", "tf-idf scores": 0.14680712924000944}, {"Date": "2007-08-07T00:00:00", "vocab": "subprime", "tf-idf scores": 0.14064304663051455}, {"Date": "2007-08-07T00:00:00", "vocab": "credit", "tf-idf scores": 0.12723278186079137}, {"Date": "2007-08-07T00:00:00", "vocab": "likely", "tf-idf scores": 0.12482679485960867}, {"Date": "2007-08-07T00:00:00", "vocab": "moderate", "tf-idf scores": 0.12281828240640442}, {"Date": "2007-08-07T00:00:00", "vocab": "policy", "tf-idf scores": 0.1174726963895351}, {"Date": "2007-08-07T00:00:00", "vocab": "prices", "tf-idf scores": 0.10324392566067388}, {"Date": "2007-06-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3018583207533583}, {"Date": "2007-06-28T00:00:00", "vocab": "recent", "tf-idf scores": 0.217581833890793}, {"Date": "2007-06-28T00:00:00", "vocab": "growth", "tf-idf scores": 0.21155110091174445}, {"Date": "2007-06-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.16150722160247813}, {"Date": "2007-06-28T00:00:00", "vocab": "core", "tf-idf scores": 0.15873178248391592}, {"Date": "2007-06-28T00:00:00", "vocab": "pace", "tf-idf scores": 0.15449069231168702}, {"Date": "2007-06-28T00:00:00", "vocab": "april", "tf-idf scores": 0.14075542619345682}, {"Date": "2007-06-28T00:00:00", "vocab": "moderate", "tf-idf scores": 0.13309230236997963}, {"Date": "2007-06-28T00:00:00", "vocab": "spending", "tf-idf scores": 0.11940114962183135}, {"Date": "2007-06-28T00:00:00", "vocab": "fail", "tf-idf scores": 0.11918374906688901}, {"Date": "2007-05-09T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23553679999688373}, {"Date": "2007-05-09T00:00:00", "vocab": "economic", "tf-idf scores": 0.18851955013915977}, {"Date": "2007-05-09T00:00:00", "vocab": "growth", "tf-idf scores": 0.16561936924239595}, {"Date": "2007-05-09T00:00:00", "vocab": "pace", "tf-idf scores": 0.14918212429983166}, {"Date": "2007-05-09T00:00:00", "vocab": "predominant", "tf-idf scores": 0.1438876933064485}, {"Date": "2007-05-09T00:00:00", "vocab": "appeared", "tf-idf scores": 0.13408818929153174}, {"Date": "2007-05-09T00:00:00", "vocab": "fail", "tf-idf scores": 0.13321457707981177}, {"Date": "2007-05-09T00:00:00", "vocab": "march", "tf-idf scores": 0.13228514316990198}, {"Date": "2007-05-09T00:00:00", "vocab": "recent", "tf-idf scores": 0.12567755669818306}, {"Date": "2007-05-09T00:00:00", "vocab": "moderate", "tf-idf scores": 0.12263797489540554}, {"Date": "2007-03-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2807693612155067}, {"Date": "2007-03-21T00:00:00", "vocab": "recent", "tf-idf scores": 0.2147217318793971}, {"Date": "2007-03-21T00:00:00", "vocab": "growth", "tf-idf scores": 0.18250082592181996}, {"Date": "2007-03-21T00:00:00", "vocab": "subprime", "tf-idf scores": 0.15825908292631374}, {"Date": "2007-03-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.14868255393627675}, {"Date": "2007-03-21T00:00:00", "vocab": "investment", "tf-idf scores": 0.13272250802236626}, {"Date": "2007-03-21T00:00:00", "vocab": "spending", "tf-idf scores": 0.13219362932652912}, {"Date": "2007-03-21T00:00:00", "vocab": "likely", "tf-idf scores": 0.12390726338917948}, {"Date": "2007-03-21T00:00:00", "vocab": "pace", "tf-idf scores": 0.12389656353564207}, {"Date": "2007-03-21T00:00:00", "vocab": "governors", "tf-idf scores": 0.12067150967013601}, {"Date": "2007-01-31T00:00:00", "vocab": "shall", "tf-idf scores": 0.2866073772432263}, {"Date": "2007-01-31T00:00:00", "vocab": "foreign", "tf-idf scores": 0.23394855194500278}, {"Date": "2007-01-31T00:00:00", "vocab": "currency", "tf-idf scores": 0.19651124472151618}, {"Date": "2007-01-31T00:00:00", "vocab": "market", "tf-idf scores": 0.16194912338959463}, {"Date": "2007-01-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15300138801976632}, {"Date": "2007-01-31T00:00:00", "vocab": "open", "tf-idf scores": 0.14915956235512146}, {"Date": "2007-01-31T00:00:00", "vocab": "growth", "tf-idf scores": 0.14006342418559764}, {"Date": "2007-01-31T00:00:00", "vocab": "bank", "tf-idf scores": 0.13556541860340865}, {"Date": "2007-01-31T00:00:00", "vocab": "chairman", "tf-idf scores": 0.1274377305633874}, {"Date": "2007-01-31T00:00:00", "vocab": "accounts", "tf-idf scores": 0.126584876622613}, {"Date": "2006-12-12T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22705080085104348}, {"Date": "2006-12-12T00:00:00", "vocab": "growth", "tf-idf scores": 0.19425847564872917}, {"Date": "2006-12-12T00:00:00", "vocab": "economic", "tf-idf scores": 0.18505766266883494}, {"Date": "2006-12-12T00:00:00", "vocab": "october", "tf-idf scores": 0.1772339932684155}, {"Date": "2006-12-12T00:00:00", "vocab": "spending", "tf-idf scores": 0.12616177427151581}, {"Date": "2006-12-12T00:00:00", "vocab": "activity", "tf-idf scores": 0.11771852646467944}, {"Date": "2006-12-12T00:00:00", "vocab": "remained", "tf-idf scores": 0.11772131845414223}, {"Date": "2006-12-12T00:00:00", "vocab": "core", "tf-idf scores": 0.11771718607242135}, {"Date": "2006-12-12T00:00:00", "vocab": "moderate", "tf-idf scores": 0.11257638764826572}, {"Date": "2006-12-12T00:00:00", "vocab": "recent", "tf-idf scores": 0.10938769196794079}, {"Date": "2006-10-25T00:00:00", "vocab": "inflation", "tf-idf scores": 0.26814409775757314}, {"Date": "2006-10-25T00:00:00", "vocab": "growth", "tf-idf scores": 0.1795303540123502}, {"Date": "2006-10-25T00:00:00", "vocab": "remained", "tf-idf scores": 0.15441098719189353}, {"Date": "2006-10-25T00:00:00", "vocab": "september", "tf-idf scores": 0.1516626695269555}, {"Date": "2006-10-25T00:00:00", "vocab": "prices", "tf-idf scores": 0.1469400510081385}, {"Date": "2006-10-25T00:00:00", "vocab": "economic", "tf-idf scores": 0.14630581804074247}, {"Date": "2006-10-25T00:00:00", "vocab": "continued", "tf-idf scores": 0.13817696430616025}, {"Date": "2006-10-25T00:00:00", "vocab": "core", "tf-idf scores": 0.1312365869514141}, {"Date": "2006-10-25T00:00:00", "vocab": "likely", "tf-idf scores": 0.12188880182944928}, {"Date": "2006-10-25T00:00:00", "vocab": "recent", "tf-idf scores": 0.12192775077697103}, {"Date": "2006-09-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23608651334739672}, {"Date": "2006-09-20T00:00:00", "vocab": "growth", "tf-idf scores": 0.18971503723841396}, {"Date": "2006-09-20T00:00:00", "vocab": "recent", "tf-idf scores": 0.16050450696282992}, {"Date": "2006-09-20T00:00:00", "vocab": "prices", "tf-idf scores": 0.15176787937100583}, {"Date": "2006-09-20T00:00:00", "vocab": "july", "tf-idf scores": 0.14399411064037348}, {"Date": "2006-09-20T00:00:00", "vocab": "august", "tf-idf scores": 0.14239017939507556}, {"Date": "2006-09-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.14168290610510514}, {"Date": "2006-09-20T00:00:00", "vocab": "energy", "tf-idf scores": 0.13333398956729836}, {"Date": "2006-09-20T00:00:00", "vocab": "increases", "tf-idf scores": 0.1254318220379707}, {"Date": "2006-09-20T00:00:00", "vocab": "pace", "tf-idf scores": 0.12277869292721663}, {"Date": "2006-08-08T00:00:00", "vocab": "inflation", "tf-idf scores": 0.26566980079493174}, {"Date": "2006-08-08T00:00:00", "vocab": "growth", "tf-idf scores": 0.21164529929010342}, {"Date": "2006-08-08T00:00:00", "vocab": "second", "tf-idf scores": 0.1887898653807505}, {"Date": "2006-08-08T00:00:00", "vocab": "june", "tf-idf scores": 0.18518985014217965}, {"Date": "2006-08-08T00:00:00", "vocab": "prices", "tf-idf scores": 0.1839925341560209}, {"Date": "2006-08-08T00:00:00", "vocab": "continued", "tf-idf scores": 0.16494879030675788}, {"Date": "2006-08-08T00:00:00", "vocab": "energy", "tf-idf scores": 0.13861809706206565}, {"Date": "2006-08-08T00:00:00", "vocab": "quarter", "tf-idf scores": 0.11961532687094421}, {"Date": "2006-08-08T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11908455219511185}, {"Date": "2006-08-08T00:00:00", "vocab": "economic", "tf-idf scores": 0.11908530906039594}, {"Date": "2006-06-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.29093096961968623}, {"Date": "2006-06-29T00:00:00", "vocab": "growth", "tf-idf scores": 0.24209740414020106}, {"Date": "2006-06-29T00:00:00", "vocab": "prices", "tf-idf scores": 0.1669599938332271}, {"Date": "2006-06-29T00:00:00", "vocab": "quarter", "tf-idf scores": 0.15030371246509747}, {"Date": "2006-06-29T00:00:00", "vocab": "core", "tf-idf scores": 0.125313432456366}, {"Date": "2006-06-29T00:00:00", "vocab": "tendency", "tf-idf scores": 0.12226345680825088}, {"Date": "2006-06-29T00:00:00", "vocab": "april", "tf-idf scores": 0.1212772116560111}, {"Date": "2006-06-29T00:00:00", "vocab": "energy", "tf-idf scores": 0.10899919364779231}, {"Date": "2006-06-29T00:00:00", "vocab": "second", "tf-idf scores": 0.10470668944852149}, {"Date": "2006-06-29T00:00:00", "vocab": "consumer", "tf-idf scores": 0.09976627767102683}, {"Date": "2006-05-10T00:00:00", "vocab": "march", "tf-idf scores": 0.2716949566347584}, {"Date": "2006-05-10T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2457339295218955}, {"Date": "2006-05-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.23912832079260693}, {"Date": "2006-05-10T00:00:00", "vocab": "prices", "tf-idf scores": 0.17739448766396818}, {"Date": "2006-05-10T00:00:00", "vocab": "quarter", "tf-idf scores": 0.1388736582071302}, {"Date": "2006-05-10T00:00:00", "vocab": "pace", "tf-idf scores": 0.13058573790926792}, {"Date": "2006-05-10T00:00:00", "vocab": "energy", "tf-idf scores": 0.12392584674955615}, {"Date": "2006-05-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.12292241502152185}, {"Date": "2006-05-10T00:00:00", "vocab": "spending", "tf-idf scores": 0.11518699740337641}, {"Date": "2006-05-10T00:00:00", "vocab": "firming", "tf-idf scores": 0.10846751211221778}, {"Date": "2006-03-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.19015971533764348}, {"Date": "2006-03-28T00:00:00", "vocab": "prices", "tf-idf scores": 0.18270022981317816}, {"Date": "2006-03-28T00:00:00", "vocab": "february", "tf-idf scores": 0.17229392557829573}, {"Date": "2006-03-28T00:00:00", "vocab": "growth", "tf-idf scores": 0.1660315812253958}, {"Date": "2006-03-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.15706454528719382}, {"Date": "2006-03-28T00:00:00", "vocab": "january", "tf-idf scores": 0.14426141275894636}, {"Date": "2006-03-28T00:00:00", "vocab": "increases", "tf-idf scores": 0.1352150182072278}, {"Date": "2006-03-28T00:00:00", "vocab": "appeared", "tf-idf scores": 0.13283371345070752}, {"Date": "2006-03-28T00:00:00", "vocab": "market", "tf-idf scores": 0.13227483624495917}, {"Date": "2006-03-28T00:00:00", "vocab": "fourth", "tf-idf scores": 0.11860206498491266}, {"Date": "2006-01-31T00:00:00", "vocab": "shall", "tf-idf scores": 0.2982550067431105}, {"Date": "2006-01-31T00:00:00", "vocab": "foreign", "tf-idf scores": 0.24731174801878245}, {"Date": "2006-01-31T00:00:00", "vocab": "currency", "tf-idf scores": 0.2120873254137147}, {"Date": "2006-01-31T00:00:00", "vocab": "market", "tf-idf scores": 0.16185111344406086}, {"Date": "2006-01-31T00:00:00", "vocab": "chairman", "tf-idf scores": 0.1592205430229242}, {"Date": "2006-01-31T00:00:00", "vocab": "open", "tf-idf scores": 0.15351763469583868}, {"Date": "2006-01-31T00:00:00", "vocab": "bank", "tf-idf scores": 0.1354874532681943}, {"Date": "2006-01-31T00:00:00", "vocab": "new", "tf-idf scores": 0.11690917540782048}, {"Date": "2006-01-31T00:00:00", "vocab": "york", "tf-idf scores": 0.11587485518103939}, {"Date": "2006-01-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1079262084034153}, {"Date": "2005-12-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22753921007446096}, {"Date": "2005-12-13T00:00:00", "vocab": "policy", "tf-idf scores": 0.200245030688445}, {"Date": "2005-12-13T00:00:00", "vocab": "energy", "tf-idf scores": 0.17447664544693378}, {"Date": "2005-12-13T00:00:00", "vocab": "growth", "tf-idf scores": 0.1737145087778555}, {"Date": "2005-12-13T00:00:00", "vocab": "prices", "tf-idf scores": 0.17370693794726433}, {"Date": "2005-12-13T00:00:00", "vocab": "november", "tf-idf scores": 0.12930352456760394}, {"Date": "2005-12-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.12751039950229323}, {"Date": "2005-12-13T00:00:00", "vocab": "firming", "tf-idf scores": 0.1124809460087083}, {"Date": "2005-12-13T00:00:00", "vocab": "market", "tf-idf scores": 0.10924943701289626}, {"Date": "2005-12-13T00:00:00", "vocab": "meeting", "tf-idf scores": 0.10928660336907685}, {"Date": "2005-11-01T00:00:00", "vocab": "hurricane", "tf-idf scores": 0.25544347435766657}, {"Date": "2005-11-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23966364674781157}, {"Date": "2005-11-01T00:00:00", "vocab": "hurricanes", "tf-idf scores": 0.18515050826345963}, {"Date": "2005-11-01T00:00:00", "vocab": "rebuilding", "tf-idf scores": 0.1794031601707584}, {"Date": "2005-11-01T00:00:00", "vocab": "remained", "tf-idf scores": 0.17117230857313334}, {"Date": "2005-11-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.145518487404992}, {"Date": "2005-11-01T00:00:00", "vocab": "growth", "tf-idf scores": 0.1375418545957537}, {"Date": "2005-11-01T00:00:00", "vocab": "policy", "tf-idf scores": 0.13696104863752984}, {"Date": "2005-11-01T00:00:00", "vocab": "energy", "tf-idf scores": 0.12949335338767348}, {"Date": "2005-11-01T00:00:00", "vocab": "price", "tf-idf scores": 0.1284621808760402}, {"Date": "2005-09-20T00:00:00", "vocab": "hurricane", "tf-idf scores": 0.3536597565372885}, {"Date": "2005-09-20T00:00:00", "vocab": "gulf", "tf-idf scores": 0.2003093873125467}, {"Date": "2005-09-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.19822924189499447}, {"Date": "2005-09-20T00:00:00", "vocab": "katrina", "tf-idf scores": 0.18412683686335135}, {"Date": "2005-09-20T00:00:00", "vocab": "probably", "tf-idf scores": 0.14951136087680816}, {"Date": "2005-09-20T00:00:00", "vocab": "coast", "tf-idf scores": 0.14625827397715857}, {"Date": "2005-09-20T00:00:00", "vocab": "energy", "tf-idf scores": 0.12180887060712207}, {"Date": "2005-09-20T00:00:00", "vocab": "growth", "tf-idf scores": 0.12126958459679282}, {"Date": "2005-09-20T00:00:00", "vocab": "august", "tf-idf scores": 0.1137573676435054}, {"Date": "2005-09-20T00:00:00", "vocab": "prices", "tf-idf scores": 0.11255325572120777}, {"Date": "2005-08-09T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23817764926025972}, {"Date": "2005-08-09T00:00:00", "vocab": "growth", "tf-idf scores": 0.17681169101823538}, {"Date": "2005-08-09T00:00:00", "vocab": "pace", "tf-idf scores": 0.1760461509428575}, {"Date": "2005-08-09T00:00:00", "vocab": "second", "tf-idf scores": 0.1660444389643362}, {"Date": "2005-08-09T00:00:00", "vocab": "remained", "tf-idf scores": 0.1553389720311842}, {"Date": "2005-08-09T00:00:00", "vocab": "policy", "tf-idf scores": 0.1449925502753094}, {"Date": "2005-08-09T00:00:00", "vocab": "recent", "tf-idf scores": 0.14502577173968237}, {"Date": "2005-08-09T00:00:00", "vocab": "june", "tf-idf scores": 0.1449881235462907}, {"Date": "2005-08-09T00:00:00", "vocab": "core", "tf-idf scores": 0.1338303759301363}, {"Date": "2005-08-09T00:00:00", "vocab": "continued", "tf-idf scores": 0.11396963747829074}, {"Date": "2005-06-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2471392542871685}, {"Date": "2005-06-30T00:00:00", "vocab": "growth", "tf-idf scores": 0.15517158153828903}, {"Date": "2005-06-30T00:00:00", "vocab": "labor", "tf-idf scores": 0.1544971862105388}, {"Date": "2005-06-30T00:00:00", "vocab": "remained", "tf-idf scores": 0.14673992175455278}, {"Date": "2005-06-30T00:00:00", "vocab": "prices", "tf-idf scores": 0.13967928349398515}, {"Date": "2005-06-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.1390519220792554}, {"Date": "2005-06-30T00:00:00", "vocab": "market", "tf-idf scores": 0.13131446029841337}, {"Date": "2005-06-30T00:00:00", "vocab": "price", "tf-idf scores": 0.13135348563534044}, {"Date": "2005-06-30T00:00:00", "vocab": "recent", "tf-idf scores": 0.13132745318396485}, {"Date": "2005-06-30T00:00:00", "vocab": "governors", "tf-idf scores": 0.12416659549876306}, {"Date": "2005-05-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23304123505463725}, {"Date": "2005-05-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.20982473385547876}, {"Date": "2005-05-03T00:00:00", "vocab": "growth", "tf-idf scores": 0.1950599872170707}, {"Date": "2005-05-03T00:00:00", "vocab": "prices", "tf-idf scores": 0.18731696130910772}, {"Date": "2005-05-03T00:00:00", "vocab": "energy", "tf-idf scores": 0.18024875642732618}, {"Date": "2005-05-03T00:00:00", "vocab": "policy", "tf-idf scores": 0.1476752435007406}, {"Date": "2005-05-03T00:00:00", "vocab": "march", "tf-idf scores": 0.14398015881014092}, {"Date": "2005-05-03T00:00:00", "vocab": "recent", "tf-idf scores": 0.13206447523180392}, {"Date": "2005-05-03T00:00:00", "vocab": "price", "tf-idf scores": 0.12429240754926117}, {"Date": "2005-05-03T00:00:00", "vocab": "remained", "tf-idf scores": 0.11660183573704416}, {"Date": "2005-03-22T00:00:00", "vocab": "inflation", "tf-idf scores": 0.268352905136967}, {"Date": "2005-03-22T00:00:00", "vocab": "policy", "tf-idf scores": 0.17084391305741184}, {"Date": "2005-03-22T00:00:00", "vocab": "prices", "tf-idf scores": 0.16335578190347988}, {"Date": "2005-03-22T00:00:00", "vocab": "growth", "tf-idf scores": 0.15525468367693815}, {"Date": "2005-03-22T00:00:00", "vocab": "labor", "tf-idf scores": 0.1545174484991649}, {"Date": "2005-03-22T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1464068486407808}, {"Date": "2005-03-22T00:00:00", "vocab": "business", "tf-idf scores": 0.12205251641786884}, {"Date": "2005-03-22T00:00:00", "vocab": "economic", "tf-idf scores": 0.12205629681309012}, {"Date": "2005-03-22T00:00:00", "vocab": "price", "tf-idf scores": 0.11386124514498441}, {"Date": "2005-03-22T00:00:00", "vocab": "likely", "tf-idf scores": 0.10577438922172161}, {"Date": "2005-02-02T00:00:00", "vocab": "shall", "tf-idf scores": 0.295554574418275}, {"Date": "2005-02-02T00:00:00", "vocab": "foreign", "tf-idf scores": 0.2552652286852991}, {"Date": "2005-02-02T00:00:00", "vocab": "currency", "tf-idf scores": 0.21079547168224266}, {"Date": "2005-02-02T00:00:00", "vocab": "market", "tf-idf scores": 0.178655666769405}, {"Date": "2005-02-02T00:00:00", "vocab": "open", "tf-idf scores": 0.1514998407158235}, {"Date": "2005-02-02T00:00:00", "vocab": "chairman", "tf-idf scores": 0.14785195613785246}, {"Date": "2005-02-02T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1206792505708619}, {"Date": "2005-02-02T00:00:00", "vocab": "operations", "tf-idf scores": 0.1102586043358103}, {"Date": "2005-02-02T00:00:00", "vocab": "paragraph", "tf-idf scores": 0.1072630263529662}, {"Date": "2005-02-02T00:00:00", "vocab": "fourth", "tf-idf scores": 0.1065019597581818}, {"Date": "2004-12-14T00:00:00", "vocab": "november", "tf-idf scores": 0.1781761709232655}, {"Date": "2004-12-14T00:00:00", "vocab": "recent", "tf-idf scores": 0.17571314073766953}, {"Date": "2004-12-14T00:00:00", "vocab": "growth", "tf-idf scores": 0.16804359119725254}, {"Date": "2004-12-14T00:00:00", "vocab": "inflation", "tf-idf scores": 0.16735125807760806}, {"Date": "2004-12-14T00:00:00", "vocab": "pace", "tf-idf scores": 0.14228132202543317}, {"Date": "2004-12-14T00:00:00", "vocab": "minutes", "tf-idf scores": 0.14111718986954702}, {"Date": "2004-12-14T00:00:00", "vocab": "economic", "tf-idf scores": 0.1338990795004891}, {"Date": "2004-12-14T00:00:00", "vocab": "prices", "tf-idf scores": 0.1260804196136165}, {"Date": "2004-12-14T00:00:00", "vocab": "policy", "tf-idf scores": 0.12551080980927853}, {"Date": "2004-12-14T00:00:00", "vocab": "productivity", "tf-idf scores": 0.12231689379084536}, {"Date": "2004-11-10T00:00:00", "vocab": "pace", "tf-idf scores": 0.17628604181399604}, {"Date": "2004-11-10T00:00:00", "vocab": "members", "tf-idf scores": 0.16800602904034143}, {"Date": "2004-11-10T00:00:00", "vocab": "likely", "tf-idf scores": 0.15674359716812847}, {"Date": "2004-11-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.1371237070476766}, {"Date": "2004-11-10T00:00:00", "vocab": "september", "tf-idf scores": 0.13714868515841125}, {"Date": "2004-11-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.12788955593489135}, {"Date": "2004-11-10T00:00:00", "vocab": "recent", "tf-idf scores": 0.1273586248340821}, {"Date": "2004-11-10T00:00:00", "vocab": "spending", "tf-idf scores": 0.12735768398481578}, {"Date": "2004-11-10T00:00:00", "vocab": "business", "tf-idf scores": 0.11758201094989802}, {"Date": "2004-11-10T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11756083320998423}, {"Date": "2004-09-21T00:00:00", "vocab": "july", "tf-idf scores": 0.2413761841811397}, {"Date": "2004-09-21T00:00:00", "vocab": "growth", "tf-idf scores": 0.16964126246363068}, {"Date": "2004-09-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.16888235353569767}, {"Date": "2004-09-21T00:00:00", "vocab": "prices", "tf-idf scores": 0.13568934225279328}, {"Date": "2004-09-21T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13515305898692528}, {"Date": "2004-09-21T00:00:00", "vocab": "policy", "tf-idf scores": 0.13514512962159936}, {"Date": "2004-09-21T00:00:00", "vocab": "policymakers", "tf-idf scores": 0.13383066470727492}, {"Date": "2004-09-21T00:00:00", "vocab": "labor", "tf-idf scores": 0.12667746743700603}, {"Date": "2004-09-21T00:00:00", "vocab": "pace", "tf-idf scores": 0.1267158263133384}, {"Date": "2004-09-21T00:00:00", "vocab": "spending", "tf-idf scores": 0.11823459729168462}, {"Date": "2004-08-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.2757838531142458}, {"Date": "2004-08-10T00:00:00", "vocab": "members", "tf-idf scores": 0.17627217310342505}, {"Date": "2004-08-10T00:00:00", "vocab": "june", "tf-idf scores": 0.16830133966206023}, {"Date": "2004-08-10T00:00:00", "vocab": "spending", "tf-idf scores": 0.15814809894773604}, {"Date": "2004-08-10T00:00:00", "vocab": "energy", "tf-idf scores": 0.15111339769262597}, {"Date": "2004-08-10T00:00:00", "vocab": "consumer", "tf-idf scores": 0.14977419624690688}, {"Date": "2004-08-10T00:00:00", "vocab": "pace", "tf-idf scores": 0.14977737743292224}, {"Date": "2004-08-10T00:00:00", "vocab": "recent", "tf-idf scores": 0.1498452615342254}, {"Date": "2004-08-10T00:00:00", "vocab": "removed", "tf-idf scores": 0.13563451459549897}, {"Date": "2004-08-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.13315164624833892}, {"Date": "2004-06-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.17023722878201947}, {"Date": "2004-06-30T00:00:00", "vocab": "members", "tf-idf scores": 0.16350988039382125}, {"Date": "2004-06-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.16210445063967757}, {"Date": "2004-06-30T00:00:00", "vocab": "recent", "tf-idf scores": 0.15405519408862695}, {"Date": "2004-06-30T00:00:00", "vocab": "growth", "tf-idf scores": 0.14652111073287624}, {"Date": "2004-06-30T00:00:00", "vocab": "consumer", "tf-idf scores": 0.14589777325831135}, {"Date": "2004-06-30T00:00:00", "vocab": "price", "tf-idf scores": 0.12970492340850656}, {"Date": "2004-06-30T00:00:00", "vocab": "increases", "tf-idf scores": 0.1242360924773972}, {"Date": "2004-06-30T00:00:00", "vocab": "spending", "tf-idf scores": 0.12157844359236966}, {"Date": "2004-06-30T00:00:00", "vocab": "april", "tf-idf scores": 0.11819232471844435}, {"Date": "2004-05-04T00:00:00", "vocab": "march", "tf-idf scores": 0.18516066785221097}, {"Date": "2004-05-04T00:00:00", "vocab": "growth", "tf-idf scores": 0.1783349958131668}, {"Date": "2004-05-04T00:00:00", "vocab": "members", "tf-idf scores": 0.1705416016128028}, {"Date": "2004-05-04T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1437298478104641}, {"Date": "2004-05-04T00:00:00", "vocab": "price", "tf-idf scores": 0.1437062094074577}, {"Date": "2004-05-04T00:00:00", "vocab": "business", "tf-idf scores": 0.1352585483779033}, {"Date": "2004-05-04T00:00:00", "vocab": "quarter", "tf-idf scores": 0.12739002778996755}, {"Date": "2004-05-04T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12680489894534974}, {"Date": "2004-05-04T00:00:00", "vocab": "increases", "tf-idf scores": 0.12103107202768396}, {"Date": "2004-05-04T00:00:00", "vocab": "market", "tf-idf scores": 0.1184355278568845}, {"Date": "2004-03-16T00:00:00", "vocab": "members", "tf-idf scores": 0.23500554487487638}, {"Date": "2004-03-16T00:00:00", "vocab": "january", "tf-idf scores": 0.20527020318492759}, {"Date": "2004-03-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.19059779993475987}, {"Date": "2004-03-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18123143030317565}, {"Date": "2004-03-16T00:00:00", "vocab": "policy", "tf-idf scores": 0.16396638133514138}, {"Date": "2004-03-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.15530066965130246}, {"Date": "2004-03-16T00:00:00", "vocab": "spending", "tf-idf scores": 0.12078609208272566}, {"Date": "2004-03-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.11215473136446634}, {"Date": "2004-03-16T00:00:00", "vocab": "remained", "tf-idf scores": 0.11221780592130634}, {"Date": "2004-03-16T00:00:00", "vocab": "consumer", "tf-idf scores": 0.10354410363488703}, {"Date": "2004-01-28T00:00:00", "vocab": "shall", "tf-idf scores": 0.26711682223309424}, {"Date": "2004-01-28T00:00:00", "vocab": "foreign", "tf-idf scores": 0.21285444907846346}, {"Date": "2004-01-28T00:00:00", "vocab": "currency", "tf-idf scores": 0.18930263194403238}, {"Date": "2004-01-28T00:00:00", "vocab": "members", "tf-idf scores": 0.183560776643646}, {"Date": "2004-01-28T00:00:00", "vocab": "market", "tf-idf scores": 0.15483112571170934}, {"Date": "2004-01-28T00:00:00", "vocab": "york", "tf-idf scores": 0.133046713693785}, {"Date": "2004-01-28T00:00:00", "vocab": "open", "tf-idf scores": 0.12829939272576676}, {"Date": "2004-01-28T00:00:00", "vocab": "fourth", "tf-idf scores": 0.1277167133201072}, {"Date": "2004-01-28T00:00:00", "vocab": "business", "tf-idf scores": 0.1200526845843766}, {"Date": "2004-01-28T00:00:00", "vocab": "bank", "tf-idf scores": 0.11275839484264369}, {"Date": "2003-12-09T00:00:00", "vocab": "members", "tf-idf scores": 0.25363980054618407}, {"Date": "2003-12-09T00:00:00", "vocab": "october", "tf-idf scores": 0.17936541397879802}, {"Date": "2003-12-09T00:00:00", "vocab": "spending", "tf-idf scores": 0.15717447771286708}, {"Date": "2003-12-09T00:00:00", "vocab": "economic", "tf-idf scores": 0.14926916745432936}, {"Date": "2003-12-09T00:00:00", "vocab": "growth", "tf-idf scores": 0.1420938665752953}, {"Date": "2003-12-09T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14145567718426885}, {"Date": "2003-12-09T00:00:00", "vocab": "governors", "tf-idf scores": 0.1263012664627054}, {"Date": "2003-12-09T00:00:00", "vocab": "policy", "tf-idf scores": 0.11786179459399343}, {"Date": "2003-12-09T00:00:00", "vocab": "recent", "tf-idf scores": 0.11790714037951264}, {"Date": "2003-12-09T00:00:00", "vocab": "gains", "tf-idf scores": 0.10906073217808103}, {"Date": "2003-10-28T00:00:00", "vocab": "members", "tf-idf scores": 0.22453340772115765}, {"Date": "2003-10-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.20474296619633048}, {"Date": "2003-10-28T00:00:00", "vocab": "business", "tf-idf scores": 0.18694925651703875}, {"Date": "2003-10-28T00:00:00", "vocab": "august", "tf-idf scores": 0.18454274573120985}, {"Date": "2003-10-28T00:00:00", "vocab": "september", "tf-idf scores": 0.17997257565120545}, {"Date": "2003-10-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14248771389661943}, {"Date": "2003-10-28T00:00:00", "vocab": "recent", "tf-idf scores": 0.12460367340829746}, {"Date": "2003-10-28T00:00:00", "vocab": "growth", "tf-idf scores": 0.11626444492153554}, {"Date": "2003-10-28T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11576392392137741}, {"Date": "2003-10-28T00:00:00", "vocab": "indications", "tf-idf scores": 0.1078839856738117}, {"Date": "2003-09-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.21474304356844565}, {"Date": "2003-09-16T00:00:00", "vocab": "members", "tf-idf scores": 0.2083401683579576}, {"Date": "2003-09-16T00:00:00", "vocab": "business", "tf-idf scores": 0.2065047673867788}, {"Date": "2003-09-16T00:00:00", "vocab": "august", "tf-idf scores": 0.14006010312314077}, {"Date": "2003-09-16T00:00:00", "vocab": "july", "tf-idf scores": 0.12589592539896716}, {"Date": "2003-09-16T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12388709297133696}, {"Date": "2003-09-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.11615064124902172}, {"Date": "2003-09-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11567782390853962}, {"Date": "2003-09-16T00:00:00", "vocab": "tax", "tf-idf scores": 0.10907190671662344}, {"Date": "2003-09-16T00:00:00", "vocab": "final", "tf-idf scores": 0.10899002691131765}, {"Date": "2003-09-15T00:00:00", "vocab": "business", "tf-idf scores": 0.19403418181151974}, {"Date": "2003-09-15T00:00:00", "vocab": "members", "tf-idf scores": 0.1864789528638604}, {"Date": "2003-09-15T00:00:00", "vocab": "june", "tf-idf scores": 0.17248135000345438}, {"Date": "2003-09-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.1663669339633661}, {"Date": "2003-09-15T00:00:00", "vocab": "consumer", "tf-idf scores": 0.14782813703575684}, {"Date": "2003-09-15T00:00:00", "vocab": "disinflation", "tf-idf scores": 0.1332219753406431}, {"Date": "2003-09-15T00:00:00", "vocab": "policy", "tf-idf scores": 0.1294026223027687}, {"Date": "2003-09-15T00:00:00", "vocab": "growth", "tf-idf scores": 0.12067374795024527}, {"Date": "2003-09-15T00:00:00", "vocab": "remained", "tf-idf scores": 0.12013283368888189}, {"Date": "2003-09-15T00:00:00", "vocab": "recent", "tf-idf scores": 0.11095352059261293}, {"Date": "2003-08-12T00:00:00", "vocab": "business", "tf-idf scores": 0.19406026754172517}, {"Date": "2003-08-12T00:00:00", "vocab": "members", "tf-idf scores": 0.18648409121731063}, {"Date": "2003-08-12T00:00:00", "vocab": "june", "tf-idf scores": 0.17251765766589924}, {"Date": "2003-08-12T00:00:00", "vocab": "economic", "tf-idf scores": 0.16633080401509284}, {"Date": "2003-08-12T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1478780030737839}, {"Date": "2003-08-12T00:00:00", "vocab": "disinflation", "tf-idf scores": 0.13318972009142535}, {"Date": "2003-08-12T00:00:00", "vocab": "policy", "tf-idf scores": 0.12935940876134383}, {"Date": "2003-08-12T00:00:00", "vocab": "growth", "tf-idf scores": 0.12071603135442792}, {"Date": "2003-08-12T00:00:00", "vocab": "remained", "tf-idf scores": 0.12013630540047877}, {"Date": "2003-08-12T00:00:00", "vocab": "recent", "tf-idf scores": 0.11087570351852202}, {"Date": "2003-06-25T00:00:00", "vocab": "members", "tf-idf scores": 0.29544925867847455}, {"Date": "2003-06-25T00:00:00", "vocab": "economic", "tf-idf scores": 0.21239392557068842}, {"Date": "2003-06-25T00:00:00", "vocab": "policy", "tf-idf scores": 0.14645649027040167}, {"Date": "2003-06-25T00:00:00", "vocab": "business", "tf-idf scores": 0.12454087551867271}, {"Date": "2003-06-25T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12454417250805955}, {"Date": "2003-06-25T00:00:00", "vocab": "april", "tf-idf scores": 0.12020014190415709}, {"Date": "2003-06-25T00:00:00", "vocab": "growth", "tf-idf scores": 0.11765994207552677}, {"Date": "2003-06-25T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11721575987701187}, {"Date": "2003-06-25T00:00:00", "vocab": "low", "tf-idf scores": 0.11327240189393836}, {"Date": "2003-06-25T00:00:00", "vocab": "continued", "tf-idf scores": 0.10982597230595367}, {"Date": "2003-05-06T00:00:00", "vocab": "economic", "tf-idf scores": 0.2983586604909975}, {"Date": "2003-05-06T00:00:00", "vocab": "members", "tf-idf scores": 0.18059592292995436}, {"Date": "2003-05-06T00:00:00", "vocab": "march", "tf-idf scores": 0.17229903457353407}, {"Date": "2003-05-06T00:00:00", "vocab": "disinflation", "tf-idf scores": 0.1720315280348964}, {"Date": "2003-05-06T00:00:00", "vocab": "growth", "tf-idf scores": 0.14557188693700604}, {"Date": "2003-05-06T00:00:00", "vocab": "business", "tf-idf scores": 0.14493179258180408}, {"Date": "2003-05-06T00:00:00", "vocab": "persisting", "tf-idf scores": 0.12077647524071193}, {"Date": "2003-05-06T00:00:00", "vocab": "activity", "tf-idf scores": 0.11936951018037614}, {"Date": "2003-05-06T00:00:00", "vocab": "iraqi", "tf-idf scores": 0.11494445085443035}, {"Date": "2003-05-06T00:00:00", "vocab": "war", "tf-idf scores": 0.11267228119216448}, {"Date": "2003-04-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.19528739444236243}, {"Date": "2003-04-16T00:00:00", "vocab": "war", "tf-idf scores": 0.1842666807145434}, {"Date": "2003-04-16T00:00:00", "vocab": "iraq", "tf-idf scores": 0.157813289303289}, {"Date": "2003-04-16T00:00:00", "vocab": "members", "tf-idf scores": 0.1501645817660535}, {"Date": "2003-04-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.14014413393291963}, {"Date": "2003-04-16T00:00:00", "vocab": "business", "tf-idf scores": 0.13950196772279427}, {"Date": "2003-04-16T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13021544513469835}, {"Date": "2003-04-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.12095956908195309}, {"Date": "2003-04-16T00:00:00", "vocab": "likely", "tf-idf scores": 0.1209062228557963}, {"Date": "2003-04-16T00:00:00", "vocab": "february", "tf-idf scores": 0.10577880606919679}, {"Date": "2003-04-08T00:00:00", "vocab": "economic", "tf-idf scores": 0.19535756863113993}, {"Date": "2003-04-08T00:00:00", "vocab": "war", "tf-idf scores": 0.18427186340251936}, {"Date": "2003-04-08T00:00:00", "vocab": "iraq", "tf-idf scores": 0.15774351185133453}, {"Date": "2003-04-08T00:00:00", "vocab": "members", "tf-idf scores": 0.1501077537445174}, {"Date": "2003-04-08T00:00:00", "vocab": "growth", "tf-idf scores": 0.14015893551389227}, {"Date": "2003-04-08T00:00:00", "vocab": "business", "tf-idf scores": 0.1395629215592128}, {"Date": "2003-04-08T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13022571578962708}, {"Date": "2003-04-08T00:00:00", "vocab": "continued", "tf-idf scores": 0.12096449656327382}, {"Date": "2003-04-08T00:00:00", "vocab": "likely", "tf-idf scores": 0.12091401758295184}, {"Date": "2003-04-08T00:00:00", "vocab": "february", "tf-idf scores": 0.10578013105377305}, {"Date": "2003-04-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.1953497556853474}, {"Date": "2003-04-01T00:00:00", "vocab": "war", "tf-idf scores": 0.18428181187156906}, {"Date": "2003-04-01T00:00:00", "vocab": "iraq", "tf-idf scores": 0.1577945762278461}, {"Date": "2003-04-01T00:00:00", "vocab": "members", "tf-idf scores": 0.15014616440170697}, {"Date": "2003-04-01T00:00:00", "vocab": "growth", "tf-idf scores": 0.14010671276072909}, {"Date": "2003-04-01T00:00:00", "vocab": "business", "tf-idf scores": 0.13954331255489527}, {"Date": "2003-04-01T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13026822695334128}, {"Date": "2003-04-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.12090410254921186}, {"Date": "2003-04-01T00:00:00", "vocab": "likely", "tf-idf scores": 0.1209408087264643}, {"Date": "2003-04-01T00:00:00", "vocab": "february", "tf-idf scores": 0.10572171322824567}, {"Date": "2003-03-25T00:00:00", "vocab": "economic", "tf-idf scores": 0.19535111402201813}, {"Date": "2003-03-25T00:00:00", "vocab": "war", "tf-idf scores": 0.18431689163929096}, {"Date": "2003-03-25T00:00:00", "vocab": "iraq", "tf-idf scores": 0.1577853924143212}, {"Date": "2003-03-25T00:00:00", "vocab": "members", "tf-idf scores": 0.1501265223348822}, {"Date": "2003-03-25T00:00:00", "vocab": "growth", "tf-idf scores": 0.1401245570433442}, {"Date": "2003-03-25T00:00:00", "vocab": "business", "tf-idf scores": 0.13954101964262355}, {"Date": "2003-03-25T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1302204198724909}, {"Date": "2003-03-25T00:00:00", "vocab": "continued", "tf-idf scores": 0.12092262456500402}, {"Date": "2003-03-25T00:00:00", "vocab": "likely", "tf-idf scores": 0.12098366838731166}, {"Date": "2003-03-25T00:00:00", "vocab": "february", "tf-idf scores": 0.10577764153019323}, {"Date": "2003-03-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.1953420206083532}, {"Date": "2003-03-18T00:00:00", "vocab": "war", "tf-idf scores": 0.18430892534141982}, {"Date": "2003-03-18T00:00:00", "vocab": "iraq", "tf-idf scores": 0.15780839495127494}, {"Date": "2003-03-18T00:00:00", "vocab": "members", "tf-idf scores": 0.15012675423864377}, {"Date": "2003-03-18T00:00:00", "vocab": "growth", "tf-idf scores": 0.14014887578764917}, {"Date": "2003-03-18T00:00:00", "vocab": "business", "tf-idf scores": 0.13954825099944013}, {"Date": "2003-03-18T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13028234118430326}, {"Date": "2003-03-18T00:00:00", "vocab": "continued", "tf-idf scores": 0.12095688152881645}, {"Date": "2003-03-18T00:00:00", "vocab": "likely", "tf-idf scores": 0.12098053228826558}, {"Date": "2003-03-18T00:00:00", "vocab": "february", "tf-idf scores": 0.10575872497905703}, {"Date": "2003-01-29T00:00:00", "vocab": "shall", "tf-idf scores": 0.2632293861630345}, {"Date": "2003-01-29T00:00:00", "vocab": "foreign", "tf-idf scores": 0.22098261539362177}, {"Date": "2003-01-29T00:00:00", "vocab": "market", "tf-idf scores": 0.19839218503047085}, {"Date": "2003-01-29T00:00:00", "vocab": "currency", "tf-idf scores": 0.18901119874871702}, {"Date": "2003-01-29T00:00:00", "vocab": "open", "tf-idf scores": 0.1584972372747649}, {"Date": "2003-01-29T00:00:00", "vocab": "chairman", "tf-idf scores": 0.12775747398675555}, {"Date": "2003-01-29T00:00:00", "vocab": "members", "tf-idf scores": 0.12282253373571089}, {"Date": "2003-01-29T00:00:00", "vocab": "bank", "tf-idf scores": 0.1223343632190927}, {"Date": "2003-01-29T00:00:00", "vocab": "business", "tf-idf scores": 0.1217669052396947}, {"Date": "2003-01-29T00:00:00", "vocab": "operations", "tf-idf scores": 0.11907501811026107}, {"Date": "2002-12-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.28541331219145194}, {"Date": "2002-12-10T00:00:00", "vocab": "members", "tf-idf scores": 0.15420660817982057}, {"Date": "2002-12-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.1330721911301296}, {"Date": "2002-12-10T00:00:00", "vocab": "continued", "tf-idf scores": 0.13248320157265556}, {"Date": "2002-12-10T00:00:00", "vocab": "november", "tf-idf scores": 0.12665038006069315}, {"Date": "2002-12-10T00:00:00", "vocab": "october", "tf-idf scores": 0.12525446850684666}, {"Date": "2002-12-10T00:00:00", "vocab": "improvement", "tf-idf scores": 0.12380341033276117}, {"Date": "2002-12-10T00:00:00", "vocab": "business", "tf-idf scores": 0.12234067142823064}, {"Date": "2002-12-10T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12235902195384948}, {"Date": "2002-12-10T00:00:00", "vocab": "remained", "tf-idf scores": 0.1222902019681352}, {"Date": "2002-11-06T00:00:00", "vocab": "economic", "tf-idf scores": 0.23569621154019588}, {"Date": "2002-11-06T00:00:00", "vocab": "members", "tf-idf scores": 0.22286103046663933}, {"Date": "2002-11-06T00:00:00", "vocab": "market", "tf-idf scores": 0.14737147064907571}, {"Date": "2002-11-06T00:00:00", "vocab": "alternatives", "tf-idf scores": 0.145610410144889}, {"Date": "2002-11-06T00:00:00", "vocab": "study", "tf-idf scores": 0.1373556026705274}, {"Date": "2002-11-06T00:00:00", "vocab": "september", "tf-idf scores": 0.12606725630493376}, {"Date": "2002-11-06T00:00:00", "vocab": "policy", "tf-idf scores": 0.12519162033117484}, {"Date": "2002-11-06T00:00:00", "vocab": "ginnie", "tf-idf scores": 0.11800764976991596}, {"Date": "2002-11-06T00:00:00", "vocab": "business", "tf-idf scores": 0.1178762252786894}, {"Date": "2002-11-06T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11785237821875386}, {"Date": "2002-09-24T00:00:00", "vocab": "economic", "tf-idf scores": 0.19503486432462835}, {"Date": "2002-09-24T00:00:00", "vocab": "growth", "tf-idf scores": 0.1691667904216303}, {"Date": "2002-09-24T00:00:00", "vocab": "august", "tf-idf scores": 0.16697714418809817}, {"Date": "2002-09-24T00:00:00", "vocab": "consumer", "tf-idf scores": 0.15951418685326713}, {"Date": "2002-09-24T00:00:00", "vocab": "spending", "tf-idf scores": 0.15952604630955325}, {"Date": "2002-09-24T00:00:00", "vocab": "business", "tf-idf scores": 0.1418576273542622}, {"Date": "2002-09-24T00:00:00", "vocab": "july", "tf-idf scores": 0.13509281240945775}, {"Date": "2002-09-24T00:00:00", "vocab": "capital", "tf-idf scores": 0.11722359540864657}, {"Date": "2002-09-24T00:00:00", "vocab": "activity", "tf-idf scores": 0.11525170740093835}, {"Date": "2002-09-24T00:00:00", "vocab": "continued", "tf-idf scores": 0.11528419507051871}, {"Date": "2002-08-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.20577426565459295}, {"Date": "2002-08-13T00:00:00", "vocab": "members", "tf-idf scores": 0.1296964565422732}, {"Date": "2002-08-13T00:00:00", "vocab": "growth", "tf-idf scores": 0.12921578659386088}, {"Date": "2002-08-13T00:00:00", "vocab": "business", "tf-idf scores": 0.1286516771915863}, {"Date": "2002-08-13T00:00:00", "vocab": "continued", "tf-idf scores": 0.1200217126230979}, {"Date": "2002-08-13T00:00:00", "vocab": "june", "tf-idf scores": 0.11997826264826923}, {"Date": "2002-08-13T00:00:00", "vocab": "equity", "tf-idf scores": 0.1158986749339022}, {"Date": "2002-08-13T00:00:00", "vocab": "foreseeable", "tf-idf scores": 0.11579685610498403}, {"Date": "2002-08-13T00:00:00", "vocab": "financial", "tf-idf scores": 0.11247790830705667}, {"Date": "2002-08-13T00:00:00", "vocab": "activity", "tf-idf scores": 0.11141935318579678}, {"Date": "2002-06-26T00:00:00", "vocab": "april", "tf-idf scores": 0.2027363356116678}, {"Date": "2002-06-26T00:00:00", "vocab": "members", "tf-idf scores": 0.17261966958818573}, {"Date": "2002-06-26T00:00:00", "vocab": "economic", "tf-idf scores": 0.15403090896145272}, {"Date": "2002-06-26T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15400107085235556}, {"Date": "2002-06-26T00:00:00", "vocab": "growth", "tf-idf scores": 0.13745350900038178}, {"Date": "2002-06-26T00:00:00", "vocab": "spending", "tf-idf scores": 0.12830809748921634}, {"Date": "2002-06-26T00:00:00", "vocab": "strength", "tf-idf scores": 0.11879177751171954}, {"Date": "2002-06-26T00:00:00", "vocab": "forecasts", "tf-idf scores": 0.11183933362323595}, {"Date": "2002-06-26T00:00:00", "vocab": "liquidation", "tf-idf scores": 0.10629972377716222}, {"Date": "2002-06-26T00:00:00", "vocab": "policy", "tf-idf scores": 0.10263784081032365}, {"Date": "2002-05-07T00:00:00", "vocab": "members", "tf-idf scores": 0.18532042569411922}, {"Date": "2002-05-07T00:00:00", "vocab": "economic", "tf-idf scores": 0.15752053282557515}, {"Date": "2002-05-07T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13995527918338482}, {"Date": "2002-05-07T00:00:00", "vocab": "policy", "tf-idf scores": 0.13120861756290997}, {"Date": "2002-05-07T00:00:00", "vocab": "growth", "tf-idf scores": 0.12302228980518036}, {"Date": "2002-05-07T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12247789627532307}, {"Date": "2002-05-07T00:00:00", "vocab": "final", "tf-idf scores": 0.11536008640830282}, {"Date": "2002-05-07T00:00:00", "vocab": "activity", "tf-idf scores": 0.1137588225670148}, {"Date": "2002-05-07T00:00:00", "vocab": "continued", "tf-idf scores": 0.11372147435698968}, {"Date": "2002-05-07T00:00:00", "vocab": "spending", "tf-idf scores": 0.11373183461387917}, {"Date": "2002-03-19T00:00:00", "vocab": "economic", "tf-idf scores": 0.23707097912003808}, {"Date": "2002-03-19T00:00:00", "vocab": "members", "tf-idf scores": 0.1943007033561164}, {"Date": "2002-03-19T00:00:00", "vocab": "business", "tf-idf scores": 0.15552744766618837}, {"Date": "2002-03-19T00:00:00", "vocab": "inventory", "tf-idf scores": 0.12987271393197047}, {"Date": "2002-03-19T00:00:00", "vocab": "february", "tf-idf scores": 0.12631028453950965}, {"Date": "2002-03-19T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12594538075870113}, {"Date": "2002-03-19T00:00:00", "vocab": "outlook", "tf-idf scores": 0.11910207567917765}, {"Date": "2002-03-19T00:00:00", "vocab": "policy", "tf-idf scores": 0.11849258758891622}, {"Date": "2002-03-19T00:00:00", "vocab": "january", "tf-idf scores": 0.11748042326864522}, {"Date": "2002-03-19T00:00:00", "vocab": "growth", "tf-idf scores": 0.11165074759707803}, {"Date": "2002-01-30T00:00:00", "vocab": "shall", "tf-idf scores": 0.27258365154359643}, {"Date": "2002-01-30T00:00:00", "vocab": "foreign", "tf-idf scores": 0.25906113773173883}, {"Date": "2002-01-30T00:00:00", "vocab": "currency", "tf-idf scores": 0.21845330259604198}, {"Date": "2002-01-30T00:00:00", "vocab": "market", "tf-idf scores": 0.19654970790759851}, {"Date": "2002-01-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.14742813935845586}, {"Date": "2002-01-30T00:00:00", "vocab": "open", "tf-idf scores": 0.14353816157563898}, {"Date": "2002-01-30T00:00:00", "vocab": "operations", "tf-idf scores": 0.11794711352423198}, {"Date": "2002-01-30T00:00:00", "vocab": "bank", "tf-idf scores": 0.11663336403680864}, {"Date": "2002-01-30T00:00:00", "vocab": "chairman", "tf-idf scores": 0.1054476503517688}, {"Date": "2002-01-30T00:00:00", "vocab": "business", "tf-idf scores": 0.10278121253331804}, {"Date": "2001-12-11T00:00:00", "vocab": "economic", "tf-idf scores": 0.2544209877129993}, {"Date": "2001-12-11T00:00:00", "vocab": "members", "tf-idf scores": 0.18667454765493216}, {"Date": "2001-12-11T00:00:00", "vocab": "september", "tf-idf scores": 0.13188101449825088}, {"Date": "2001-12-11T00:00:00", "vocab": "terrorist", "tf-idf scores": 0.13079349266720203}, {"Date": "2001-12-11T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1233867432375572}, {"Date": "2001-12-11T00:00:00", "vocab": "remained", "tf-idf scores": 0.1233423013780786}, {"Date": "2001-12-11T00:00:00", "vocab": "october", "tf-idf scores": 0.10829569836887232}, {"Date": "2001-12-11T00:00:00", "vocab": "business", "tf-idf scores": 0.10799657154961766}, {"Date": "2001-12-11T00:00:00", "vocab": "easing", "tf-idf scores": 0.10317457858943747}, {"Date": "2001-12-11T00:00:00", "vocab": "governors", "tf-idf scores": 0.10141355067534222}, {"Date": "2001-11-06T00:00:00", "vocab": "members", "tf-idf scores": 0.181409227851225}, {"Date": "2001-11-06T00:00:00", "vocab": "economic", "tf-idf scores": 0.1635026052568567}, {"Date": "2001-11-06T00:00:00", "vocab": "weakness", "tf-idf scores": 0.15251244081960516}, {"Date": "2001-11-06T00:00:00", "vocab": "terrorist", "tf-idf scores": 0.13866350572165811}, {"Date": "2001-11-06T00:00:00", "vocab": "business", "tf-idf scores": 0.13081025522930423}, {"Date": "2001-11-06T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1308142528977719}, {"Date": "2001-11-06T00:00:00", "vocab": "september", "tf-idf scores": 0.12716906036055234}, {"Date": "2001-11-06T00:00:00", "vocab": "economy", "tf-idf scores": 0.1174765630151555}, {"Date": "2001-11-06T00:00:00", "vocab": "activity", "tf-idf scores": 0.11449487322514133}, {"Date": "2001-11-06T00:00:00", "vocab": "policy", "tf-idf scores": 0.11452103838036812}, {"Date": "2001-10-02T00:00:00", "vocab": "terrorist", "tf-idf scores": 0.37852527238886163}, {"Date": "2001-10-02T00:00:00", "vocab": "september", "tf-idf scores": 0.19220625772362887}, {"Date": "2001-10-02T00:00:00", "vocab": "business", "tf-idf scores": 0.151094503113864}, {"Date": "2001-10-02T00:00:00", "vocab": "consumer", "tf-idf scores": 0.14424439250667884}, {"Date": "2001-10-02T00:00:00", "vocab": "economic", "tf-idf scores": 0.1373395417271633}, {"Date": "2001-10-02T00:00:00", "vocab": "august", "tf-idf scores": 0.12935354622179374}, {"Date": "2001-10-02T00:00:00", "vocab": "attacks", "tf-idf scores": 0.1136162458135472}, {"Date": "2001-10-02T00:00:00", "vocab": "spending", "tf-idf scores": 0.1098863403782044}, {"Date": "2001-10-02T00:00:00", "vocab": "july", "tf-idf scores": 0.10466399144964145}, {"Date": "2001-10-02T00:00:00", "vocab": "weakness", "tf-idf scores": 0.10060862485324194}, {"Date": "2001-09-17T00:00:00", "vocab": "business", "tf-idf scores": 0.18756991100692083}, {"Date": "2001-09-17T00:00:00", "vocab": "members", "tf-idf scores": 0.18131933082233795}, {"Date": "2001-09-17T00:00:00", "vocab": "growth", "tf-idf scores": 0.15705322119624893}, {"Date": "2001-09-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.14847455611678945}, {"Date": "2001-09-17T00:00:00", "vocab": "weakness", "tf-idf scores": 0.13538601257388153}, {"Date": "2001-09-17T00:00:00", "vocab": "governors", "tf-idf scores": 0.12570056228723622}, {"Date": "2001-09-17T00:00:00", "vocab": "votes", "tf-idf scores": 0.11783141507803599}, {"Date": "2001-09-17T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11724425942926676}, {"Date": "2001-09-17T00:00:00", "vocab": "recent", "tf-idf scores": 0.11730215885219439}, {"Date": "2001-09-17T00:00:00", "vocab": "second", "tf-idf scores": 0.11632894893870964}, {"Date": "2001-09-13T00:00:00", "vocab": "business", "tf-idf scores": 0.18758486505904262}, {"Date": "2001-09-13T00:00:00", "vocab": "members", "tf-idf scores": 0.1813877292552833}, {"Date": "2001-09-13T00:00:00", "vocab": "growth", "tf-idf scores": 0.1570251289319285}, {"Date": "2001-09-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.1485629971208445}, {"Date": "2001-09-13T00:00:00", "vocab": "weakness", "tf-idf scores": 0.13539190768429923}, {"Date": "2001-09-13T00:00:00", "vocab": "governors", "tf-idf scores": 0.12567592536122013}, {"Date": "2001-09-13T00:00:00", "vocab": "votes", "tf-idf scores": 0.11778968097037305}, {"Date": "2001-09-13T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11729898825755566}, {"Date": "2001-09-13T00:00:00", "vocab": "recent", "tf-idf scores": 0.11726148367918035}, {"Date": "2001-09-13T00:00:00", "vocab": "second", "tf-idf scores": 0.11630977426605305}, {"Date": "2001-08-21T00:00:00", "vocab": "business", "tf-idf scores": 0.1876123750447971}, {"Date": "2001-08-21T00:00:00", "vocab": "members", "tf-idf scores": 0.18134119987100816}, {"Date": "2001-08-21T00:00:00", "vocab": "growth", "tf-idf scores": 0.15702554043213385}, {"Date": "2001-08-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.14850308778922724}, {"Date": "2001-08-21T00:00:00", "vocab": "weakness", "tf-idf scores": 0.13539123276356518}, {"Date": "2001-08-21T00:00:00", "vocab": "governors", "tf-idf scores": 0.12562277468120742}, {"Date": "2001-08-21T00:00:00", "vocab": "votes", "tf-idf scores": 0.1177927880547721}, {"Date": "2001-08-21T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11723065708762956}, {"Date": "2001-08-21T00:00:00", "vocab": "recent", "tf-idf scores": 0.11727487639658982}, {"Date": "2001-08-21T00:00:00", "vocab": "second", "tf-idf scores": 0.11628925898743209}, {"Date": "2001-06-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.23202168012664903}, {"Date": "2001-06-27T00:00:00", "vocab": "growth", "tf-idf scores": 0.1728332922492675}, {"Date": "2001-06-27T00:00:00", "vocab": "easing", "tf-idf scores": 0.1602546962503417}, {"Date": "2001-06-27T00:00:00", "vocab": "members", "tf-idf scores": 0.15853851163829658}, {"Date": "2001-06-27T00:00:00", "vocab": "activity", "tf-idf scores": 0.15714555707536668}, {"Date": "2001-06-27T00:00:00", "vocab": "business", "tf-idf scores": 0.14967695192701594}, {"Date": "2001-06-27T00:00:00", "vocab": "weakness", "tf-idf scores": 0.13956525949355084}, {"Date": "2001-06-27T00:00:00", "vocab": "april", "tf-idf scores": 0.1364449705663795}, {"Date": "2001-06-27T00:00:00", "vocab": "continued", "tf-idf scores": 0.1346810260324572}, {"Date": "2001-06-27T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11970771840614419}, {"Date": "2001-05-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.2096534149613295}, {"Date": "2001-05-15T00:00:00", "vocab": "members", "tf-idf scores": 0.1963932474695607}, {"Date": "2001-05-15T00:00:00", "vocab": "growth", "tf-idf scores": 0.18051172495672027}, {"Date": "2001-05-15T00:00:00", "vocab": "easing", "tf-idf scores": 0.18034241815819252}, {"Date": "2001-05-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14974452051787954}, {"Date": "2001-05-15T00:00:00", "vocab": "business", "tf-idf scores": 0.134789540405196}, {"Date": "2001-05-15T00:00:00", "vocab": "march", "tf-idf scores": 0.12612157235917262}, {"Date": "2001-05-15T00:00:00", "vocab": "year", "tf-idf scores": 0.12039148133617193}, {"Date": "2001-05-15T00:00:00", "vocab": "spending", "tf-idf scores": 0.10487556432342703}, {"Date": "2001-05-15T00:00:00", "vocab": "weakness", "tf-idf scores": 0.0998006047996561}, {"Date": "2001-04-18T00:00:00", "vocab": "january", "tf-idf scores": 0.2242944552165247}, {"Date": "2001-04-18T00:00:00", "vocab": "members", "tf-idf scores": 0.20608005004278468}, {"Date": "2001-04-18T00:00:00", "vocab": "business", "tf-idf scores": 0.19644554882581822}, {"Date": "2001-04-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.18856800526488238}, {"Date": "2001-04-18T00:00:00", "vocab": "growth", "tf-idf scores": 0.17362946409219573}, {"Date": "2001-04-18T00:00:00", "vocab": "consumer", "tf-idf scores": 0.16504164759233197}, {"Date": "2001-04-18T00:00:00", "vocab": "easing", "tf-idf scores": 0.14725291212332023}, {"Date": "2001-04-18T00:00:00", "vocab": "expansion", "tf-idf scores": 0.13853297176665583}, {"Date": "2001-04-18T00:00:00", "vocab": "relatively", "tf-idf scores": 0.12691874652649995}, {"Date": "2001-04-18T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1178408475304414}, {"Date": "2001-04-11T00:00:00", "vocab": "january", "tf-idf scores": 0.22434238503051646}, {"Date": "2001-04-11T00:00:00", "vocab": "members", "tf-idf scores": 0.2060830857386849}, {"Date": "2001-04-11T00:00:00", "vocab": "business", "tf-idf scores": 0.19644800637619297}, {"Date": "2001-04-11T00:00:00", "vocab": "economic", "tf-idf scores": 0.18858080940758457}, {"Date": "2001-04-11T00:00:00", "vocab": "growth", "tf-idf scores": 0.17362821093778288}, {"Date": "2001-04-11T00:00:00", "vocab": "consumer", "tf-idf scores": 0.16499494557885716}, {"Date": "2001-04-11T00:00:00", "vocab": "easing", "tf-idf scores": 0.14721090937665335}, {"Date": "2001-04-11T00:00:00", "vocab": "expansion", "tf-idf scores": 0.1385645467639997}, {"Date": "2001-04-11T00:00:00", "vocab": "relatively", "tf-idf scores": 0.12694934769987232}, {"Date": "2001-04-11T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1178426314522858}, {"Date": "2001-03-20T00:00:00", "vocab": "january", "tf-idf scores": 0.22429432964819632}, {"Date": "2001-03-20T00:00:00", "vocab": "members", "tf-idf scores": 0.20607128005102604}, {"Date": "2001-03-20T00:00:00", "vocab": "business", "tf-idf scores": 0.19643966331400536}, {"Date": "2001-03-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.18855295563489177}, {"Date": "2001-03-20T00:00:00", "vocab": "growth", "tf-idf scores": 0.1736328991557389}, {"Date": "2001-03-20T00:00:00", "vocab": "consumer", "tf-idf scores": 0.16496604481584523}, {"Date": "2001-03-20T00:00:00", "vocab": "easing", "tf-idf scores": 0.14725330829213987}, {"Date": "2001-03-20T00:00:00", "vocab": "expansion", "tf-idf scores": 0.13853910149036722}, {"Date": "2001-03-20T00:00:00", "vocab": "relatively", "tf-idf scores": 0.12696385960805298}, {"Date": "2001-03-20T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11786002383334036}, {"Date": "2001-01-31T00:00:00", "vocab": "shall", "tf-idf scores": 0.23786788895197922}, {"Date": "2001-01-31T00:00:00", "vocab": "foreign", "tf-idf scores": 0.2182659121539731}, {"Date": "2001-01-31T00:00:00", "vocab": "market", "tf-idf scores": 0.19093868342716824}, {"Date": "2001-01-31T00:00:00", "vocab": "currency", "tf-idf scores": 0.1837620184449126}, {"Date": "2001-01-31T00:00:00", "vocab": "members", "tf-idf scores": 0.16117170745911344}, {"Date": "2001-01-31T00:00:00", "vocab": "open", "tf-idf scores": 0.13312421630640872}, {"Date": "2001-01-31T00:00:00", "vocab": "operations", "tf-idf scores": 0.12352154413434265}, {"Date": "2001-01-31T00:00:00", "vocab": "business", "tf-idf scores": 0.12078047282507197}, {"Date": "2001-01-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.12078820926831452}, {"Date": "2001-01-31T00:00:00", "vocab": "securities", "tf-idf scores": 0.11446255108433435}, {"Date": "2001-01-03T00:00:00", "vocab": "growth", "tf-idf scores": 0.25286068169887943}, {"Date": "2001-01-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.21682768696196086}, {"Date": "2001-01-03T00:00:00", "vocab": "expansion", "tf-idf scores": 0.17267642169320147}, {"Date": "2001-01-03T00:00:00", "vocab": "october", "tf-idf scores": 0.1720115219081924}, {"Date": "2001-01-03T00:00:00", "vocab": "members", "tf-idf scores": 0.1411209863532026}, {"Date": "2001-01-03T00:00:00", "vocab": "business", "tf-idf scores": 0.1258817885031428}, {"Date": "2001-01-03T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12595699143866035}, {"Date": "2001-01-03T00:00:00", "vocab": "november", "tf-idf scores": 0.12419894943845415}, {"Date": "2001-01-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1189527659064734}, {"Date": "2001-01-03T00:00:00", "vocab": "risks", "tf-idf scores": 0.11238056723006287}, {"Date": "2000-12-19T00:00:00", "vocab": "growth", "tf-idf scores": 0.2528864866825924}, {"Date": "2000-12-19T00:00:00", "vocab": "economic", "tf-idf scores": 0.21684972833057423}, {"Date": "2000-12-19T00:00:00", "vocab": "expansion", "tf-idf scores": 0.17263027347029836}, {"Date": "2000-12-19T00:00:00", "vocab": "october", "tf-idf scores": 0.17192763753736406}, {"Date": "2000-12-19T00:00:00", "vocab": "members", "tf-idf scores": 0.14109614091816336}, {"Date": "2000-12-19T00:00:00", "vocab": "business", "tf-idf scores": 0.12588989603857798}, {"Date": "2000-12-19T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12591010695406954}, {"Date": "2000-12-19T00:00:00", "vocab": "november", "tf-idf scores": 0.12410615441271633}, {"Date": "2000-12-19T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11890290640371863}, {"Date": "2000-12-19T00:00:00", "vocab": "risks", "tf-idf scores": 0.11245427324929716}, {"Date": "2000-11-15T00:00:00", "vocab": "growth", "tf-idf scores": 0.2985229556503244}, {"Date": "2000-11-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18769466399917276}, {"Date": "2000-11-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.15647640327929901}, {"Date": "2000-11-15T00:00:00", "vocab": "expansion", "tf-idf scores": 0.1562642790936065}, {"Date": "2000-11-15T00:00:00", "vocab": "members", "tf-idf scores": 0.1498743381677577}, {"Date": "2000-11-15T00:00:00", "vocab": "october", "tf-idf scores": 0.12365574470406956}, {"Date": "2000-11-15T00:00:00", "vocab": "prices", "tf-idf scores": 0.11790219138136002}, {"Date": "2000-11-15T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11737266790247959}, {"Date": "2000-11-15T00:00:00", "vocab": "energy", "tf-idf scores": 0.1105090394743481}, {"Date": "2000-11-15T00:00:00", "vocab": "direction", "tf-idf scores": 0.10205118010115963}, {"Date": "2000-10-03T00:00:00", "vocab": "growth", "tf-idf scores": 0.25346824398454204}, {"Date": "2000-10-03T00:00:00", "vocab": "august", "tf-idf scores": 0.22896468641088294}, {"Date": "2000-10-03T00:00:00", "vocab": "july", "tf-idf scores": 0.17813706823768274}, {"Date": "2000-10-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1589617194593779}, {"Date": "2000-10-03T00:00:00", "vocab": "prices", "tf-idf scores": 0.15017403637553794}, {"Date": "2000-10-03T00:00:00", "vocab": "recent", "tf-idf scores": 0.1495676448576003}, {"Date": "2000-10-03T00:00:00", "vocab": "members", "tf-idf scores": 0.1414141347641014}, {"Date": "2000-10-03T00:00:00", "vocab": "expansion", "tf-idf scores": 0.1318249768251547}, {"Date": "2000-10-03T00:00:00", "vocab": "gains", "tf-idf scores": 0.12968156965050345}, {"Date": "2000-10-03T00:00:00", "vocab": "somewhat", "tf-idf scores": 0.11312876007752647}, {"Date": "2000-08-22T00:00:00", "vocab": "productivity", "tf-idf scores": 0.24442759824768828}, {"Date": "2000-08-22T00:00:00", "vocab": "growth", "tf-idf scores": 0.23671537474222634}, {"Date": "2000-08-22T00:00:00", "vocab": "prices", "tf-idf scores": 0.17569304365899557}, {"Date": "2000-08-22T00:00:00", "vocab": "members", "tf-idf scores": 0.15342077313907765}, {"Date": "2000-08-22T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1521361221166038}, {"Date": "2000-08-22T00:00:00", "vocab": "recent", "tf-idf scores": 0.14451858613214655}, {"Date": "2000-08-22T00:00:00", "vocab": "demand", "tf-idf scores": 0.13985808537665123}, {"Date": "2000-08-22T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13694126160483513}, {"Date": "2000-08-22T00:00:00", "vocab": "labor", "tf-idf scores": 0.11404463108096358}, {"Date": "2000-08-22T00:00:00", "vocab": "june", "tf-idf scores": 0.106485119407165}, {"Date": "2000-06-28T00:00:00", "vocab": "growth", "tf-idf scores": 0.22176609770550793}, {"Date": "2000-06-28T00:00:00", "vocab": "members", "tf-idf scores": 0.17498998574413308}, {"Date": "2000-06-28T00:00:00", "vocab": "consumer", "tf-idf scores": 0.15768623203871743}, {"Date": "2000-06-28T00:00:00", "vocab": "prices", "tf-idf scores": 0.1505054288278458}, {"Date": "2000-06-28T00:00:00", "vocab": "ranges", "tf-idf scores": 0.1452784650448824}, {"Date": "2000-06-28T00:00:00", "vocab": "expansion", "tf-idf scores": 0.139052310236735}, {"Date": "2000-06-28T00:00:00", "vocab": "april", "tf-idf scores": 0.1293970585900123}, {"Date": "2000-06-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.12622384530569716}, {"Date": "2000-06-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11827682811150039}, {"Date": "2000-06-28T00:00:00", "vocab": "indications", "tf-idf scores": 0.11147459219979616}, {"Date": "2000-05-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.22549035949944443}, {"Date": "2000-05-16T00:00:00", "vocab": "demand", "tf-idf scores": 0.17441069421206232}, {"Date": "2000-05-16T00:00:00", "vocab": "members", "tf-idf scores": 0.15395855590728671}, {"Date": "2000-05-16T00:00:00", "vocab": "april", "tf-idf scores": 0.1473862061268713}, {"Date": "2000-05-16T00:00:00", "vocab": "labor", "tf-idf scores": 0.1347520521928197}, {"Date": "2000-05-16T00:00:00", "vocab": "indications", "tf-idf scores": 0.10886589216266904}, {"Date": "2000-05-16T00:00:00", "vocab": "prices", "tf-idf scores": 0.10828988280514333}, {"Date": "2000-05-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.10779143638344045}, {"Date": "2000-05-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.10775626157256334}, {"Date": "2000-05-16T00:00:00", "vocab": "recent", "tf-idf scores": 0.10774295492034064}, {"Date": "2000-03-21T00:00:00", "vocab": "growth", "tf-idf scores": 0.1882817909532903}, {"Date": "2000-03-21T00:00:00", "vocab": "members", "tf-idf scores": 0.1479799293197182}, {"Date": "2000-03-21T00:00:00", "vocab": "acceleration", "tf-idf scores": 0.1366089089570518}, {"Date": "2000-03-21T00:00:00", "vocab": "aggregate", "tf-idf scores": 0.1347290489533027}, {"Date": "2000-03-21T00:00:00", "vocab": "demand", "tf-idf scores": 0.1332898327115908}, {"Date": "2000-03-21T00:00:00", "vocab": "february", "tf-idf scores": 0.12356313559738719}, {"Date": "2000-03-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.12228521563552633}, {"Date": "2000-03-21T00:00:00", "vocab": "collateral", "tf-idf scores": 0.11890926063276591}, {"Date": "2000-03-21T00:00:00", "vocab": "century", "tf-idf scores": 0.11808927581264304}, {"Date": "2000-03-21T00:00:00", "vocab": "january", "tf-idf scores": 0.11636907710120085}, {"Date": "2000-02-02T00:00:00", "vocab": "shall", "tf-idf scores": 0.2562687785607788}, {"Date": "2000-02-02T00:00:00", "vocab": "foreign", "tf-idf scores": 0.2225629997610411}, {"Date": "2000-02-02T00:00:00", "vocab": "ranges", "tf-idf scores": 0.22099019378352222}, {"Date": "2000-02-02T00:00:00", "vocab": "currency", "tf-idf scores": 0.18335814098010808}, {"Date": "2000-02-02T00:00:00", "vocab": "growth", "tf-idf scores": 0.16872698225089036}, {"Date": "2000-02-02T00:00:00", "vocab": "market", "tf-idf scores": 0.1553295730498932}, {"Date": "2000-02-02T00:00:00", "vocab": "bank", "tf-idf scores": 0.12651376682307727}, {"Date": "2000-02-02T00:00:00", "vocab": "members", "tf-idf scores": 0.12289717009630978}, {"Date": "2000-02-02T00:00:00", "vocab": "open", "tf-idf scores": 0.11815411259641756}, {"Date": "2000-02-02T00:00:00", "vocab": "new", "tf-idf scores": 0.11342994266164491}]}}, {"mode": "vega-lite"});
</script>



**For Minutes**


```python
pd.options.display.max_rows = 600
# adding a little randomness to break ties in term ranking
top_tfidf_plusRand = top_minutes.copy()
top_tfidf_plusRand['tf-idf scores'] = top_tfidf_plusRand['tf-idf scores'] + np.random.rand(top_minutes.shape[0])*0.0001

# base for all visualizations, with rank calculation
base = alt.Chart(top_tfidf_plusRand).encode(
    x = 'rank:O',
    y = 'Date:N'
).transform_window(
    rank = "rank()",
    sort = [alt.SortField("tf-idf scores", order="descending")],
    groupby = ["Date"],
)

# heatmap specification, the color follows the number of tfidf
heatmap = base.mark_rect().encode(
    color = 'tf-idf scores:Q'
)


# text labels, white for darker heatmap colors
text = base.mark_text(baseline='middle').encode(
    text = 'vocab:N',
    color = alt.condition(alt.datum.tfidf >= 0.23, alt.value('white'), alt.value('black'))
)

# display the three superimposed visualizations
(heatmap + text).properties(width=600)
```





<style>
  #altair-viz-af09bce9d86e4e2e8e7ded83b268ffde.vega-embed {
    width: 100%;
    display: flex;
  }

  #altair-viz-af09bce9d86e4e2e8e7ded83b268ffde.vega-embed details,
  #altair-viz-af09bce9d86e4e2e8e7ded83b268ffde.vega-embed details summary {
    position: relative;
  }
</style>
<div id="altair-viz-af09bce9d86e4e2e8e7ded83b268ffde"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-af09bce9d86e4e2e8e7ded83b268ffde") {
      outputDiv = document.getElementById("altair-viz-af09bce9d86e4e2e8e7ded83b268ffde");
    }

    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm/vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm/vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm/vega-lite@5.20.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm/vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      let deps = ["vega-embed"];
      require(deps, displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "5.20.1"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 300, "continuousHeight": 300}}, "layer": [{"mark": {"type": "rect"}, "encoding": {"color": {"field": "tf-idf scores", "type": "quantitative"}, "x": {"field": "rank", "type": "ordinal"}, "y": {"field": "Date", "type": "nominal"}}, "transform": [{"window": [{"op": "rank", "field": "", "as": "rank"}], "groupby": ["Date"], "sort": [{"field": "tf-idf scores", "order": "descending"}]}]}, {"mark": {"type": "text", "baseline": "middle"}, "encoding": {"color": {"condition": {"test": "(datum.tfidf >= 0.23)", "value": "white"}, "value": "black"}, "text": {"field": "vocab", "type": "nominal"}, "x": {"field": "rank", "type": "ordinal"}, "y": {"field": "Date", "type": "nominal"}}, "transform": [{"window": [{"op": "rank", "field": "", "as": "rank"}], "groupby": ["Date"], "sort": [{"field": "tf-idf scores", "order": "descending"}]}]}], "data": {"name": "data-502e4a9bd67a7b867863f7089c7f608b"}, "width": 600, "$schema": "https://vega.github.io/schema/vega-lite/v5.20.1.json", "datasets": {"data-502e4a9bd67a7b867863f7089c7f608b": [{"Date": "2024-11-07T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21657848862171}, {"Date": "2024-11-07T00:00:00", "vocab": "market", "tf-idf scores": 0.2067461736605233}, {"Date": "2024-11-07T00:00:00", "vocab": "remained", "tf-idf scores": 0.18702198099408188}, {"Date": "2024-11-07T00:00:00", "vocab": "labor", "tf-idf scores": 0.1722900148705858}, {"Date": "2024-11-07T00:00:00", "vocab": "continued", "tf-idf scores": 0.14771764591623066}, {"Date": "2024-11-07T00:00:00", "vocab": "policy", "tf-idf scores": 0.14766466231971181}, {"Date": "2024-11-07T00:00:00", "vocab": "risks", "tf-idf scores": 0.13347494267129184}, {"Date": "2024-11-07T00:00:00", "vocab": "economic", "tf-idf scores": 0.128030511428715}, {"Date": "2024-11-07T00:00:00", "vocab": "rrp", "tf-idf scores": 0.11965032175978017}, {"Date": "2024-11-07T00:00:00", "vocab": "rates", "tf-idf scores": 0.10830128340873593}, {"Date": "2024-09-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.29777014765968834}, {"Date": "2024-09-18T00:00:00", "vocab": "policy", "tf-idf scores": 0.18293109641451266}, {"Date": "2024-09-18T00:00:00", "vocab": "market", "tf-idf scores": 0.1567874220203914}, {"Date": "2024-09-18T00:00:00", "vocab": "remained", "tf-idf scores": 0.15150700416273194}, {"Date": "2024-09-18T00:00:00", "vocab": "labor", "tf-idf scores": 0.13585374912824827}, {"Date": "2024-09-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.12020049376296095}, {"Date": "2024-09-18T00:00:00", "vocab": "risks", "tf-idf scores": 0.11550747007491571}, {"Date": "2024-09-18T00:00:00", "vocab": "continued", "tf-idf scores": 0.10979594244141862}, {"Date": "2024-09-18T00:00:00", "vocab": "july", "tf-idf scores": 0.10958135823344405}, {"Date": "2024-09-18T00:00:00", "vocab": "credit", "tf-idf scores": 0.10865960714875567}, {"Date": "2024-07-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23596584478128313}, {"Date": "2024-07-31T00:00:00", "vocab": "remained", "tf-idf scores": 0.2000809090098478}, {"Date": "2024-07-31T00:00:00", "vocab": "market", "tf-idf scores": 0.16928284055898404}, {"Date": "2024-07-31T00:00:00", "vocab": "noted", "tf-idf scores": 0.16491254775539765}, {"Date": "2024-07-31T00:00:00", "vocab": "continued", "tf-idf scores": 0.14363214638769872}, {"Date": "2024-07-31T00:00:00", "vocab": "labor", "tf-idf scores": 0.13856923957573455}, {"Date": "2024-07-31T00:00:00", "vocab": "risks", "tf-idf scores": 0.11855679300737168}, {"Date": "2024-07-31T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11804999039033531}, {"Date": "2024-07-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.11293754139812326}, {"Date": "2024-07-31T00:00:00", "vocab": "policy", "tf-idf scores": 0.10778091357920919}, {"Date": "2024-06-12T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3076028172001549}, {"Date": "2024-06-12T00:00:00", "vocab": "market", "tf-idf scores": 0.17947158639932376}, {"Date": "2024-06-12T00:00:00", "vocab": "economic", "tf-idf scores": 0.1640449587834783}, {"Date": "2024-06-12T00:00:00", "vocab": "remained", "tf-idf scores": 0.16403835951179596}, {"Date": "2024-06-12T00:00:00", "vocab": "labor", "tf-idf scores": 0.15896926469406358}, {"Date": "2024-06-12T00:00:00", "vocab": "continued", "tf-idf scores": 0.13845922099192387}, {"Date": "2024-06-12T00:00:00", "vocab": "policy", "tf-idf scores": 0.13332852525446093}, {"Date": "2024-06-12T00:00:00", "vocab": "credit", "tf-idf scores": 0.11255714438771135}, {"Date": "2024-06-12T00:00:00", "vocab": "rates", "tf-idf scores": 0.10769440070461836}, {"Date": "2024-06-12T00:00:00", "vocab": "april", "tf-idf scores": 0.10283930177513873}, {"Date": "2024-05-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2484901205793719}, {"Date": "2024-05-01T00:00:00", "vocab": "cap", "tf-idf scores": 0.19709194672164804}, {"Date": "2024-05-01T00:00:00", "vocab": "remained", "tf-idf scores": 0.16108861600263996}, {"Date": "2024-05-01T00:00:00", "vocab": "redemption", "tf-idf scores": 0.14259684697858904}, {"Date": "2024-05-01T00:00:00", "vocab": "growth", "tf-idf scores": 0.12021547566681713}, {"Date": "2024-05-01T00:00:00", "vocab": "commented", "tf-idf scores": 0.11863273084535333}, {"Date": "2024-05-01T00:00:00", "vocab": "treasury", "tf-idf scores": 0.11846046650859843}, {"Date": "2024-05-01T00:00:00", "vocab": "agency", "tf-idf scores": 0.11663336274283503}, {"Date": "2024-05-01T00:00:00", "vocab": "recent", "tf-idf scores": 0.11504225672349715}, {"Date": "2024-05-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.11053408395438631}, {"Date": "2024-03-20T00:00:00", "vocab": "runoff", "tf-idf scores": 0.2738530313999843}, {"Date": "2024-03-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2198872410744506}, {"Date": "2024-03-20T00:00:00", "vocab": "sheet", "tf-idf scores": 0.16831732732270116}, {"Date": "2024-03-20T00:00:00", "vocab": "balance", "tf-idf scores": 0.14612863479386123}, {"Date": "2024-03-20T00:00:00", "vocab": "policy", "tf-idf scores": 0.14357994636829705}, {"Date": "2024-03-20T00:00:00", "vocab": "remained", "tf-idf scores": 0.1391136078258698}, {"Date": "2024-03-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.1256300446850665}, {"Date": "2024-03-20T00:00:00", "vocab": "january", "tf-idf scores": 0.1209677447081078}, {"Date": "2024-03-20T00:00:00", "vocab": "credit", "tf-idf scores": 0.10880047849479141}, {"Date": "2024-03-20T00:00:00", "vocab": "pace", "tf-idf scores": 0.10767750947491972}, {"Date": "2024-01-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.32296438362723034}, {"Date": "2024-01-31T00:00:00", "vocab": "remained", "tf-idf scores": 0.20794212687073976}, {"Date": "2024-01-31T00:00:00", "vocab": "policy", "tf-idf scores": 0.1681874373093676}, {"Date": "2024-01-31T00:00:00", "vocab": "continued", "tf-idf scores": 0.13277976832009025}, {"Date": "2024-01-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.1327679799194767}, {"Date": "2024-01-31T00:00:00", "vocab": "market", "tf-idf scores": 0.1327352952910685}, {"Date": "2024-01-31T00:00:00", "vocab": "noted", "tf-idf scores": 0.1199724991905311}, {"Date": "2024-01-31T00:00:00", "vocab": "better", "tf-idf scores": 0.10278364317292339}, {"Date": "2024-01-31T00:00:00", "vocab": "sustainably", "tf-idf scores": 0.09738111554053219}, {"Date": "2024-01-31T00:00:00", "vocab": "fourth", "tf-idf scores": 0.09520949504917954}, {"Date": "2023-12-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.325753806354552}, {"Date": "2023-12-13T00:00:00", "vocab": "policy", "tf-idf scores": 0.1896198395097529}, {"Date": "2023-12-13T00:00:00", "vocab": "market", "tf-idf scores": 0.1702193035782812}, {"Date": "2023-12-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.1652974736003653}, {"Date": "2023-12-13T00:00:00", "vocab": "growth", "tf-idf scores": 0.14651300115550778}, {"Date": "2023-12-13T00:00:00", "vocab": "remained", "tf-idf scores": 0.14585783621358564}, {"Date": "2023-12-13T00:00:00", "vocab": "labor", "tf-idf scores": 0.1410402900027101}, {"Date": "2023-12-13T00:00:00", "vocab": "continued", "tf-idf scores": 0.12646023142612056}, {"Date": "2023-12-13T00:00:00", "vocab": "restrictive", "tf-idf scores": 0.11713719936867546}, {"Date": "2023-12-13T00:00:00", "vocab": "credit", "tf-idf scores": 0.11228941850857105}, {"Date": "2023-11-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2640918188520987}, {"Date": "2023-11-01T00:00:00", "vocab": "market", "tf-idf scores": 0.17936094013445164}, {"Date": "2023-11-01T00:00:00", "vocab": "remained", "tf-idf scores": 0.16948264797019916}, {"Date": "2023-11-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.1495156956137903}, {"Date": "2023-11-01T00:00:00", "vocab": "policy", "tf-idf scores": 0.14455613395849132}, {"Date": "2023-11-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.13957526697451172}, {"Date": "2023-11-01T00:00:00", "vocab": "financial", "tf-idf scores": 0.1356897591656355}, {"Date": "2023-11-01T00:00:00", "vocab": "credit", "tf-idf scores": 0.13234369663807208}, {"Date": "2023-11-01T00:00:00", "vocab": "labor", "tf-idf scores": 0.12958645067025137}, {"Date": "2023-11-01T00:00:00", "vocab": "treasury", "tf-idf scores": 0.12823445525194901}, {"Date": "2023-09-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3010490518153733}, {"Date": "2023-09-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.2007111636418095}, {"Date": "2023-09-20T00:00:00", "vocab": "policy", "tf-idf scores": 0.20069794770536636}, {"Date": "2023-09-20T00:00:00", "vocab": "market", "tf-idf scores": 0.15053033503317634}, {"Date": "2023-09-20T00:00:00", "vocab": "remained", "tf-idf scores": 0.1354887043293156}, {"Date": "2023-09-20T00:00:00", "vocab": "july", "tf-idf scores": 0.1338983295715686}, {"Date": "2023-09-20T00:00:00", "vocab": "credit", "tf-idf scores": 0.12166369418004912}, {"Date": "2023-09-20T00:00:00", "vocab": "percent", "tf-idf scores": 0.11734772598589167}, {"Date": "2023-09-20T00:00:00", "vocab": "noted", "tf-idf scores": 0.11591378107592754}, {"Date": "2023-09-20T00:00:00", "vocab": "labor", "tf-idf scores": 0.11038316941052201}, {"Date": "2023-07-26T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3241903094184457}, {"Date": "2023-07-26T00:00:00", "vocab": "remained", "tf-idf scores": 0.2294307056318453}, {"Date": "2023-07-26T00:00:00", "vocab": "policy", "tf-idf scores": 0.17956098596473224}, {"Date": "2023-07-26T00:00:00", "vocab": "economic", "tf-idf scores": 0.17453235859355976}, {"Date": "2023-07-26T00:00:00", "vocab": "market", "tf-idf scores": 0.164559216467804}, {"Date": "2023-07-26T00:00:00", "vocab": "percent", "tf-idf scores": 0.13250916524710743}, {"Date": "2023-07-26T00:00:00", "vocab": "credit", "tf-idf scores": 0.13250444360345123}, {"Date": "2023-07-26T00:00:00", "vocab": "continued", "tf-idf scores": 0.12970294598398383}, {"Date": "2023-07-26T00:00:00", "vocab": "banks", "tf-idf scores": 0.120476083205715}, {"Date": "2023-07-26T00:00:00", "vocab": "july", "tf-idf scores": 0.11407686147513942}, {"Date": "2023-06-14T00:00:00", "vocab": "inflation", "tf-idf scores": 0.34916708469566043}, {"Date": "2023-06-14T00:00:00", "vocab": "economic", "tf-idf scores": 0.19822471101443068}, {"Date": "2023-06-14T00:00:00", "vocab": "policy", "tf-idf scores": 0.16992373497339006}, {"Date": "2023-06-14T00:00:00", "vocab": "remained", "tf-idf scores": 0.16984690252279455}, {"Date": "2023-06-14T00:00:00", "vocab": "credit", "tf-idf scores": 0.15807337974798252}, {"Date": "2023-06-14T00:00:00", "vocab": "market", "tf-idf scores": 0.15571310531456553}, {"Date": "2023-06-14T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1226659620957846}, {"Date": "2023-06-14T00:00:00", "vocab": "monetary", "tf-idf scores": 0.11795566018226664}, {"Date": "2023-06-14T00:00:00", "vocab": "percent", "tf-idf scores": 0.11541085355412037}, {"Date": "2023-06-14T00:00:00", "vocab": "noted", "tf-idf scores": 0.11375516164948991}, {"Date": "2023-05-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.26005634157245616}, {"Date": "2023-05-03T00:00:00", "vocab": "banking", "tf-idf scores": 0.23693175609288789}, {"Date": "2023-05-03T00:00:00", "vocab": "remained", "tf-idf scores": 0.18166979640912725}, {"Date": "2023-05-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.16926401258100573}, {"Date": "2023-05-03T00:00:00", "vocab": "stress", "tf-idf scores": 0.15643082387326818}, {"Date": "2023-05-03T00:00:00", "vocab": "market", "tf-idf scores": 0.1527327535250154}, {"Date": "2023-05-03T00:00:00", "vocab": "credit", "tf-idf scores": 0.1430784827358643}, {"Date": "2023-05-03T00:00:00", "vocab": "banks", "tf-idf scores": 0.118690230095781}, {"Date": "2023-05-03T00:00:00", "vocab": "noted", "tf-idf scores": 0.11616497164756294}, {"Date": "2023-05-03T00:00:00", "vocab": "policy", "tf-idf scores": 0.11562481496473967}, {"Date": "2023-03-22T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3046703774195745}, {"Date": "2023-03-22T00:00:00", "vocab": "banking", "tf-idf scores": 0.2404879277936468}, {"Date": "2023-03-22T00:00:00", "vocab": "economic", "tf-idf scores": 0.1868413879786472}, {"Date": "2023-03-22T00:00:00", "vocab": "signature", "tf-idf scores": 0.18666900051435725}, {"Date": "2023-03-22T00:00:00", "vocab": "silicon", "tf-idf scores": 0.1765558249967353}, {"Date": "2023-03-22T00:00:00", "vocab": "valley", "tf-idf scores": 0.1765795831109977}, {"Date": "2023-03-22T00:00:00", "vocab": "developments", "tf-idf scores": 0.1417749087927213}, {"Date": "2023-03-22T00:00:00", "vocab": "policy", "tf-idf scores": 0.1381595000880383}, {"Date": "2023-03-22T00:00:00", "vocab": "recent", "tf-idf scores": 0.1381094979877398}, {"Date": "2023-03-22T00:00:00", "vocab": "market", "tf-idf scores": 0.13407860687162013}, {"Date": "2023-02-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.27991250790599814}, {"Date": "2023-02-01T00:00:00", "vocab": "remained", "tf-idf scores": 0.1983029921035137}, {"Date": "2023-02-01T00:00:00", "vocab": "market", "tf-idf scores": 0.18663270938905832}, {"Date": "2023-02-01T00:00:00", "vocab": "policy", "tf-idf scores": 0.13219035330871298}, {"Date": "2023-02-01T00:00:00", "vocab": "growth", "tf-idf scores": 0.12502364439768118}, {"Date": "2023-02-01T00:00:00", "vocab": "noted", "tf-idf scores": 0.12113301741464975}, {"Date": "2023-02-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.10892971450394431}, {"Date": "2023-02-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.10115939866938095}, {"Date": "2023-02-01T00:00:00", "vocab": "financial", "tf-idf scores": 0.09813278428895147}, {"Date": "2023-02-01T00:00:00", "vocab": "credit", "tf-idf scores": 0.09428666255846095}, {"Date": "2022-12-14T00:00:00", "vocab": "inflation", "tf-idf scores": 0.36486727923299656}, {"Date": "2022-12-14T00:00:00", "vocab": "remained", "tf-idf scores": 0.1824463970850048}, {"Date": "2022-12-14T00:00:00", "vocab": "economic", "tf-idf scores": 0.14336796188904735}, {"Date": "2022-12-14T00:00:00", "vocab": "restrictive", "tf-idf scores": 0.13954765527322124}, {"Date": "2022-12-14T00:00:00", "vocab": "market", "tf-idf scores": 0.13472451798418753}, {"Date": "2022-12-14T00:00:00", "vocab": "policy", "tf-idf scores": 0.1259757954837487}, {"Date": "2022-12-14T00:00:00", "vocab": "october", "tf-idf scores": 0.12202488525623893}, {"Date": "2022-12-14T00:00:00", "vocab": "credit", "tf-idf scores": 0.11536806232158671}, {"Date": "2022-12-14T00:00:00", "vocab": "war", "tf-idf scores": 0.11484324632968368}, {"Date": "2022-12-14T00:00:00", "vocab": "continued", "tf-idf scores": 0.10865825790459002}, {"Date": "2022-11-02T00:00:00", "vocab": "inflation", "tf-idf scores": 0.31306024259949833}, {"Date": "2022-11-02T00:00:00", "vocab": "market", "tf-idf scores": 0.1992268788651078}, {"Date": "2022-11-02T00:00:00", "vocab": "policy", "tf-idf scores": 0.19918093341466506}, {"Date": "2022-11-02T00:00:00", "vocab": "monetary", "tf-idf scores": 0.17076746684392552}, {"Date": "2022-11-02T00:00:00", "vocab": "economic", "tf-idf scores": 0.15856482803669117}, {"Date": "2022-11-02T00:00:00", "vocab": "remained", "tf-idf scores": 0.15041466358080555}, {"Date": "2022-11-02T00:00:00", "vocab": "financial", "tf-idf scores": 0.1476330498013299}, {"Date": "2022-11-02T00:00:00", "vocab": "noted", "tf-idf scores": 0.11022423929169695}, {"Date": "2022-11-02T00:00:00", "vocab": "restrictive", "tf-idf scores": 0.10889760320411639}, {"Date": "2022-11-02T00:00:00", "vocab": "war", "tf-idf scores": 0.10747290027647541}, {"Date": "2022-09-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3144523081853645}, {"Date": "2022-09-21T00:00:00", "vocab": "policy", "tf-idf scores": 0.20548185425496526}, {"Date": "2022-09-21T00:00:00", "vocab": "remained", "tf-idf scores": 0.1593771610433447}, {"Date": "2022-09-21T00:00:00", "vocab": "market", "tf-idf scores": 0.15098244802293204}, {"Date": "2022-09-21T00:00:00", "vocab": "restrictive", "tf-idf scores": 0.13476979045879917}, {"Date": "2022-09-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.1342422224642663}, {"Date": "2022-09-21T00:00:00", "vocab": "july", "tf-idf scores": 0.12784379920464958}, {"Date": "2022-09-21T00:00:00", "vocab": "war", "tf-idf scores": 0.12467420503423758}, {"Date": "2022-09-21T00:00:00", "vocab": "continued", "tf-idf scores": 0.11742954316153258}, {"Date": "2022-09-21T00:00:00", "vocab": "labor", "tf-idf scores": 0.1174193635396345}, {"Date": "2022-07-27T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3225593748932739}, {"Date": "2022-07-27T00:00:00", "vocab": "policy", "tf-idf scores": 0.18263011606622212}, {"Date": "2022-07-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.14770642846952517}, {"Date": "2022-07-27T00:00:00", "vocab": "remained", "tf-idf scores": 0.13601976802518642}, {"Date": "2022-07-27T00:00:00", "vocab": "supply", "tf-idf scores": 0.1319037023671494}, {"Date": "2022-07-27T00:00:00", "vocab": "financial", "tf-idf scores": 0.12942218775930855}, {"Date": "2022-07-27T00:00:00", "vocab": "market", "tf-idf scores": 0.1243795533343847}, {"Date": "2022-07-27T00:00:00", "vocab": "credit", "tf-idf scores": 0.12115491046204584}, {"Date": "2022-07-27T00:00:00", "vocab": "war", "tf-idf scores": 0.11548315679371735}, {"Date": "2022-07-27T00:00:00", "vocab": "growth", "tf-idf scores": 0.10930643261664989}, {"Date": "2022-06-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3216544329352834}, {"Date": "2022-06-15T00:00:00", "vocab": "invasion", "tf-idf scores": 0.2708470858383764}, {"Date": "2022-06-15T00:00:00", "vocab": "policy", "tf-idf scores": 0.17328208805863635}, {"Date": "2022-06-15T00:00:00", "vocab": "supply", "tf-idf scores": 0.1516782856964559}, {"Date": "2022-06-15T00:00:00", "vocab": "ukraine", "tf-idf scores": 0.13384376892674565}, {"Date": "2022-06-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.13204487372082724}, {"Date": "2022-06-15T00:00:00", "vocab": "market", "tf-idf scores": 0.13198314583226886}, {"Date": "2022-06-15T00:00:00", "vocab": "remained", "tf-idf scores": 0.1278590849943098}, {"Date": "2022-06-15T00:00:00", "vocab": "percent", "tf-idf scores": 0.122806146955818}, {"Date": "2022-06-15T00:00:00", "vocab": "credit", "tf-idf scores": 0.11912592078816574}, {"Date": "2022-05-04T00:00:00", "vocab": "invasion", "tf-idf scores": 0.33495876103987493}, {"Date": "2022-05-04T00:00:00", "vocab": "inflation", "tf-idf scores": 0.24862882819745272}, {"Date": "2022-05-04T00:00:00", "vocab": "lockdowns", "tf-idf scores": 0.15424578981489112}, {"Date": "2022-05-04T00:00:00", "vocab": "ukraine", "tf-idf scores": 0.14938737581771633}, {"Date": "2022-05-04T00:00:00", "vocab": "policy", "tf-idf scores": 0.1450630732899032}, {"Date": "2022-05-04T00:00:00", "vocab": "market", "tf-idf scores": 0.14094047648388144}, {"Date": "2022-05-04T00:00:00", "vocab": "supply", "tf-idf scores": 0.12899951147513847}, {"Date": "2022-05-04T00:00:00", "vocab": "continued", "tf-idf scores": 0.12017491746241522}, {"Date": "2022-05-04T00:00:00", "vocab": "remained", "tf-idf scores": 0.11607794421881333}, {"Date": "2022-05-04T00:00:00", "vocab": "economic", "tf-idf scores": 0.10362465841823966}, {"Date": "2022-03-16T00:00:00", "vocab": "invasion", "tf-idf scores": 0.3922720046237758}, {"Date": "2022-03-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.24712658109483165}, {"Date": "2022-03-16T00:00:00", "vocab": "russian", "tf-idf scores": 0.2154423718787304}, {"Date": "2022-03-16T00:00:00", "vocab": "ukraine", "tf-idf scores": 0.16546512441546174}, {"Date": "2022-03-16T00:00:00", "vocab": "market", "tf-idf scores": 0.14119008286254683}, {"Date": "2022-03-16T00:00:00", "vocab": "treasury", "tf-idf scores": 0.12109783512851388}, {"Date": "2022-03-16T00:00:00", "vocab": "sheet", "tf-idf scores": 0.11585855534797859}, {"Date": "2022-03-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.10594016198637812}, {"Date": "2022-03-16T00:00:00", "vocab": "supply", "tf-idf scores": 0.10489455588751335}, {"Date": "2022-03-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.10242688796445794}, {"Date": "2022-01-26T00:00:00", "vocab": "selected", "tf-idf scores": 0.2052753390879433}, {"Date": "2022-01-26T00:00:00", "vocab": "omicron", "tf-idf scores": 0.20207291078682818}, {"Date": "2022-01-26T00:00:00", "vocab": "foreign", "tf-idf scores": 0.19876241890894236}, {"Date": "2022-01-26T00:00:00", "vocab": "market", "tf-idf scores": 0.15693642341308664}, {"Date": "2022-01-26T00:00:00", "vocab": "currency", "tf-idf scores": 0.15076776729421865}, {"Date": "2022-01-26T00:00:00", "vocab": "securities", "tf-idf scores": 0.1418566526007985}, {"Date": "2022-01-26T00:00:00", "vocab": "bank", "tf-idf scores": 0.1314135352556411}, {"Date": "2022-01-26T00:00:00", "vocab": "inflation", "tf-idf scores": 0.12823570968675213}, {"Date": "2022-01-26T00:00:00", "vocab": "policy", "tf-idf scores": 0.1256077772367}, {"Date": "2022-01-26T00:00:00", "vocab": "eligible", "tf-idf scores": 0.11802922118793614}, {"Date": "2021-12-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1841917851251461}, {"Date": "2021-12-15T00:00:00", "vocab": "policy", "tf-idf scores": 0.18075478676799012}, {"Date": "2021-12-15T00:00:00", "vocab": "continued", "tf-idf scores": 0.15289385503326858}, {"Date": "2021-12-15T00:00:00", "vocab": "omicron", "tf-idf scores": 0.1510827897652921}, {"Date": "2021-12-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.14603011178694186}, {"Date": "2021-12-15T00:00:00", "vocab": "normalization", "tf-idf scores": 0.14164222860319464}, {"Date": "2021-12-15T00:00:00", "vocab": "sheet", "tf-idf scores": 0.1358370318372244}, {"Date": "2021-12-15T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.13259232062960555}, {"Date": "2021-12-15T00:00:00", "vocab": "labor", "tf-idf scores": 0.1251279714437182}, {"Date": "2021-12-15T00:00:00", "vocab": "market", "tf-idf scores": 0.12514678814105779}, {"Date": "2021-11-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22014732171049173}, {"Date": "2021-11-03T00:00:00", "vocab": "supply", "tf-idf scores": 0.18680963936132075}, {"Date": "2021-11-03T00:00:00", "vocab": "remained", "tf-idf scores": 0.14971112172148654}, {"Date": "2021-11-03T00:00:00", "vocab": "vaccinations", "tf-idf scores": 0.14227919863793448}, {"Date": "2021-11-03T00:00:00", "vocab": "market", "tf-idf scores": 0.14085954613336388}, {"Date": "2021-11-03T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.12601639914659546}, {"Date": "2021-11-03T00:00:00", "vocab": "continued", "tf-idf scores": 0.11889238396185592}, {"Date": "2021-11-03T00:00:00", "vocab": "asset", "tf-idf scores": 0.11472727468577906}, {"Date": "2021-11-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.11447366661070428}, {"Date": "2021-11-03T00:00:00", "vocab": "treasury", "tf-idf scores": 0.11331222435930532}, {"Date": "2021-09-22T00:00:00", "vocab": "inflation", "tf-idf scores": 0.244884072909908}, {"Date": "2021-09-22T00:00:00", "vocab": "market", "tf-idf scores": 0.17493269905178754}, {"Date": "2021-09-22T00:00:00", "vocab": "delta", "tf-idf scores": 0.15022419100429013}, {"Date": "2021-09-22T00:00:00", "vocab": "july", "tf-idf scores": 0.14074651124316806}, {"Date": "2021-09-22T00:00:00", "vocab": "variant", "tf-idf scores": 0.13976063093878427}, {"Date": "2021-09-22T00:00:00", "vocab": "supply", "tf-idf scores": 0.1374114418342559}, {"Date": "2021-09-22T00:00:00", "vocab": "labor", "tf-idf scores": 0.1321554297718566}, {"Date": "2021-09-22T00:00:00", "vocab": "remained", "tf-idf scores": 0.13214945303694417}, {"Date": "2021-09-22T00:00:00", "vocab": "asset", "tf-idf scores": 0.12943840499754655}, {"Date": "2021-09-22T00:00:00", "vocab": "tapering", "tf-idf scores": 0.1256867891525742}, {"Date": "2021-07-28T00:00:00", "vocab": "asset", "tf-idf scores": 0.24292489852016028}, {"Date": "2021-07-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21480487388063868}, {"Date": "2021-07-28T00:00:00", "vocab": "tapering", "tf-idf scores": 0.20341631535128094}, {"Date": "2021-07-28T00:00:00", "vocab": "purchases", "tf-idf scores": 0.19103985367556384}, {"Date": "2021-07-28T00:00:00", "vocab": "remained", "tf-idf scores": 0.1343113875153602}, {"Date": "2021-07-28T00:00:00", "vocab": "progress", "tf-idf scores": 0.13047749447769572}, {"Date": "2021-07-28T00:00:00", "vocab": "market", "tf-idf scores": 0.12750435702500829}, {"Date": "2021-07-28T00:00:00", "vocab": "financial", "tf-idf scores": 0.12190203594305418}, {"Date": "2021-07-28T00:00:00", "vocab": "continued", "tf-idf scores": 0.1141008238843562}, {"Date": "2021-07-28T00:00:00", "vocab": "labor", "tf-idf scores": 0.10745104470110575}, {"Date": "2021-06-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23811861423273437}, {"Date": "2021-06-16T00:00:00", "vocab": "srf", "tf-idf scores": 0.23118104820525726}, {"Date": "2021-06-16T00:00:00", "vocab": "fima", "tf-idf scores": 0.18082059232616396}, {"Date": "2021-06-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.14216006051868382}, {"Date": "2021-06-16T00:00:00", "vocab": "repo", "tf-idf scores": 0.1419144412688517}, {"Date": "2021-06-16T00:00:00", "vocab": "market", "tf-idf scores": 0.13859482123518047}, {"Date": "2021-06-16T00:00:00", "vocab": "facility", "tf-idf scores": 0.11793532741717583}, {"Date": "2021-06-16T00:00:00", "vocab": "progress", "tf-idf scores": 0.11605718367179094}, {"Date": "2021-06-16T00:00:00", "vocab": "vaccinations", "tf-idf scores": 0.11488375053516331}, {"Date": "2021-06-16T00:00:00", "vocab": "remained", "tf-idf scores": 0.10311311583356692}, {"Date": "2021-04-28T00:00:00", "vocab": "repo", "tf-idf scores": 0.266299831006007}, {"Date": "2021-04-28T00:00:00", "vocab": "standing", "tf-idf scores": 0.19124132481552658}, {"Date": "2021-04-28T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.17107391143494532}, {"Date": "2021-04-28T00:00:00", "vocab": "continued", "tf-idf scores": 0.1449280751440939}, {"Date": "2021-04-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14073070212607652}, {"Date": "2021-04-28T00:00:00", "vocab": "noted", "tf-idf scores": 0.1372502302948643}, {"Date": "2021-04-28T00:00:00", "vocab": "market", "tf-idf scores": 0.1366094235289378}, {"Date": "2021-04-28T00:00:00", "vocab": "remained", "tf-idf scores": 0.13660433213128148}, {"Date": "2021-04-28T00:00:00", "vocab": "facility", "tf-idf scores": 0.12362214477022773}, {"Date": "2021-04-28T00:00:00", "vocab": "fima", "tf-idf scores": 0.12291444287148816}, {"Date": "2021-03-17T00:00:00", "vocab": "market", "tf-idf scores": 0.18739086365218252}, {"Date": "2021-03-17T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1830539543219932}, {"Date": "2021-03-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.15685605299711466}, {"Date": "2021-03-17T00:00:00", "vocab": "remained", "tf-idf scores": 0.14813820911334452}, {"Date": "2021-03-17T00:00:00", "vocab": "continued", "tf-idf scores": 0.14382923902704464}, {"Date": "2021-03-17T00:00:00", "vocab": "rrp", "tf-idf scores": 0.13765927512501966}, {"Date": "2021-03-17T00:00:00", "vocab": "january", "tf-idf scores": 0.13135268355236285}, {"Date": "2021-03-17T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12199772120325161}, {"Date": "2021-03-17T00:00:00", "vocab": "facility", "tf-idf scores": 0.11569196591813218}, {"Date": "2021-03-17T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.11083475422927289}, {"Date": "2021-01-27T00:00:00", "vocab": "foreign", "tf-idf scores": 0.22896403950037494}, {"Date": "2021-01-27T00:00:00", "vocab": "selected", "tf-idf scores": 0.22180520310720261}, {"Date": "2021-01-27T00:00:00", "vocab": "market", "tf-idf scores": 0.16969475854032587}, {"Date": "2021-01-27T00:00:00", "vocab": "currency", "tf-idf scores": 0.16794844853175053}, {"Date": "2021-01-27T00:00:00", "vocab": "bank", "tf-idf scores": 0.16469473836899376}, {"Date": "2021-01-27T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.1348733961439981}, {"Date": "2021-01-27T00:00:00", "vocab": "eligible", "tf-idf scores": 0.1275713967223579}, {"Date": "2021-01-27T00:00:00", "vocab": "securities", "tf-idf scores": 0.12458814781543769}, {"Date": "2021-01-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.11876187684925639}, {"Date": "2021-01-27T00:00:00", "vocab": "transactions", "tf-idf scores": 0.11363749024298181}, {"Date": "2020-12-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.19716829392435323}, {"Date": "2020-12-16T00:00:00", "vocab": "remained", "tf-idf scores": 0.19263045604793597}, {"Date": "2020-12-16T00:00:00", "vocab": "vaccines", "tf-idf scores": 0.1674162599417067}, {"Date": "2020-12-16T00:00:00", "vocab": "vaccine", "tf-idf scores": 0.16656803034460252}, {"Date": "2020-12-16T00:00:00", "vocab": "market", "tf-idf scores": 0.1523062625664005}, {"Date": "2020-12-16T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.1425072147221398}, {"Date": "2020-12-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.1344567164633667}, {"Date": "2020-12-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13439285506712914}, {"Date": "2020-12-16T00:00:00", "vocab": "policy", "tf-idf scores": 0.1164965534454357}, {"Date": "2020-12-16T00:00:00", "vocab": "october", "tf-idf scores": 0.10225963756437845}, {"Date": "2020-11-05T00:00:00", "vocab": "asset", "tf-idf scores": 0.20131764652698433}, {"Date": "2020-11-05T00:00:00", "vocab": "market", "tf-idf scores": 0.1708869518195645}, {"Date": "2020-11-05T00:00:00", "vocab": "remained", "tf-idf scores": 0.16291989681162014}, {"Date": "2020-11-05T00:00:00", "vocab": "economic", "tf-idf scores": 0.15897902596947486}, {"Date": "2020-11-05T00:00:00", "vocab": "purchases", "tf-idf scores": 0.15074929406007315}, {"Date": "2020-11-05T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.13900798705242287}, {"Date": "2020-11-05T00:00:00", "vocab": "noted", "tf-idf scores": 0.12371848790247082}, {"Date": "2020-11-05T00:00:00", "vocab": "continued", "tf-idf scores": 0.12324538978379096}, {"Date": "2020-11-05T00:00:00", "vocab": "financial", "tf-idf scores": 0.12025804354966167}, {"Date": "2020-11-05T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11524399288938106}, {"Date": "2020-09-16T00:00:00", "vocab": "thomas", "tf-idf scores": 0.19978705965374619}, {"Date": "2020-09-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1749435645691413}, {"Date": "2020-09-16T00:00:00", "vocab": "july", "tf-idf scores": 0.17398310271957737}, {"Date": "2020-09-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.1673799419118748}, {"Date": "2020-09-16T00:00:00", "vocab": "market", "tf-idf scores": 0.16738131656658248}, {"Date": "2020-09-16T00:00:00", "vocab": "guidance", "tf-idf scores": 0.1495305434463439}, {"Date": "2020-09-16T00:00:00", "vocab": "policy", "tf-idf scores": 0.1369166282142797}, {"Date": "2020-09-16T00:00:00", "vocab": "remained", "tf-idf scores": 0.12933985100464926}, {"Date": "2020-09-16T00:00:00", "vocab": "consensus", "tf-idf scores": 0.1270478597051733}, {"Date": "2020-09-16T00:00:00", "vocab": "agency", "tf-idf scores": 0.11249310722765997}, {"Date": "2020-07-29T00:00:00", "vocab": "coronavirus", "tf-idf scores": 0.22479962133327172}, {"Date": "2020-07-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.17560176692522336}, {"Date": "2020-07-29T00:00:00", "vocab": "policy", "tf-idf scores": 0.15890882248989735}, {"Date": "2020-07-29T00:00:00", "vocab": "market", "tf-idf scores": 0.15466193978088422}, {"Date": "2020-07-29T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.13294224516703748}, {"Date": "2020-07-29T00:00:00", "vocab": "credit", "tf-idf scores": 0.1255211395198164}, {"Date": "2020-07-29T00:00:00", "vocab": "virus", "tf-idf scores": 0.11512695840092338}, {"Date": "2020-07-29T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1087592376917448}, {"Date": "2020-07-29T00:00:00", "vocab": "intermeeting", "tf-idf scores": 0.10496305001117101}, {"Date": "2020-07-29T00:00:00", "vocab": "monetary", "tf-idf scores": 0.10455285703892686}, {"Date": "2020-06-10T00:00:00", "vocab": "yct", "tf-idf scores": 0.2755603566247757}, {"Date": "2020-06-10T00:00:00", "vocab": "coronavirus", "tf-idf scores": 0.21254172698614837}, {"Date": "2020-06-10T00:00:00", "vocab": "policy", "tf-idf scores": 0.17815714194812543}, {"Date": "2020-06-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.16791379433237208}, {"Date": "2020-06-10T00:00:00", "vocab": "market", "tf-idf scores": 0.1541925322735521}, {"Date": "2020-06-10T00:00:00", "vocab": "agency", "tf-idf scores": 0.13991992209113555}, {"Date": "2020-06-10T00:00:00", "vocab": "guidance", "tf-idf scores": 0.1347037990220993}, {"Date": "2020-06-10T00:00:00", "vocab": "forward", "tf-idf scores": 0.10663433864578532}, {"Date": "2020-06-10T00:00:00", "vocab": "support", "tf-idf scores": 0.10651294188668962}, {"Date": "2020-06-10T00:00:00", "vocab": "monetary", "tf-idf scores": 0.1027949433417113}, {"Date": "2020-04-29T00:00:00", "vocab": "coronavirus", "tf-idf scores": 0.31638709089822714}, {"Date": "2020-04-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.19310128645332983}, {"Date": "2020-04-29T00:00:00", "vocab": "outbreak", "tf-idf scores": 0.19025478747901137}, {"Date": "2020-04-29T00:00:00", "vocab": "market", "tf-idf scores": 0.1712129492253906}, {"Date": "2020-04-29T00:00:00", "vocab": "financial", "tf-idf scores": 0.13965184540340758}, {"Date": "2020-04-29T00:00:00", "vocab": "flow", "tf-idf scores": 0.12573874261685372}, {"Date": "2020-04-29T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1166326444558651}, {"Date": "2020-04-29T00:00:00", "vocab": "credit", "tf-idf scores": 0.10937156108137884}, {"Date": "2020-04-29T00:00:00", "vocab": "pandemic", "tf-idf scores": 0.10426770853939363}, {"Date": "2020-04-29T00:00:00", "vocab": "agency", "tf-idf scores": 0.09747936199583322}, {"Date": "2020-03-15T00:00:00", "vocab": "coronavirus", "tf-idf scores": 0.3955848682848249}, {"Date": "2020-03-15T00:00:00", "vocab": "outbreak", "tf-idf scores": 0.22838434423640405}, {"Date": "2020-03-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.225760776306728}, {"Date": "2020-03-15T00:00:00", "vocab": "market", "tf-idf scores": 0.1607480601510673}, {"Date": "2020-03-15T00:00:00", "vocab": "activity", "tf-idf scores": 0.11101164566549324}, {"Date": "2020-03-15T00:00:00", "vocab": "credit", "tf-idf scores": 0.11054728829908497}, {"Date": "2020-03-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.10714745968582583}, {"Date": "2020-03-15T00:00:00", "vocab": "range", "tf-idf scores": 0.10243877843942503}, {"Date": "2020-03-15T00:00:00", "vocab": "households", "tf-idf scores": 0.09298274984081265}, {"Date": "2020-03-15T00:00:00", "vocab": "noted", "tf-idf scores": 0.09225054640658162}, {"Date": "2020-03-03T00:00:00", "vocab": "coronavirus", "tf-idf scores": 0.39554389626286846}, {"Date": "2020-03-03T00:00:00", "vocab": "outbreak", "tf-idf scores": 0.22838160660195056}, {"Date": "2020-03-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.22575572995042928}, {"Date": "2020-03-03T00:00:00", "vocab": "market", "tf-idf scores": 0.1607089059736645}, {"Date": "2020-03-03T00:00:00", "vocab": "activity", "tf-idf scores": 0.11103308263717591}, {"Date": "2020-03-03T00:00:00", "vocab": "credit", "tf-idf scores": 0.1104573162019048}, {"Date": "2020-03-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.10711429102355245}, {"Date": "2020-03-03T00:00:00", "vocab": "range", "tf-idf scores": 0.10243341602666864}, {"Date": "2020-03-03T00:00:00", "vocab": "households", "tf-idf scores": 0.09301640271490545}, {"Date": "2020-03-03T00:00:00", "vocab": "noted", "tf-idf scores": 0.09222670546278228}, {"Date": "2020-01-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23823347355170885}, {"Date": "2020-01-29T00:00:00", "vocab": "selected", "tf-idf scores": 0.2100200325148025}, {"Date": "2020-01-29T00:00:00", "vocab": "foreign", "tf-idf scores": 0.19272789089577252}, {"Date": "2020-01-29T00:00:00", "vocab": "market", "tf-idf scores": 0.18741940071285154}, {"Date": "2020-01-29T00:00:00", "vocab": "bank", "tf-idf scores": 0.1505757118009518}, {"Date": "2020-01-29T00:00:00", "vocab": "financial", "tf-idf scores": 0.14584963776544638}, {"Date": "2020-01-29T00:00:00", "vocab": "currency", "tf-idf scores": 0.1449893392457437}, {"Date": "2020-01-29T00:00:00", "vocab": "policy", "tf-idf scores": 0.13919148528812864}, {"Date": "2020-01-29T00:00:00", "vocab": "operations", "tf-idf scores": 0.1307752527683921}, {"Date": "2020-01-29T00:00:00", "vocab": "eligible", "tf-idf scores": 0.11372972329439937}, {"Date": "2019-12-11T00:00:00", "vocab": "inflation", "tf-idf scores": 0.29982035762497833}, {"Date": "2019-12-11T00:00:00", "vocab": "representatives", "tf-idf scores": 0.18873156318297404}, {"Date": "2019-12-11T00:00:00", "vocab": "market", "tf-idf scores": 0.17655906215840197}, {"Date": "2019-12-11T00:00:00", "vocab": "economic", "tf-idf scores": 0.15608703483349787}, {"Date": "2019-12-11T00:00:00", "vocab": "policy", "tf-idf scores": 0.1437314712905899}, {"Date": "2019-12-11T00:00:00", "vocab": "remained", "tf-idf scores": 0.13961636704919114}, {"Date": "2019-12-11T00:00:00", "vocab": "october", "tf-idf scores": 0.13706320441854578}, {"Date": "2019-12-11T00:00:00", "vocab": "monetary", "tf-idf scores": 0.13554969326897964}, {"Date": "2019-12-11T00:00:00", "vocab": "labor", "tf-idf scores": 0.13139113025403285}, {"Date": "2019-12-11T00:00:00", "vocab": "listens", "tf-idf scores": 0.12453114987846299}, {"Date": "2019-10-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23426186112219358}, {"Date": "2019-10-30T00:00:00", "vocab": "repo", "tf-idf scores": 0.21358602328369405}, {"Date": "2019-10-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.16687745532168297}, {"Date": "2019-10-30T00:00:00", "vocab": "market", "tf-idf scores": 0.14761564108998354}, {"Date": "2019-10-30T00:00:00", "vocab": "remained", "tf-idf scores": 0.14762383607604465}, {"Date": "2019-10-30T00:00:00", "vocab": "operations", "tf-idf scores": 0.1440723117144722}, {"Date": "2019-10-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.14122244610698417}, {"Date": "2019-10-30T00:00:00", "vocab": "funds", "tf-idf scores": 0.12197787874993501}, {"Date": "2019-10-30T00:00:00", "vocab": "symmetric", "tf-idf scores": 0.11217947752696095}, {"Date": "2019-10-30T00:00:00", "vocab": "financial", "tf-idf scores": 0.10360046456858829}, {"Date": "2019-10-04T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2342173212828446}, {"Date": "2019-10-04T00:00:00", "vocab": "repo", "tf-idf scores": 0.21354931350954462}, {"Date": "2019-10-04T00:00:00", "vocab": "economic", "tf-idf scores": 0.1668400554763839}, {"Date": "2019-10-04T00:00:00", "vocab": "market", "tf-idf scores": 0.14760560202177325}, {"Date": "2019-10-04T00:00:00", "vocab": "remained", "tf-idf scores": 0.14763573089074267}, {"Date": "2019-10-04T00:00:00", "vocab": "operations", "tf-idf scores": 0.14404828132326508}, {"Date": "2019-10-04T00:00:00", "vocab": "policy", "tf-idf scores": 0.14124049166617847}, {"Date": "2019-10-04T00:00:00", "vocab": "funds", "tf-idf scores": 0.12199561230421513}, {"Date": "2019-10-04T00:00:00", "vocab": "symmetric", "tf-idf scores": 0.11216738140002422}, {"Date": "2019-10-04T00:00:00", "vocab": "financial", "tf-idf scores": 0.10364114900243146}, {"Date": "2019-09-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3188801212220249}, {"Date": "2019-09-18T00:00:00", "vocab": "makeup", "tf-idf scores": 0.216824871259575}, {"Date": "2019-09-18T00:00:00", "vocab": "policy", "tf-idf scores": 0.2023996746697927}, {"Date": "2019-09-18T00:00:00", "vocab": "strategies", "tf-idf scores": 0.16997872350435217}, {"Date": "2019-09-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.15638003478411075}, {"Date": "2019-09-18T00:00:00", "vocab": "percent", "tf-idf scores": 0.1368953936390169}, {"Date": "2019-09-18T00:00:00", "vocab": "market", "tf-idf scores": 0.13186814248504744}, {"Date": "2019-09-18T00:00:00", "vocab": "july", "tf-idf scores": 0.12856223362427213}, {"Date": "2019-09-18T00:00:00", "vocab": "elb", "tf-idf scores": 0.11150435525868772}, {"Date": "2019-09-18T00:00:00", "vocab": "growth", "tf-idf scores": 0.10476165279632588}, {"Date": "2019-07-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3414395050368489}, {"Date": "2019-07-31T00:00:00", "vocab": "policy", "tf-idf scores": 0.25605500584087926}, {"Date": "2019-07-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.18021318090344998}, {"Date": "2019-07-31T00:00:00", "vocab": "market", "tf-idf scores": 0.1517314068405435}, {"Date": "2019-07-31T00:00:00", "vocab": "elb", "tf-idf scores": 0.11502217305082836}, {"Date": "2019-07-31T00:00:00", "vocab": "monetary", "tf-idf scores": 0.11387712130907604}, {"Date": "2019-07-31T00:00:00", "vocab": "growth", "tf-idf scores": 0.10802665246306098}, {"Date": "2019-07-31T00:00:00", "vocab": "percent", "tf-idf scores": 0.10754800864248797}, {"Date": "2019-07-31T00:00:00", "vocab": "remained", "tf-idf scores": 0.10753930690035257}, {"Date": "2019-07-31T00:00:00", "vocab": "range", "tf-idf scores": 0.09525748947131724}, {"Date": "2019-06-19T00:00:00", "vocab": "inflation", "tf-idf scores": 0.326080616022315}, {"Date": "2019-06-19T00:00:00", "vocab": "economic", "tf-idf scores": 0.20790133594463447}, {"Date": "2019-06-19T00:00:00", "vocab": "market", "tf-idf scores": 0.17933300843866262}, {"Date": "2019-06-19T00:00:00", "vocab": "facility", "tf-idf scores": 0.14873942507238616}, {"Date": "2019-06-19T00:00:00", "vocab": "noted", "tf-idf scores": 0.1432752092172936}, {"Date": "2019-06-19T00:00:00", "vocab": "symmetric", "tf-idf scores": 0.1424728790529168}, {"Date": "2019-06-19T00:00:00", "vocab": "percent", "tf-idf scores": 0.13865313180161742}, {"Date": "2019-06-19T00:00:00", "vocab": "april", "tf-idf scores": 0.1337407694609424}, {"Date": "2019-06-19T00:00:00", "vocab": "growth", "tf-idf scores": 0.10644931758149037}, {"Date": "2019-06-19T00:00:00", "vocab": "repo", "tf-idf scores": 0.09952089428003855}, {"Date": "2019-05-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.24991543636629968}, {"Date": "2019-05-01T00:00:00", "vocab": "portfolio", "tf-idf scores": 0.23437285707232589}, {"Date": "2019-05-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.1963852215379345}, {"Date": "2019-05-01T00:00:00", "vocab": "maturity", "tf-idf scores": 0.1468684941444574}, {"Date": "2019-05-01T00:00:00", "vocab": "market", "tf-idf scores": 0.14284643406147132}, {"Date": "2019-05-01T00:00:00", "vocab": "composition", "tf-idf scores": 0.13865253064245006}, {"Date": "2019-05-01T00:00:00", "vocab": "financial", "tf-idf scores": 0.13332909701650567}, {"Date": "2019-05-01T00:00:00", "vocab": "growth", "tf-idf scores": 0.13274234977779384}, {"Date": "2019-05-01T00:00:00", "vocab": "shorter", "tf-idf scores": 0.12806482939343705}, {"Date": "2019-05-01T00:00:00", "vocab": "target", "tf-idf scores": 0.1268967043319569}, {"Date": "2019-03-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22097385310366485}, {"Date": "2019-03-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.19636530012669015}, {"Date": "2019-03-20T00:00:00", "vocab": "reserves", "tf-idf scores": 0.1936642256906251}, {"Date": "2019-03-20T00:00:00", "vocab": "growth", "tf-idf scores": 0.16207267690752017}, {"Date": "2019-03-20T00:00:00", "vocab": "market", "tf-idf scores": 0.15780244353245546}, {"Date": "2019-03-20T00:00:00", "vocab": "remained", "tf-idf scores": 0.13683609457375992}, {"Date": "2019-03-20T00:00:00", "vocab": "policy", "tf-idf scores": 0.1297561223290237}, {"Date": "2019-03-20T00:00:00", "vocab": "recent", "tf-idf scores": 0.12980629527130041}, {"Date": "2019-03-20T00:00:00", "vocab": "securities", "tf-idf scores": 0.12286496367767988}, {"Date": "2019-03-20T00:00:00", "vocab": "level", "tf-idf scores": 0.1116208892929171}, {"Date": "2019-01-30T00:00:00", "vocab": "market", "tf-idf scores": 0.21363903899346365}, {"Date": "2019-01-30T00:00:00", "vocab": "selected", "tf-idf scores": 0.20082241074819135}, {"Date": "2019-01-30T00:00:00", "vocab": "foreign", "tf-idf scores": 0.19127563515755666}, {"Date": "2019-01-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18629227669669846}, {"Date": "2019-01-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.15893011152111977}, {"Date": "2019-01-30T00:00:00", "vocab": "financial", "tf-idf scores": 0.1352856884617952}, {"Date": "2019-01-30T00:00:00", "vocab": "currency", "tf-idf scores": 0.13447907529146588}, {"Date": "2019-01-30T00:00:00", "vocab": "bank", "tf-idf scores": 0.1247501604277262}, {"Date": "2019-01-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.12424208703803184}, {"Date": "2019-01-30T00:00:00", "vocab": "securities", "tf-idf scores": 0.10941487418062547}, {"Date": "2018-12-19T00:00:00", "vocab": "economic", "tf-idf scores": 0.22823894370145642}, {"Date": "2018-12-19T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2075186604269346}, {"Date": "2018-12-19T00:00:00", "vocab": "market", "tf-idf scores": 0.20749907357331715}, {"Date": "2018-12-19T00:00:00", "vocab": "growth", "tf-idf scores": 0.15419374336781871}, {"Date": "2018-12-19T00:00:00", "vocab": "financial", "tf-idf scores": 0.13397011626478003}, {"Date": "2018-12-19T00:00:00", "vocab": "policy", "tf-idf scores": 0.1286847443988873}, {"Date": "2018-12-19T00:00:00", "vocab": "liabilities", "tf-idf scores": 0.1250625608930928}, {"Date": "2018-12-19T00:00:00", "vocab": "funds", "tf-idf scores": 0.12040949458341392}, {"Date": "2018-12-19T00:00:00", "vocab": "recent", "tf-idf scores": 0.12037615779194415}, {"Date": "2018-12-19T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11202270175656785}, {"Date": "2018-11-08T00:00:00", "vocab": "regime", "tf-idf scores": 0.22108661421112816}, {"Date": "2018-11-08T00:00:00", "vocab": "market", "tf-idf scores": 0.19540453238427227}, {"Date": "2018-11-08T00:00:00", "vocab": "economic", "tf-idf scores": 0.16634895716148862}, {"Date": "2018-11-08T00:00:00", "vocab": "reserves", "tf-idf scores": 0.16395825335183775}, {"Date": "2018-11-08T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15388074955155243}, {"Date": "2018-11-08T00:00:00", "vocab": "obfr", "tf-idf scores": 0.14329725116553868}, {"Date": "2018-11-08T00:00:00", "vocab": "rates", "tf-idf scores": 0.14140920387416397}, {"Date": "2018-11-08T00:00:00", "vocab": "funds", "tf-idf scores": 0.12473873653763871}, {"Date": "2018-11-08T00:00:00", "vocab": "abundant", "tf-idf scores": 0.12036754325542683}, {"Date": "2018-11-08T00:00:00", "vocab": "effr", "tf-idf scores": 0.12043117596020524}, {"Date": "2018-09-26T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2531819797836006}, {"Date": "2018-09-26T00:00:00", "vocab": "economic", "tf-idf scores": 0.19517158614660796}, {"Date": "2018-09-26T00:00:00", "vocab": "market", "tf-idf scores": 0.15827543830235122}, {"Date": "2018-09-26T00:00:00", "vocab": "policy", "tf-idf scores": 0.14773560492805746}, {"Date": "2018-09-26T00:00:00", "vocab": "growth", "tf-idf scores": 0.1377491149051554}, {"Date": "2018-09-26T00:00:00", "vocab": "funds", "tf-idf scores": 0.13718977866714765}, {"Date": "2018-09-26T00:00:00", "vocab": "continued", "tf-idf scores": 0.1319430384024825}, {"Date": "2018-09-26T00:00:00", "vocab": "labor", "tf-idf scores": 0.1266508539466273}, {"Date": "2018-09-26T00:00:00", "vocab": "percent", "tf-idf scores": 0.12337584779996223}, {"Date": "2018-09-26T00:00:00", "vocab": "recent", "tf-idf scores": 0.11610927362175148}, {"Date": "2018-08-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.20905954109952662}, {"Date": "2018-08-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.20497976302468635}, {"Date": "2018-08-01T00:00:00", "vocab": "elb", "tf-idf scores": 0.19876431369196643}, {"Date": "2018-08-01T00:00:00", "vocab": "policy", "tf-idf scores": 0.17222705837136246}, {"Date": "2018-08-01T00:00:00", "vocab": "market", "tf-idf scores": 0.16812425335531606}, {"Date": "2018-08-01T00:00:00", "vocab": "june", "tf-idf scores": 0.1338896537869895}, {"Date": "2018-08-01T00:00:00", "vocab": "funds", "tf-idf scores": 0.12705159444231182}, {"Date": "2018-08-01T00:00:00", "vocab": "labor", "tf-idf scores": 0.12301757391682744}, {"Date": "2018-08-01T00:00:00", "vocab": "tariff", "tf-idf scores": 0.10356639746564525}, {"Date": "2018-08-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.10248770515378262}, {"Date": "2018-06-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2879470455503191}, {"Date": "2018-06-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.18367996437948536}, {"Date": "2018-06-13T00:00:00", "vocab": "market", "tf-idf scores": 0.16883390911103974}, {"Date": "2018-06-13T00:00:00", "vocab": "percent", "tf-idf scores": 0.15298393597380064}, {"Date": "2018-06-13T00:00:00", "vocab": "funds", "tf-idf scores": 0.14889304806284137}, {"Date": "2018-06-13T00:00:00", "vocab": "recent", "tf-idf scores": 0.13900895165032165}, {"Date": "2018-06-13T00:00:00", "vocab": "labor", "tf-idf scores": 0.13400952471806998}, {"Date": "2018-06-13T00:00:00", "vocab": "growth", "tf-idf scores": 0.1296366435130784}, {"Date": "2018-06-13T00:00:00", "vocab": "policy", "tf-idf scores": 0.1291005932916971}, {"Date": "2018-06-13T00:00:00", "vocab": "real", "tf-idf scores": 0.12192712162979699}, {"Date": "2018-05-02T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3004657946572996}, {"Date": "2018-05-02T00:00:00", "vocab": "economic", "tf-idf scores": 0.18785536593497207}, {"Date": "2018-05-02T00:00:00", "vocab": "funds", "tf-idf scores": 0.17530413227656483}, {"Date": "2018-05-02T00:00:00", "vocab": "market", "tf-idf scores": 0.16691627183722266}, {"Date": "2018-05-02T00:00:00", "vocab": "labor", "tf-idf scores": 0.14187647671923093}, {"Date": "2018-05-02T00:00:00", "vocab": "recent", "tf-idf scores": 0.14196091502438538}, {"Date": "2018-05-02T00:00:00", "vocab": "march", "tf-idf scores": 0.14059658250528415}, {"Date": "2018-05-02T00:00:00", "vocab": "symmetric", "tf-idf scores": 0.1337165442054444}, {"Date": "2018-05-02T00:00:00", "vocab": "percent", "tf-idf scores": 0.13312512758231135}, {"Date": "2018-05-02T00:00:00", "vocab": "policy", "tf-idf scores": 0.12935295423758203}, {"Date": "2018-03-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3031765219536024}, {"Date": "2018-03-21T00:00:00", "vocab": "market", "tf-idf scores": 0.22055194587294183}, {"Date": "2018-03-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.21132868759918175}, {"Date": "2018-03-21T00:00:00", "vocab": "funds", "tf-idf scores": 0.14242928334113183}, {"Date": "2018-03-21T00:00:00", "vocab": "recent", "tf-idf scores": 0.14248025757105895}, {"Date": "2018-03-21T00:00:00", "vocab": "conditions", "tf-idf scores": 0.13325199244775385}, {"Date": "2018-03-21T00:00:00", "vocab": "labor", "tf-idf scores": 0.13331116149733885}, {"Date": "2018-03-21T00:00:00", "vocab": "january", "tf-idf scores": 0.13123838379210467}, {"Date": "2018-03-21T00:00:00", "vocab": "percent", "tf-idf scores": 0.1171926127990988}, {"Date": "2018-03-21T00:00:00", "vocab": "continued", "tf-idf scores": 0.11492487265828139}, {"Date": "2018-01-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3470002020049111}, {"Date": "2018-01-31T00:00:00", "vocab": "selected", "tf-idf scores": 0.21916920086290279}, {"Date": "2018-01-31T00:00:00", "vocab": "market", "tf-idf scores": 0.2141502245798862}, {"Date": "2018-01-31T00:00:00", "vocab": "foreign", "tf-idf scores": 0.20066619963455748}, {"Date": "2018-01-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.1707662082040399}, {"Date": "2018-01-31T00:00:00", "vocab": "bank", "tf-idf scores": 0.14707229091436108}, {"Date": "2018-01-31T00:00:00", "vocab": "currency", "tf-idf scores": 0.14679856890871695}, {"Date": "2018-01-31T00:00:00", "vocab": "eligible", "tf-idf scores": 0.11512074826660895}, {"Date": "2018-01-31T00:00:00", "vocab": "securities", "tf-idf scores": 0.11032155986586431}, {"Date": "2018-01-31T00:00:00", "vocab": "shall", "tf-idf scores": 0.107986280620061}, {"Date": "2017-12-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3924853056981522}, {"Date": "2017-12-13T00:00:00", "vocab": "market", "tf-idf scores": 0.19623154504214443}, {"Date": "2017-12-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.18673642185446604}, {"Date": "2017-12-13T00:00:00", "vocab": "percent", "tf-idf scores": 0.1475916927562839}, {"Date": "2017-12-13T00:00:00", "vocab": "labor", "tf-idf scores": 0.1436278786082235}, {"Date": "2017-12-13T00:00:00", "vocab": "continued", "tf-idf scores": 0.1340959125255917}, {"Date": "2017-12-13T00:00:00", "vocab": "treasury", "tf-idf scores": 0.12833630554537123}, {"Date": "2017-12-13T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1244781858615713}, {"Date": "2017-12-13T00:00:00", "vocab": "range", "tf-idf scores": 0.11743642330765004}, {"Date": "2017-12-13T00:00:00", "vocab": "remained", "tf-idf scores": 0.11486844208590444}, {"Date": "2017-11-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.42369516988044903}, {"Date": "2017-11-01T00:00:00", "vocab": "market", "tf-idf scores": 0.21406500355696254}, {"Date": "2017-11-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.1616186688437523}, {"Date": "2017-11-01T00:00:00", "vocab": "hurricanes", "tf-idf scores": 0.15754927197513927}, {"Date": "2017-11-01T00:00:00", "vocab": "labor", "tf-idf scores": 0.15726041589198422}, {"Date": "2017-11-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.14423200496141544}, {"Date": "2017-11-01T00:00:00", "vocab": "percent", "tf-idf scores": 0.14399492980076295}, {"Date": "2017-11-01T00:00:00", "vocab": "funds", "tf-idf scores": 0.1223610175405564}, {"Date": "2017-11-01T00:00:00", "vocab": "september", "tf-idf scores": 0.1222807547443309}, {"Date": "2017-11-01T00:00:00", "vocab": "remained", "tf-idf scores": 0.11363517538101443}, {"Date": "2017-09-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.30227363574138005}, {"Date": "2017-09-20T00:00:00", "vocab": "market", "tf-idf scores": 0.2001706908590171}, {"Date": "2017-09-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.1844965715977986}, {"Date": "2017-09-20T00:00:00", "vocab": "hurricanes", "tf-idf scores": 0.18397185164000973}, {"Date": "2017-09-20T00:00:00", "vocab": "harvey", "tf-idf scores": 0.18041995325819715}, {"Date": "2017-09-20T00:00:00", "vocab": "july", "tf-idf scores": 0.17948207972980001}, {"Date": "2017-09-20T00:00:00", "vocab": "labor", "tf-idf scores": 0.12954063485041484}, {"Date": "2017-09-20T00:00:00", "vocab": "expected", "tf-idf scores": 0.12563020165892688}, {"Date": "2017-09-20T00:00:00", "vocab": "funds", "tf-idf scores": 0.12173667291378358}, {"Date": "2017-09-20T00:00:00", "vocab": "storms", "tf-idf scores": 0.1048655625226222}, {"Date": "2017-07-26T00:00:00", "vocab": "inflation", "tf-idf scores": 0.32875996384601197}, {"Date": "2017-07-26T00:00:00", "vocab": "market", "tf-idf scores": 0.20663564054148847}, {"Date": "2017-07-26T00:00:00", "vocab": "financial", "tf-idf scores": 0.16583033304202655}, {"Date": "2017-07-26T00:00:00", "vocab": "june", "tf-idf scores": 0.15338869739210326}, {"Date": "2017-07-26T00:00:00", "vocab": "economic", "tf-idf scores": 0.15033668499260733}, {"Date": "2017-07-26T00:00:00", "vocab": "remained", "tf-idf scores": 0.12218179313750342}, {"Date": "2017-07-26T00:00:00", "vocab": "percent", "tf-idf scores": 0.11984989066181367}, {"Date": "2017-07-26T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11746894188419565}, {"Date": "2017-07-26T00:00:00", "vocab": "continued", "tf-idf scores": 0.11273996081285323}, {"Date": "2017-07-26T00:00:00", "vocab": "policy", "tf-idf scores": 0.11276013453316008}, {"Date": "2017-06-14T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3247703454581905}, {"Date": "2017-06-14T00:00:00", "vocab": "market", "tf-idf scores": 0.21654036690213063}, {"Date": "2017-06-14T00:00:00", "vocab": "economic", "tf-idf scores": 0.17760594501236515}, {"Date": "2017-06-14T00:00:00", "vocab": "percent", "tf-idf scores": 0.14725354861396547}, {"Date": "2017-06-14T00:00:00", "vocab": "policy", "tf-idf scores": 0.1472352303716341}, {"Date": "2017-06-14T00:00:00", "vocab": "funds", "tf-idf scores": 0.13857482136452884}, {"Date": "2017-06-14T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1298887620825328}, {"Date": "2017-06-14T00:00:00", "vocab": "continued", "tf-idf scores": 0.12996002520352942}, {"Date": "2017-06-14T00:00:00", "vocab": "labor", "tf-idf scores": 0.1299788119441001}, {"Date": "2017-06-14T00:00:00", "vocab": "normalization", "tf-idf scores": 0.12131236049811517}, {"Date": "2017-05-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2934801411208174}, {"Date": "2017-05-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.19123749917745056}, {"Date": "2017-05-03T00:00:00", "vocab": "march", "tf-idf scores": 0.17972696549694753}, {"Date": "2017-05-03T00:00:00", "vocab": "market", "tf-idf scores": 0.17340195157057497}, {"Date": "2017-05-03T00:00:00", "vocab": "growth", "tf-idf scores": 0.15181262828124642}, {"Date": "2017-05-03T00:00:00", "vocab": "continued", "tf-idf scores": 0.14678751236756163}, {"Date": "2017-05-03T00:00:00", "vocab": "percent", "tf-idf scores": 0.14654393131118185}, {"Date": "2017-05-03T00:00:00", "vocab": "remained", "tf-idf scores": 0.12454373687783353}, {"Date": "2017-05-03T00:00:00", "vocab": "real", "tf-idf scores": 0.11870617136954108}, {"Date": "2017-05-03T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11563751443974145}, {"Date": "2017-03-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2876709187133566}, {"Date": "2017-03-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.21173099498785003}, {"Date": "2017-03-15T00:00:00", "vocab": "market", "tf-idf scores": 0.18778728131863276}, {"Date": "2017-03-15T00:00:00", "vocab": "continued", "tf-idf scores": 0.16782264196020327}, {"Date": "2017-03-15T00:00:00", "vocab": "policy", "tf-idf scores": 0.15981023297181318}, {"Date": "2017-03-15T00:00:00", "vocab": "percent", "tf-idf scores": 0.15291264310244634}, {"Date": "2017-03-15T00:00:00", "vocab": "reinvestments", "tf-idf scores": 0.1464121357123853}, {"Date": "2017-03-15T00:00:00", "vocab": "recent", "tf-idf scores": 0.1318597622495338}, {"Date": "2017-03-15T00:00:00", "vocab": "labor", "tf-idf scores": 0.1278898655623146}, {"Date": "2017-03-15T00:00:00", "vocab": "funds", "tf-idf scores": 0.11985946995231354}, {"Date": "2017-02-01T00:00:00", "vocab": "selected", "tf-idf scores": 0.2241830396972499}, {"Date": "2017-02-01T00:00:00", "vocab": "foreign", "tf-idf scores": 0.2200482769190265}, {"Date": "2017-02-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.19719378754416148}, {"Date": "2017-02-01T00:00:00", "vocab": "market", "tf-idf scores": 0.18286238925412673}, {"Date": "2017-02-01T00:00:00", "vocab": "currency", "tf-idf scores": 0.17473648766199737}, {"Date": "2017-02-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.16005095766361227}, {"Date": "2017-02-01T00:00:00", "vocab": "bank", "tf-idf scores": 0.1520769905672114}, {"Date": "2017-02-01T00:00:00", "vocab": "paragraph", "tf-idf scores": 0.1247347232136434}, {"Date": "2017-02-01T00:00:00", "vocab": "eligible", "tf-idf scores": 0.12131411211343404}, {"Date": "2017-02-01T00:00:00", "vocab": "shall", "tf-idf scores": 0.11380844956151118}, {"Date": "2016-12-14T00:00:00", "vocab": "inflation", "tf-idf scores": 0.28548857887220086}, {"Date": "2016-12-14T00:00:00", "vocab": "market", "tf-idf scores": 0.24152651179457907}, {"Date": "2016-12-14T00:00:00", "vocab": "economic", "tf-idf scores": 0.2196258593298985}, {"Date": "2016-12-14T00:00:00", "vocab": "labor", "tf-idf scores": 0.16688521971088724}, {"Date": "2016-12-14T00:00:00", "vocab": "percent", "tf-idf scores": 0.15408006368940033}, {"Date": "2016-12-14T00:00:00", "vocab": "prices", "tf-idf scores": 0.13235254862304227}, {"Date": "2016-12-14T00:00:00", "vocab": "policy", "tf-idf scores": 0.13179578781031356}, {"Date": "2016-12-14T00:00:00", "vocab": "funds", "tf-idf scores": 0.11862046142485982}, {"Date": "2016-12-14T00:00:00", "vocab": "conditions", "tf-idf scores": 0.10981618611481757}, {"Date": "2016-12-14T00:00:00", "vocab": "recent", "tf-idf scores": 0.10542447137102332}, {"Date": "2016-11-02T00:00:00", "vocab": "inflation", "tf-idf scores": 0.27401222233227857}, {"Date": "2016-11-02T00:00:00", "vocab": "market", "tf-idf scores": 0.181311708123326}, {"Date": "2016-11-02T00:00:00", "vocab": "continued", "tf-idf scores": 0.17290673630087158}, {"Date": "2016-11-02T00:00:00", "vocab": "economic", "tf-idf scores": 0.16861180117237137}, {"Date": "2016-11-02T00:00:00", "vocab": "policy", "tf-idf scores": 0.16026403061918976}, {"Date": "2016-11-02T00:00:00", "vocab": "remained", "tf-idf scores": 0.13918263289171137}, {"Date": "2016-11-02T00:00:00", "vocab": "labor", "tf-idf scores": 0.1306866888024543}, {"Date": "2016-11-02T00:00:00", "vocab": "growth", "tf-idf scores": 0.12282426744458956}, {"Date": "2016-11-02T00:00:00", "vocab": "monetary", "tf-idf scores": 0.11385323773985638}, {"Date": "2016-11-02T00:00:00", "vocab": "implementation", "tf-idf scores": 0.1107039485797397}, {"Date": "2016-09-21T00:00:00", "vocab": "foreign", "tf-idf scores": 0.22608538890594726}, {"Date": "2016-09-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21478405499351555}, {"Date": "2016-09-21T00:00:00", "vocab": "currency", "tf-idf scores": 0.19090014343715012}, {"Date": "2016-09-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.18841636690673308}, {"Date": "2016-09-21T00:00:00", "vocab": "market", "tf-idf scores": 0.18842271184216616}, {"Date": "2016-09-21T00:00:00", "vocab": "continued", "tf-idf scores": 0.14699754552839203}, {"Date": "2016-09-21T00:00:00", "vocab": "labor", "tf-idf scores": 0.13566130153655687}, {"Date": "2016-09-21T00:00:00", "vocab": "policy", "tf-idf scores": 0.13185750101552202}, {"Date": "2016-09-21T00:00:00", "vocab": "recent", "tf-idf scores": 0.11678571873153519}, {"Date": "2016-09-21T00:00:00", "vocab": "standing", "tf-idf scores": 0.10636419183790509}, {"Date": "2016-07-27T00:00:00", "vocab": "market", "tf-idf scores": 0.2580234050840342}, {"Date": "2016-07-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.22971717472171096}, {"Date": "2016-07-27T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22979190563330862}, {"Date": "2016-07-27T00:00:00", "vocab": "brexit", "tf-idf scores": 0.1943097247814802}, {"Date": "2016-07-27T00:00:00", "vocab": "labor", "tf-idf scores": 0.18735839097331353}, {"Date": "2016-07-27T00:00:00", "vocab": "financial", "tf-idf scores": 0.16760067325215053}, {"Date": "2016-07-27T00:00:00", "vocab": "june", "tf-idf scores": 0.15939892489352683}, {"Date": "2016-07-27T00:00:00", "vocab": "prices", "tf-idf scores": 0.1349518650089557}, {"Date": "2016-07-27T00:00:00", "vocab": "policy", "tf-idf scores": 0.13083653929471298}, {"Date": "2016-07-27T00:00:00", "vocab": "continued", "tf-idf scores": 0.12370236581153007}, {"Date": "2016-06-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2643640041518212}, {"Date": "2016-06-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.2602905557189893}, {"Date": "2016-06-15T00:00:00", "vocab": "market", "tf-idf scores": 0.21073240707719323}, {"Date": "2016-06-15T00:00:00", "vocab": "labor", "tf-idf scores": 0.206551588251956}, {"Date": "2016-06-15T00:00:00", "vocab": "april", "tf-idf scores": 0.16574795378068602}, {"Date": "2016-06-15T00:00:00", "vocab": "continued", "tf-idf scores": 0.1363176288091873}, {"Date": "2016-06-15T00:00:00", "vocab": "referendum", "tf-idf scores": 0.12607291819434652}, {"Date": "2016-06-15T00:00:00", "vocab": "percent", "tf-idf scores": 0.12301765499535658}, {"Date": "2016-06-15T00:00:00", "vocab": "growth", "tf-idf scores": 0.12032039962345695}, {"Date": "2016-06-15T00:00:00", "vocab": "recent", "tf-idf scores": 0.11982083623964733}, {"Date": "2016-04-27T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22841752866318996}, {"Date": "2016-04-27T00:00:00", "vocab": "financial", "tf-idf scores": 0.21872483652538702}, {"Date": "2016-04-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.20905013021314486}, {"Date": "2016-04-27T00:00:00", "vocab": "market", "tf-idf scores": 0.19741360144919023}, {"Date": "2016-04-27T00:00:00", "vocab": "continued", "tf-idf scores": 0.17810059711444923}, {"Date": "2016-04-27T00:00:00", "vocab": "growth", "tf-idf scores": 0.16337066516132895}, {"Date": "2016-04-27T00:00:00", "vocab": "labor", "tf-idf scores": 0.15102961138997995}, {"Date": "2016-04-27T00:00:00", "vocab": "prices", "tf-idf scores": 0.14385745195416133}, {"Date": "2016-04-27T00:00:00", "vocab": "recent", "tf-idf scores": 0.1432892944685301}, {"Date": "2016-04-27T00:00:00", "vocab": "march", "tf-idf scores": 0.12396539797678852}, {"Date": "2016-03-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.27414119996621594}, {"Date": "2016-03-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.24269837125042595}, {"Date": "2016-03-16T00:00:00", "vocab": "market", "tf-idf scores": 0.19777870524071461}, {"Date": "2016-03-16T00:00:00", "vocab": "labor", "tf-idf scores": 0.15286051742944115}, {"Date": "2016-03-16T00:00:00", "vocab": "recent", "tf-idf scores": 0.14830200347162134}, {"Date": "2016-03-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.13543456701405296}, {"Date": "2016-03-16T00:00:00", "vocab": "rrps", "tf-idf scores": 0.13346304586360852}, {"Date": "2016-03-16T00:00:00", "vocab": "financial", "tf-idf scores": 0.1315159792207498}, {"Date": "2016-03-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.1258657025676194}, {"Date": "2016-03-16T00:00:00", "vocab": "global", "tf-idf scores": 0.12523196937047285}, {"Date": "2016-01-27T00:00:00", "vocab": "foreign", "tf-idf scores": 0.24340500348571797}, {"Date": "2016-01-27T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23196397645979702}, {"Date": "2016-01-27T00:00:00", "vocab": "market", "tf-idf scores": 0.2061461288694849}, {"Date": "2016-01-27T00:00:00", "vocab": "shall", "tf-idf scores": 0.20514693962929656}, {"Date": "2016-01-27T00:00:00", "vocab": "currency", "tf-idf scores": 0.18511203661801764}, {"Date": "2016-01-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.17468070670235197}, {"Date": "2016-01-27T00:00:00", "vocab": "financial", "tf-idf scores": 0.12715321712041988}, {"Date": "2016-01-27T00:00:00", "vocab": "selected", "tf-idf scores": 0.12254376121939133}, {"Date": "2016-01-27T00:00:00", "vocab": "chairman", "tf-idf scores": 0.1216709919462761}, {"Date": "2016-01-27T00:00:00", "vocab": "eligible", "tf-idf scores": 0.12157398296803724}, {"Date": "2015-12-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3425742643905545}, {"Date": "2015-12-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.23639249795248596}, {"Date": "2015-12-16T00:00:00", "vocab": "market", "tf-idf scores": 0.212330847922961}, {"Date": "2015-12-16T00:00:00", "vocab": "labor", "tf-idf scores": 0.17853284445077036}, {"Date": "2015-12-16T00:00:00", "vocab": "prices", "tf-idf scores": 0.17444406737270327}, {"Date": "2015-12-16T00:00:00", "vocab": "october", "tf-idf scores": 0.13562071128909958}, {"Date": "2015-12-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.12548769670445806}, {"Date": "2015-12-16T00:00:00", "vocab": "energy", "tf-idf scores": 0.11682252046160085}, {"Date": "2015-12-16T00:00:00", "vocab": "activity", "tf-idf scores": 0.11101964693693787}, {"Date": "2015-12-16T00:00:00", "vocab": "policy", "tf-idf scores": 0.10622162372662756}, {"Date": "2015-10-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2619288217197365}, {"Date": "2015-10-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.22641670098792116}, {"Date": "2015-10-28T00:00:00", "vocab": "market", "tf-idf scores": 0.20867103563482078}, {"Date": "2015-10-28T00:00:00", "vocab": "labor", "tf-idf scores": 0.20416445030430458}, {"Date": "2015-10-28T00:00:00", "vocab": "real", "tf-idf scores": 0.14217643802433122}, {"Date": "2015-10-28T00:00:00", "vocab": "policy", "tf-idf scores": 0.13314211278643398}, {"Date": "2015-10-28T00:00:00", "vocab": "financial", "tf-idf scores": 0.12987376314409488}, {"Date": "2015-10-28T00:00:00", "vocab": "continued", "tf-idf scores": 0.11988779903974225}, {"Date": "2015-10-28T00:00:00", "vocab": "prices", "tf-idf scores": 0.1114584814933603}, {"Date": "2015-10-28T00:00:00", "vocab": "range", "tf-idf scores": 0.10894338562558303}, {"Date": "2015-09-17T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3401709042020031}, {"Date": "2015-09-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.2857789757636659}, {"Date": "2015-09-17T00:00:00", "vocab": "market", "tf-idf scores": 0.22228612775487605}, {"Date": "2015-09-17T00:00:00", "vocab": "labor", "tf-idf scores": 0.16784267491238428}, {"Date": "2015-09-17T00:00:00", "vocab": "prices", "tf-idf scores": 0.1549063184035275}, {"Date": "2015-09-17T00:00:00", "vocab": "july", "tf-idf scores": 0.12970054886965304}, {"Date": "2015-09-17T00:00:00", "vocab": "activity", "tf-idf scores": 0.12253623342216523}, {"Date": "2015-09-17T00:00:00", "vocab": "policy", "tf-idf scores": 0.12251326416654064}, {"Date": "2015-09-17T00:00:00", "vocab": "reinvestments", "tf-idf scores": 0.11641732333378474}, {"Date": "2015-09-17T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11340293239130345}, {"Date": "2015-07-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2611990675792382}, {"Date": "2015-07-29T00:00:00", "vocab": "market", "tf-idf scores": 0.23022215328607004}, {"Date": "2015-07-29T00:00:00", "vocab": "labor", "tf-idf scores": 0.2036810218819193}, {"Date": "2015-07-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.19034817821878844}, {"Date": "2015-07-29T00:00:00", "vocab": "reinvestments", "tf-idf scores": 0.1622733890132823}, {"Date": "2015-07-29T00:00:00", "vocab": "continued", "tf-idf scores": 0.14166029415685394}, {"Date": "2015-07-29T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12843546048682272}, {"Date": "2015-07-29T00:00:00", "vocab": "prices", "tf-idf scores": 0.12457538744125991}, {"Date": "2015-07-29T00:00:00", "vocab": "policy", "tf-idf scores": 0.11514351007563302}, {"Date": "2015-07-29T00:00:00", "vocab": "range", "tf-idf scores": 0.10371846406674019}, {"Date": "2015-06-17T00:00:00", "vocab": "inflation", "tf-idf scores": 0.25883566620180853}, {"Date": "2015-06-17T00:00:00", "vocab": "market", "tf-idf scores": 0.24959183813273025}, {"Date": "2015-06-17T00:00:00", "vocab": "labor", "tf-idf scores": 0.22189511478847765}, {"Date": "2015-06-17T00:00:00", "vocab": "real", "tf-idf scores": 0.15789607072593712}, {"Date": "2015-06-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.1571525515281028}, {"Date": "2015-06-17T00:00:00", "vocab": "policy", "tf-idf scores": 0.15255727555729984}, {"Date": "2015-06-17T00:00:00", "vocab": "april", "tf-idf scores": 0.15171599087917534}, {"Date": "2015-06-17T00:00:00", "vocab": "prices", "tf-idf scores": 0.13931699353952545}, {"Date": "2015-06-17T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11557270947729181}, {"Date": "2015-06-17T00:00:00", "vocab": "continued", "tf-idf scores": 0.11557719698174819}, {"Date": "2015-04-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.26123733771511176}, {"Date": "2015-04-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.22920228867028872}, {"Date": "2015-04-29T00:00:00", "vocab": "market", "tf-idf scores": 0.22460731050791286}, {"Date": "2015-04-29T00:00:00", "vocab": "prices", "tf-idf scores": 0.17034611952517012}, {"Date": "2015-04-29T00:00:00", "vocab": "labor", "tf-idf scores": 0.16507892382501266}, {"Date": "2015-04-29T00:00:00", "vocab": "growth", "tf-idf scores": 0.15651411784266825}, {"Date": "2015-04-29T00:00:00", "vocab": "remained", "tf-idf scores": 0.15128994832691017}, {"Date": "2015-04-29T00:00:00", "vocab": "policy", "tf-idf scores": 0.13297337525742142}, {"Date": "2015-04-29T00:00:00", "vocab": "real", "tf-idf scores": 0.13215365828239}, {"Date": "2015-04-29T00:00:00", "vocab": "transitory", "tf-idf scores": 0.11814128967942222}, {"Date": "2015-03-18T00:00:00", "vocab": "rrp", "tf-idf scores": 0.2943995584757664}, {"Date": "2015-03-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2255660234977213}, {"Date": "2015-03-18T00:00:00", "vocab": "market", "tf-idf scores": 0.19214030474338104}, {"Date": "2015-03-18T00:00:00", "vocab": "term", "tf-idf scores": 0.14952061913520792}, {"Date": "2015-03-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.1462150226865815}, {"Date": "2015-03-18T00:00:00", "vocab": "labor", "tf-idf scores": 0.14622678657172747}, {"Date": "2015-03-18T00:00:00", "vocab": "policy", "tf-idf scores": 0.14620344790654777}, {"Date": "2015-03-18T00:00:00", "vocab": "january", "tf-idf scores": 0.1259068117873524}, {"Date": "2015-03-18T00:00:00", "vocab": "normalization", "tf-idf scores": 0.11699173489756003}, {"Date": "2015-03-18T00:00:00", "vocab": "liftoff", "tf-idf scores": 0.11322176617214955}, {"Date": "2015-01-28T00:00:00", "vocab": "rrp", "tf-idf scores": 0.20087573334928643}, {"Date": "2015-01-28T00:00:00", "vocab": "foreign", "tf-idf scores": 0.19529329729573813}, {"Date": "2015-01-28T00:00:00", "vocab": "shall", "tf-idf scores": 0.19269031892407326}, {"Date": "2015-01-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18531187163067894}, {"Date": "2015-01-28T00:00:00", "vocab": "market", "tf-idf scores": 0.1827716038625964}, {"Date": "2015-01-28T00:00:00", "vocab": "policy", "tf-idf scores": 0.15523117792357272}, {"Date": "2015-01-28T00:00:00", "vocab": "currency", "tf-idf scores": 0.15313265927066116}, {"Date": "2015-01-28T00:00:00", "vocab": "operations", "tf-idf scores": 0.13888408234496996}, {"Date": "2015-01-28T00:00:00", "vocab": "selected", "tf-idf scores": 0.11908159795883523}, {"Date": "2015-01-28T00:00:00", "vocab": "bank", "tf-idf scores": 0.11818480383379448}, {"Date": "2014-12-17T00:00:00", "vocab": "inflation", "tf-idf scores": 0.3202345708759561}, {"Date": "2014-12-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.21353818730868995}, {"Date": "2014-12-17T00:00:00", "vocab": "market", "tf-idf scores": 0.2038366251686656}, {"Date": "2014-12-17T00:00:00", "vocab": "prices", "tf-idf scores": 0.15592149779717296}, {"Date": "2014-12-17T00:00:00", "vocab": "labor", "tf-idf scores": 0.15527973744675364}, {"Date": "2014-12-17T00:00:00", "vocab": "october", "tf-idf scores": 0.15333133992263528}, {"Date": "2014-12-17T00:00:00", "vocab": "policy", "tf-idf scores": 0.1407691310321671}, {"Date": "2014-12-17T00:00:00", "vocab": "real", "tf-idf scores": 0.11914649106755451}, {"Date": "2014-12-17T00:00:00", "vocab": "continued", "tf-idf scores": 0.11650057436031834}, {"Date": "2014-12-17T00:00:00", "vocab": "oil", "tf-idf scores": 0.10367924654756493}, {"Date": "2014-10-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2670706297630973}, {"Date": "2014-10-29T00:00:00", "vocab": "market", "tf-idf scores": 0.20978090845027586}, {"Date": "2014-10-29T00:00:00", "vocab": "rrp", "tf-idf scores": 0.19705088201455656}, {"Date": "2014-10-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.16695805169848374}, {"Date": "2014-10-29T00:00:00", "vocab": "policy", "tf-idf scores": 0.12877558430924294}, {"Date": "2014-10-29T00:00:00", "vocab": "september", "tf-idf scores": 0.12607082443174067}, {"Date": "2014-10-29T00:00:00", "vocab": "continued", "tf-idf scores": 0.11922745846980877}, {"Date": "2014-10-29T00:00:00", "vocab": "financial", "tf-idf scores": 0.11549539043771644}, {"Date": "2014-10-29T00:00:00", "vocab": "labor", "tf-idf scores": 0.11443852472129623}, {"Date": "2014-10-29T00:00:00", "vocab": "preannounced", "tf-idf scores": 0.11076859164606363}, {"Date": "2014-09-17T00:00:00", "vocab": "inflation", "tf-idf scores": 0.25982671499766186}, {"Date": "2014-09-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.20301290853579623}, {"Date": "2014-09-17T00:00:00", "vocab": "market", "tf-idf scores": 0.18270709263044133}, {"Date": "2014-09-17T00:00:00", "vocab": "labor", "tf-idf scores": 0.16652301395129251}, {"Date": "2014-09-17T00:00:00", "vocab": "policy", "tf-idf scores": 0.1461496164157041}, {"Date": "2014-09-17T00:00:00", "vocab": "guidance", "tf-idf scores": 0.13444096672485029}, {"Date": "2014-09-17T00:00:00", "vocab": "funds", "tf-idf scores": 0.13404200763754753}, {"Date": "2014-09-17T00:00:00", "vocab": "july", "tf-idf scores": 0.12386659784506586}, {"Date": "2014-09-17T00:00:00", "vocab": "normalization", "tf-idf scores": 0.10337500298260707}, {"Date": "2014-09-17T00:00:00", "vocab": "continued", "tf-idf scores": 0.10155185462354284}, {"Date": "2014-07-30T00:00:00", "vocab": "market", "tf-idf scores": 0.2522651898097037}, {"Date": "2014-07-30T00:00:00", "vocab": "labor", "tf-idf scores": 0.23013560920180015}, {"Date": "2014-07-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22124217530604676}, {"Date": "2014-07-30T00:00:00", "vocab": "second", "tf-idf scores": 0.17731498382215977}, {"Date": "2014-07-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.15488711453671825}, {"Date": "2014-07-30T00:00:00", "vocab": "continued", "tf-idf scores": 0.1505122893040351}, {"Date": "2014-07-30T00:00:00", "vocab": "rrp", "tf-idf scores": 0.13990182297096038}, {"Date": "2014-07-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.11955398263286768}, {"Date": "2014-07-30T00:00:00", "vocab": "normalization", "tf-idf scores": 0.11270568353634887}, {"Date": "2014-07-30T00:00:00", "vocab": "remained", "tf-idf scores": 0.1062546466328642}, {"Date": "2014-06-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2538493531854074}, {"Date": "2014-06-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.21006441122246455}, {"Date": "2014-06-18T00:00:00", "vocab": "policy", "tf-idf scores": 0.19694044131577282}, {"Date": "2014-06-18T00:00:00", "vocab": "market", "tf-idf scores": 0.1838535718510279}, {"Date": "2014-06-18T00:00:00", "vocab": "april", "tf-idf scores": 0.14366808526380892}, {"Date": "2014-06-18T00:00:00", "vocab": "rrp", "tf-idf scores": 0.13835231836900921}, {"Date": "2014-06-18T00:00:00", "vocab": "normalization", "tf-idf scores": 0.122607086913932}, {"Date": "2014-06-18T00:00:00", "vocab": "labor", "tf-idf scores": 0.12256094919332099}, {"Date": "2014-06-18T00:00:00", "vocab": "real", "tf-idf scores": 0.11681196657107752}, {"Date": "2014-06-18T00:00:00", "vocab": "remained", "tf-idf scores": 0.11380155188154377}, {"Date": "2014-04-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2348564436628143}, {"Date": "2014-04-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.19965922844332692}, {"Date": "2014-04-30T00:00:00", "vocab": "market", "tf-idf scores": 0.19377252607037132}, {"Date": "2014-04-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.14684512173465286}, {"Date": "2014-04-30T00:00:00", "vocab": "continued", "tf-idf scores": 0.1292095347113433}, {"Date": "2014-04-30T00:00:00", "vocab": "labor", "tf-idf scores": 0.12921086333964024}, {"Date": "2014-04-30T00:00:00", "vocab": "financial", "tf-idf scores": 0.1244540651739205}, {"Date": "2014-04-30T00:00:00", "vocab": "real", "tf-idf scores": 0.11913332778574681}, {"Date": "2014-04-30T00:00:00", "vocab": "remained", "tf-idf scores": 0.11745360257267431}, {"Date": "2014-04-30T00:00:00", "vocab": "unemployment", "tf-idf scores": 0.11456895675232488}, {"Date": "2014-03-19T00:00:00", "vocab": "economic", "tf-idf scores": 0.2378831318494833}, {"Date": "2014-03-19T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21407899339288497}, {"Date": "2014-03-19T00:00:00", "vocab": "growth", "tf-idf scores": 0.18634395405871423}, {"Date": "2014-03-19T00:00:00", "vocab": "market", "tf-idf scores": 0.18084982613717474}, {"Date": "2014-03-19T00:00:00", "vocab": "policy", "tf-idf scores": 0.13802482530503832}, {"Date": "2014-03-19T00:00:00", "vocab": "unemployment", "tf-idf scores": 0.13679513304790206}, {"Date": "2014-03-19T00:00:00", "vocab": "labor", "tf-idf scores": 0.13323418668235454}, {"Date": "2014-03-19T00:00:00", "vocab": "pace", "tf-idf scores": 0.12850399350452565}, {"Date": "2014-03-19T00:00:00", "vocab": "winter", "tf-idf scores": 0.1272989907441647}, {"Date": "2014-03-19T00:00:00", "vocab": "january", "tf-idf scores": 0.12073237879762223}, {"Date": "2014-03-04T00:00:00", "vocab": "economic", "tf-idf scores": 0.23785652261822854}, {"Date": "2014-03-04T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21406770956056265}, {"Date": "2014-03-04T00:00:00", "vocab": "growth", "tf-idf scores": 0.18633084823031718}, {"Date": "2014-03-04T00:00:00", "vocab": "market", "tf-idf scores": 0.1808066561459116}, {"Date": "2014-03-04T00:00:00", "vocab": "policy", "tf-idf scores": 0.1379644224513016}, {"Date": "2014-03-04T00:00:00", "vocab": "unemployment", "tf-idf scores": 0.1367427490109201}, {"Date": "2014-03-04T00:00:00", "vocab": "labor", "tf-idf scores": 0.13324776888621048}, {"Date": "2014-03-04T00:00:00", "vocab": "pace", "tf-idf scores": 0.12846410604245498}, {"Date": "2014-03-04T00:00:00", "vocab": "winter", "tf-idf scores": 0.12734057411073346}, {"Date": "2014-03-04T00:00:00", "vocab": "january", "tf-idf scores": 0.1207356733976693}, {"Date": "2014-01-29T00:00:00", "vocab": "market", "tf-idf scores": 0.2862014604105687}, {"Date": "2014-01-29T00:00:00", "vocab": "shall", "tf-idf scores": 0.2430482922546158}, {"Date": "2014-01-29T00:00:00", "vocab": "foreign", "tf-idf scores": 0.20604434846218836}, {"Date": "2014-01-29T00:00:00", "vocab": "currency", "tf-idf scores": 0.1799494367425549}, {"Date": "2014-01-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15175244207474153}, {"Date": "2014-01-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.13742523946952798}, {"Date": "2014-01-29T00:00:00", "vocab": "open", "tf-idf scores": 0.1351534990065637}, {"Date": "2014-01-29T00:00:00", "vocab": "bank", "tf-idf scores": 0.12933754271506578}, {"Date": "2014-01-29T00:00:00", "vocab": "securities", "tf-idf scores": 0.12290416641265425}, {"Date": "2014-01-29T00:00:00", "vocab": "policy", "tf-idf scores": 0.12026543729540523}, {"Date": "2013-12-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22223273446823524}, {"Date": "2013-12-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.20205165376070094}, {"Date": "2013-12-18T00:00:00", "vocab": "market", "tf-idf scores": 0.17778934506795419}, {"Date": "2013-12-18T00:00:00", "vocab": "financial", "tf-idf scores": 0.15489470513876172}, {"Date": "2013-12-18T00:00:00", "vocab": "asset", "tf-idf scores": 0.15216128819566038}, {"Date": "2013-12-18T00:00:00", "vocab": "labor", "tf-idf scores": 0.1455117311488781}, {"Date": "2013-12-18T00:00:00", "vocab": "threshold", "tf-idf scores": 0.14124335399862334}, {"Date": "2013-12-18T00:00:00", "vocab": "marginal", "tf-idf scores": 0.14091327211446025}, {"Date": "2013-12-18T00:00:00", "vocab": "purchases", "tf-idf scores": 0.13895584592391602}, {"Date": "2013-12-18T00:00:00", "vocab": "policy", "tf-idf scores": 0.1373868937258576}, {"Date": "2013-10-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.2107583753071906}, {"Date": "2013-10-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.1856281877361391}, {"Date": "2013-10-30T00:00:00", "vocab": "market", "tf-idf scores": 0.17060150058864984}, {"Date": "2013-10-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15051737847351762}, {"Date": "2013-10-30T00:00:00", "vocab": "asset", "tf-idf scores": 0.13805781175201667}, {"Date": "2013-10-30T00:00:00", "vocab": "labor", "tf-idf scores": 0.13549894333263382}, {"Date": "2013-10-30T00:00:00", "vocab": "pace", "tf-idf scores": 0.13048381311591764}, {"Date": "2013-10-30T00:00:00", "vocab": "september", "tf-idf scores": 0.12491371821757702}, {"Date": "2013-10-30T00:00:00", "vocab": "continued", "tf-idf scores": 0.11538937183573375}, {"Date": "2013-10-30T00:00:00", "vocab": "funds", "tf-idf scores": 0.11547021521999913}, {"Date": "2013-10-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.2107253495780376}, {"Date": "2013-10-16T00:00:00", "vocab": "policy", "tf-idf scores": 0.18569740234445756}, {"Date": "2013-10-16T00:00:00", "vocab": "market", "tf-idf scores": 0.1705915595245426}, {"Date": "2013-10-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15053409784939556}, {"Date": "2013-10-16T00:00:00", "vocab": "asset", "tf-idf scores": 0.13803016257768438}, {"Date": "2013-10-16T00:00:00", "vocab": "labor", "tf-idf scores": 0.13548880830535137}, {"Date": "2013-10-16T00:00:00", "vocab": "pace", "tf-idf scores": 0.1304440364519806}, {"Date": "2013-10-16T00:00:00", "vocab": "september", "tf-idf scores": 0.12491528468962498}, {"Date": "2013-10-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.11545763487625882}, {"Date": "2013-10-16T00:00:00", "vocab": "funds", "tf-idf scores": 0.11547105786962691}, {"Date": "2013-09-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.21854306260544848}, {"Date": "2013-09-18T00:00:00", "vocab": "asset", "tf-idf scores": 0.18078566426068832}, {"Date": "2013-09-18T00:00:00", "vocab": "market", "tf-idf scores": 0.17843771018372925}, {"Date": "2013-09-18T00:00:00", "vocab": "july", "tf-idf scores": 0.1700178806373643}, {"Date": "2013-09-18T00:00:00", "vocab": "pace", "tf-idf scores": 0.1560740824456354}, {"Date": "2013-09-18T00:00:00", "vocab": "purchases", "tf-idf scores": 0.1533974299848495}, {"Date": "2013-09-18T00:00:00", "vocab": "financial", "tf-idf scores": 0.1529422968705947}, {"Date": "2013-09-18T00:00:00", "vocab": "policy", "tf-idf scores": 0.15168405679383853}, {"Date": "2013-09-18T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1427911529361782}, {"Date": "2013-09-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1383140530206162}, {"Date": "2013-07-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2293328765817862}, {"Date": "2013-07-31T00:00:00", "vocab": "policy", "tf-idf scores": 0.2140288222529055}, {"Date": "2013-07-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.20386028193500974}, {"Date": "2013-07-31T00:00:00", "vocab": "market", "tf-idf scores": 0.1682546349440958}, {"Date": "2013-07-31T00:00:00", "vocab": "pace", "tf-idf scores": 0.1274833079714481}, {"Date": "2013-07-31T00:00:00", "vocab": "contingent", "tf-idf scores": 0.12663518263729534}, {"Date": "2013-07-31T00:00:00", "vocab": "recent", "tf-idf scores": 0.12235665733585163}, {"Date": "2013-07-31T00:00:00", "vocab": "june", "tf-idf scores": 0.118972870345954}, {"Date": "2013-07-31T00:00:00", "vocab": "asset", "tf-idf scores": 0.11807833965501283}, {"Date": "2013-07-31T00:00:00", "vocab": "second", "tf-idf scores": 0.11669879392900778}, {"Date": "2013-06-19T00:00:00", "vocab": "economic", "tf-idf scores": 0.2653070889227625}, {"Date": "2013-06-19T00:00:00", "vocab": "market", "tf-idf scores": 0.19666810349789168}, {"Date": "2013-06-19T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18301606206687296}, {"Date": "2013-06-19T00:00:00", "vocab": "asset", "tf-idf scores": 0.17876695665516562}, {"Date": "2013-06-19T00:00:00", "vocab": "labor", "tf-idf scores": 0.14181393543878942}, {"Date": "2013-06-19T00:00:00", "vocab": "policy", "tf-idf scores": 0.1372266266873316}, {"Date": "2013-06-19T00:00:00", "vocab": "recent", "tf-idf scores": 0.12350293315256426}, {"Date": "2013-06-19T00:00:00", "vocab": "april", "tf-idf scores": 0.1167909468249867}, {"Date": "2013-06-19T00:00:00", "vocab": "purchases", "tf-idf scores": 0.11398613372575854}, {"Date": "2013-06-19T00:00:00", "vocab": "activity", "tf-idf scores": 0.10985875815314576}, {"Date": "2013-05-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.20424292386945522}, {"Date": "2013-05-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.19258636345934135}, {"Date": "2013-05-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.15755381792943005}, {"Date": "2013-05-01T00:00:00", "vocab": "market", "tf-idf scores": 0.1516592723469619}, {"Date": "2013-05-01T00:00:00", "vocab": "pace", "tf-idf scores": 0.15175794597143774}, {"Date": "2013-05-01T00:00:00", "vocab": "policy", "tf-idf scores": 0.14585807299634368}, {"Date": "2013-05-01T00:00:00", "vocab": "purchases", "tf-idf scores": 0.13145822500288726}, {"Date": "2013-05-01T00:00:00", "vocab": "march", "tf-idf scores": 0.12774399907931577}, {"Date": "2013-05-01T00:00:00", "vocab": "asset", "tf-idf scores": 0.11826807416510707}, {"Date": "2013-05-01T00:00:00", "vocab": "recent", "tf-idf scores": 0.1167175809716288}, {"Date": "2013-03-20T00:00:00", "vocab": "purchases", "tf-idf scores": 0.23323384076736642}, {"Date": "2013-03-20T00:00:00", "vocab": "asset", "tf-idf scores": 0.21355615496113814}, {"Date": "2013-03-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.19673464696261667}, {"Date": "2013-03-20T00:00:00", "vocab": "financial", "tf-idf scores": 0.16366139248002173}, {"Date": "2013-03-20T00:00:00", "vocab": "policy", "tf-idf scores": 0.15733477660417802}, {"Date": "2013-03-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14752269831459985}, {"Date": "2013-03-20T00:00:00", "vocab": "market", "tf-idf scores": 0.13768034330626186}, {"Date": "2013-03-20T00:00:00", "vocab": "monetary", "tf-idf scores": 0.1130770760431042}, {"Date": "2013-03-20T00:00:00", "vocab": "pace", "tf-idf scores": 0.11312539209868128}, {"Date": "2013-03-20T00:00:00", "vocab": "january", "tf-idf scores": 0.10917924330531042}, {"Date": "2013-01-30T00:00:00", "vocab": "shall", "tf-idf scores": 0.26620829561169035}, {"Date": "2013-01-30T00:00:00", "vocab": "market", "tf-idf scores": 0.19722068074523447}, {"Date": "2013-01-30T00:00:00", "vocab": "foreign", "tf-idf scores": 0.19056304103433053}, {"Date": "2013-01-30T00:00:00", "vocab": "open", "tf-idf scores": 0.1477545949110308}, {"Date": "2013-01-30T00:00:00", "vocab": "securities", "tf-idf scores": 0.1435824796036074}, {"Date": "2013-01-30T00:00:00", "vocab": "currency", "tf-idf scores": 0.13435335191061784}, {"Date": "2013-01-30T00:00:00", "vocab": "bank", "tf-idf scores": 0.13431081965690447}, {"Date": "2013-01-30T00:00:00", "vocab": "fourth", "tf-idf scores": 0.13422991058301778}, {"Date": "2013-01-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.13038715491382802}, {"Date": "2013-01-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.12374749937958035}, {"Date": "2012-12-12T00:00:00", "vocab": "thresholds", "tf-idf scores": 0.23614158141778774}, {"Date": "2012-12-12T00:00:00", "vocab": "economic", "tf-idf scores": 0.21391882895570158}, {"Date": "2012-12-12T00:00:00", "vocab": "inflation", "tf-idf scores": 0.16640212659013426}, {"Date": "2012-12-12T00:00:00", "vocab": "purchases", "tf-idf scores": 0.15222492685954497}, {"Date": "2012-12-12T00:00:00", "vocab": "financial", "tf-idf scores": 0.14386080245036995}, {"Date": "2012-12-12T00:00:00", "vocab": "october", "tf-idf scores": 0.13360336080083574}, {"Date": "2012-12-12T00:00:00", "vocab": "market", "tf-idf scores": 0.13310615942031412}, {"Date": "2012-12-12T00:00:00", "vocab": "policy", "tf-idf scores": 0.13311741016135945}, {"Date": "2012-12-12T00:00:00", "vocab": "remained", "tf-idf scores": 0.13310349347504985}, {"Date": "2012-12-12T00:00:00", "vocab": "securities", "tf-idf scores": 0.12895495207083213}, {"Date": "2012-10-24T00:00:00", "vocab": "thresholds", "tf-idf scores": 0.24722054767828544}, {"Date": "2012-10-24T00:00:00", "vocab": "economic", "tf-idf scores": 0.2036435327242876}, {"Date": "2012-10-24T00:00:00", "vocab": "september", "tf-idf scores": 0.16670085957986563}, {"Date": "2012-10-24T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15547561269664686}, {"Date": "2012-10-24T00:00:00", "vocab": "policy", "tf-idf scores": 0.1554591894161762}, {"Date": "2012-10-24T00:00:00", "vocab": "financial", "tf-idf scores": 0.13519151244579333}, {"Date": "2012-10-24T00:00:00", "vocab": "recent", "tf-idf scores": 0.12863818605679403}, {"Date": "2012-10-24T00:00:00", "vocab": "remained", "tf-idf scores": 0.12326583459442651}, {"Date": "2012-10-24T00:00:00", "vocab": "quantitative", "tf-idf scores": 0.12224552516870824}, {"Date": "2012-10-24T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11792628741132279}, {"Date": "2012-09-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.23701469250376378}, {"Date": "2012-09-13T00:00:00", "vocab": "policy", "tf-idf scores": 0.18272638023615}, {"Date": "2012-09-13T00:00:00", "vocab": "purchases", "tf-idf scores": 0.15813331358379462}, {"Date": "2012-09-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1531621475578747}, {"Date": "2012-09-13T00:00:00", "vocab": "august", "tf-idf scores": 0.14891067689885712}, {"Date": "2012-09-13T00:00:00", "vocab": "continued", "tf-idf scores": 0.14322664388504916}, {"Date": "2012-09-13T00:00:00", "vocab": "financial", "tf-idf scores": 0.13952688451290693}, {"Date": "2012-09-13T00:00:00", "vocab": "pace", "tf-idf scores": 0.11852647855538892}, {"Date": "2012-09-13T00:00:00", "vocab": "additional", "tf-idf scores": 0.11818369002895761}, {"Date": "2012-09-13T00:00:00", "vocab": "agency", "tf-idf scores": 0.11127577641746399}, {"Date": "2012-08-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.27164446450044766}, {"Date": "2012-08-01T00:00:00", "vocab": "rules", "tf-idf scores": 0.18253669741706469}, {"Date": "2012-08-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.17242126093943777}, {"Date": "2012-08-01T00:00:00", "vocab": "policy", "tf-idf scores": 0.16716196072839723}, {"Date": "2012-08-01T00:00:00", "vocab": "second", "tf-idf scores": 0.1435469229418875}, {"Date": "2012-08-01T00:00:00", "vocab": "prices", "tf-idf scores": 0.12587929297814204}, {"Date": "2012-08-01T00:00:00", "vocab": "june", "tf-idf scores": 0.12189337635136234}, {"Date": "2012-08-01T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12011747610351457}, {"Date": "2012-08-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.12020309313332934}, {"Date": "2012-08-01T00:00:00", "vocab": "recent", "tf-idf scores": 0.12012742882877704}, {"Date": "2012-06-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.24150750801173407}, {"Date": "2012-06-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2076733523794775}, {"Date": "2012-06-20T00:00:00", "vocab": "april", "tf-idf scores": 0.20251030786924706}, {"Date": "2012-06-20T00:00:00", "vocab": "policy", "tf-idf scores": 0.16425009633105708}, {"Date": "2012-06-20T00:00:00", "vocab": "securities", "tf-idf scores": 0.13641422446056473}, {"Date": "2012-06-20T00:00:00", "vocab": "growth", "tf-idf scores": 0.1358443621983492}, {"Date": "2012-06-20T00:00:00", "vocab": "recent", "tf-idf scores": 0.13047972679787326}, {"Date": "2012-06-20T00:00:00", "vocab": "prices", "tf-idf scores": 0.12128484630903273}, {"Date": "2012-06-20T00:00:00", "vocab": "continued", "tf-idf scores": 0.11590630185997586}, {"Date": "2012-06-20T00:00:00", "vocab": "treasury", "tf-idf scores": 0.10363544391143918}, {"Date": "2012-04-25T00:00:00", "vocab": "economic", "tf-idf scores": 0.2667459406954695}, {"Date": "2012-04-25T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1725973416412908}, {"Date": "2012-04-25T00:00:00", "vocab": "recent", "tf-idf scores": 0.15173458837176781}, {"Date": "2012-04-25T00:00:00", "vocab": "continued", "tf-idf scores": 0.14645625269552837}, {"Date": "2012-04-25T00:00:00", "vocab": "policy", "tf-idf scores": 0.14647123558288838}, {"Date": "2012-04-25T00:00:00", "vocab": "rules", "tf-idf scores": 0.13713241109685975}, {"Date": "2012-04-25T00:00:00", "vocab": "march", "tf-idf scores": 0.13219131836352427}, {"Date": "2012-04-25T00:00:00", "vocab": "prices", "tf-idf scores": 0.13140380759806936}, {"Date": "2012-04-25T00:00:00", "vocab": "unemployment", "tf-idf scores": 0.12349686172394383}, {"Date": "2012-04-25T00:00:00", "vocab": "monetary", "tf-idf scores": 0.1203552598562114}, {"Date": "2012-03-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2185127037918775}, {"Date": "2012-03-13T00:00:00", "vocab": "recent", "tf-idf scores": 0.21282642839464333}, {"Date": "2012-03-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.207059454108429}, {"Date": "2012-03-13T00:00:00", "vocab": "continued", "tf-idf scores": 0.1897624237476435}, {"Date": "2012-03-13T00:00:00", "vocab": "january", "tf-idf scores": 0.18245268903612788}, {"Date": "2012-03-13T00:00:00", "vocab": "remained", "tf-idf scores": 0.15532413686966615}, {"Date": "2012-03-13T00:00:00", "vocab": "gasoline", "tf-idf scores": 0.13949026421633431}, {"Date": "2012-03-13T00:00:00", "vocab": "prices", "tf-idf scores": 0.13291468746845944}, {"Date": "2012-03-13T00:00:00", "vocab": "conditions", "tf-idf scores": 0.13230607944137696}, {"Date": "2012-03-13T00:00:00", "vocab": "market", "tf-idf scores": 0.12653621969892578}, {"Date": "2012-01-25T00:00:00", "vocab": "shall", "tf-idf scores": 0.21366385842486676}, {"Date": "2012-01-25T00:00:00", "vocab": "market", "tf-idf scores": 0.2029908303518377}, {"Date": "2012-01-25T00:00:00", "vocab": "foreign", "tf-idf scores": 0.1820224360551648}, {"Date": "2012-01-25T00:00:00", "vocab": "economic", "tf-idf scores": 0.15404511431712967}, {"Date": "2012-01-25T00:00:00", "vocab": "open", "tf-idf scores": 0.1371435361727725}, {"Date": "2012-01-25T00:00:00", "vocab": "currency", "tf-idf scores": 0.13458326068597753}, {"Date": "2012-01-25T00:00:00", "vocab": "bank", "tf-idf scores": 0.1336211698873872}, {"Date": "2012-01-25T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1330168920342889}, {"Date": "2012-01-25T00:00:00", "vocab": "securities", "tf-idf scores": 0.12654512441536145}, {"Date": "2012-01-25T00:00:00", "vocab": "recent", "tf-idf scores": 0.1260833761114236}, {"Date": "2011-12-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.2215820038102965}, {"Date": "2011-12-13T00:00:00", "vocab": "swap", "tf-idf scores": 0.15485513166386086}, {"Date": "2011-12-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14145378040302464}, {"Date": "2011-12-13T00:00:00", "vocab": "market", "tf-idf scores": 0.14145777164714174}, {"Date": "2011-12-13T00:00:00", "vocab": "european", "tf-idf scores": 0.1386989803573803}, {"Date": "2011-12-13T00:00:00", "vocab": "financial", "tf-idf scores": 0.1284074970416776}, {"Date": "2011-12-13T00:00:00", "vocab": "november", "tf-idf scores": 0.12554945857924846}, {"Date": "2011-12-13T00:00:00", "vocab": "remained", "tf-idf scores": 0.12265273626579953}, {"Date": "2011-12-13T00:00:00", "vocab": "policy", "tf-idf scores": 0.11788896460928595}, {"Date": "2011-12-13T00:00:00", "vocab": "recent", "tf-idf scores": 0.1179065075039796}, {"Date": "2011-11-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.22163381320003941}, {"Date": "2011-11-28T00:00:00", "vocab": "swap", "tf-idf scores": 0.15486398893695918}, {"Date": "2011-11-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14151677613797184}, {"Date": "2011-11-28T00:00:00", "vocab": "market", "tf-idf scores": 0.14150794894853655}, {"Date": "2011-11-28T00:00:00", "vocab": "european", "tf-idf scores": 0.13868265562528628}, {"Date": "2011-11-28T00:00:00", "vocab": "financial", "tf-idf scores": 0.12848331361897777}, {"Date": "2011-11-28T00:00:00", "vocab": "november", "tf-idf scores": 0.12556810788801157}, {"Date": "2011-11-28T00:00:00", "vocab": "remained", "tf-idf scores": 0.12263347595327044}, {"Date": "2011-11-28T00:00:00", "vocab": "policy", "tf-idf scores": 0.11794107229346511}, {"Date": "2011-11-28T00:00:00", "vocab": "recent", "tf-idf scores": 0.11794967034579625}, {"Date": "2011-11-02T00:00:00", "vocab": "economic", "tf-idf scores": 0.20376346055444175}, {"Date": "2011-11-02T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1936365028540263}, {"Date": "2011-11-02T00:00:00", "vocab": "september", "tf-idf scores": 0.15060378625642765}, {"Date": "2011-11-02T00:00:00", "vocab": "remained", "tf-idf scores": 0.14780917268338423}, {"Date": "2011-11-02T00:00:00", "vocab": "securities", "tf-idf scores": 0.1382090983658506}, {"Date": "2011-11-02T00:00:00", "vocab": "continued", "tf-idf scores": 0.13757696076479028}, {"Date": "2011-11-02T00:00:00", "vocab": "policy", "tf-idf scores": 0.1324389935682879}, {"Date": "2011-11-02T00:00:00", "vocab": "prices", "tf-idf scores": 0.12282509295298213}, {"Date": "2011-11-02T00:00:00", "vocab": "growth", "tf-idf scores": 0.11774393845878643}, {"Date": "2011-11-02T00:00:00", "vocab": "financial", "tf-idf scores": 0.11308860687431073}, {"Date": "2011-09-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.20333799556141124}, {"Date": "2011-09-21T00:00:00", "vocab": "securities", "tf-idf scores": 0.17868955444099735}, {"Date": "2011-09-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.15815380169135299}, {"Date": "2011-09-21T00:00:00", "vocab": "ior", "tf-idf scores": 0.1558222991148324}, {"Date": "2011-09-21T00:00:00", "vocab": "august", "tf-idf scores": 0.1361986688810897}, {"Date": "2011-09-21T00:00:00", "vocab": "policy", "tf-idf scores": 0.131059238845773}, {"Date": "2011-09-21T00:00:00", "vocab": "treasury", "tf-idf scores": 0.1308203051925357}, {"Date": "2011-09-21T00:00:00", "vocab": "financial", "tf-idf scores": 0.12306450438511073}, {"Date": "2011-09-21T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1220610930183044}, {"Date": "2011-09-21T00:00:00", "vocab": "maturity", "tf-idf scores": 0.120762536231483}, {"Date": "2011-08-09T00:00:00", "vocab": "economic", "tf-idf scores": 0.2552933179444841}, {"Date": "2011-08-09T00:00:00", "vocab": "inflation", "tf-idf scores": 0.17434034196176604}, {"Date": "2011-08-09T00:00:00", "vocab": "prices", "tf-idf scores": 0.16256749420987066}, {"Date": "2011-08-09T00:00:00", "vocab": "recent", "tf-idf scores": 0.15571012197652004}, {"Date": "2011-08-09T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13072285259244895}, {"Date": "2011-08-09T00:00:00", "vocab": "second", "tf-idf scores": 0.1282774434453134}, {"Date": "2011-08-09T00:00:00", "vocab": "remained", "tf-idf scores": 0.12455297435780084}, {"Date": "2011-08-09T00:00:00", "vocab": "market", "tf-idf scores": 0.11828723808522913}, {"Date": "2011-08-09T00:00:00", "vocab": "real", "tf-idf scores": 0.11297950020992557}, {"Date": "2011-08-09T00:00:00", "vocab": "continued", "tf-idf scores": 0.11207537068409648}, {"Date": "2011-08-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.25521638420107773}, {"Date": "2011-08-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.17430564355692715}, {"Date": "2011-08-01T00:00:00", "vocab": "prices", "tf-idf scores": 0.16257011084608083}, {"Date": "2011-08-01T00:00:00", "vocab": "recent", "tf-idf scores": 0.15565551241050196}, {"Date": "2011-08-01T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13077247384080948}, {"Date": "2011-08-01T00:00:00", "vocab": "second", "tf-idf scores": 0.12831978353887413}, {"Date": "2011-08-01T00:00:00", "vocab": "remained", "tf-idf scores": 0.12452369487930896}, {"Date": "2011-08-01T00:00:00", "vocab": "market", "tf-idf scores": 0.11828964330632617}, {"Date": "2011-08-01T00:00:00", "vocab": "real", "tf-idf scores": 0.11304333329807705}, {"Date": "2011-08-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.1121261277808939}, {"Date": "2011-06-22T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2435247103258877}, {"Date": "2011-06-22T00:00:00", "vocab": "april", "tf-idf scores": 0.23906729100551616}, {"Date": "2011-06-22T00:00:00", "vocab": "economic", "tf-idf scores": 0.19671614467066667}, {"Date": "2011-06-22T00:00:00", "vocab": "dsge", "tf-idf scores": 0.1883802935196854}, {"Date": "2011-06-22T00:00:00", "vocab": "prices", "tf-idf scores": 0.1458139646922792}, {"Date": "2011-06-22T00:00:00", "vocab": "recent", "tf-idf scores": 0.1358282279473194}, {"Date": "2011-06-22T00:00:00", "vocab": "pace", "tf-idf scores": 0.13110377792603287}, {"Date": "2011-06-22T00:00:00", "vocab": "remained", "tf-idf scores": 0.1311357944721744}, {"Date": "2011-06-22T00:00:00", "vocab": "market", "tf-idf scores": 0.12643409718636694}, {"Date": "2011-06-22T00:00:00", "vocab": "models", "tf-idf scores": 0.1215295709888867}, {"Date": "2011-04-27T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2852318745568668}, {"Date": "2011-04-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.19614904100948397}, {"Date": "2011-04-27T00:00:00", "vocab": "securities", "tf-idf scores": 0.14106640804270948}, {"Date": "2011-04-27T00:00:00", "vocab": "continued", "tf-idf scores": 0.13819344221373903}, {"Date": "2011-04-27T00:00:00", "vocab": "policy", "tf-idf scores": 0.13820999260420755}, {"Date": "2011-04-27T00:00:00", "vocab": "remained", "tf-idf scores": 0.13821356582530633}, {"Date": "2011-04-27T00:00:00", "vocab": "february", "tf-idf scores": 0.12665803490115393}, {"Date": "2011-04-27T00:00:00", "vocab": "prices", "tf-idf scores": 0.125359030619097}, {"Date": "2011-04-27T00:00:00", "vocab": "pace", "tf-idf scores": 0.12040829684124826}, {"Date": "2011-04-27T00:00:00", "vocab": "commodity", "tf-idf scores": 0.11619731050796944}, {"Date": "2011-03-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.30193692018641527}, {"Date": "2011-03-15T00:00:00", "vocab": "january", "tf-idf scores": 0.20402453005258595}, {"Date": "2011-03-15T00:00:00", "vocab": "prices", "tf-idf scores": 0.17407343990394997}, {"Date": "2011-03-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.16778337843270977}, {"Date": "2011-03-15T00:00:00", "vocab": "remained", "tf-idf scores": 0.14537950925795756}, {"Date": "2011-03-15T00:00:00", "vocab": "market", "tf-idf scores": 0.13977020170417556}, {"Date": "2011-03-15T00:00:00", "vocab": "labor", "tf-idf scores": 0.12301604203317097}, {"Date": "2011-03-15T00:00:00", "vocab": "recent", "tf-idf scores": 0.11742535696828403}, {"Date": "2011-03-15T00:00:00", "vocab": "fourth", "tf-idf scores": 0.11225127080986934}, {"Date": "2011-03-15T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11185929866679127}, {"Date": "2011-01-26T00:00:00", "vocab": "shall", "tf-idf scores": 0.21717071100439594}, {"Date": "2011-01-26T00:00:00", "vocab": "foreign", "tf-idf scores": 0.21698148935902326}, {"Date": "2011-01-26T00:00:00", "vocab": "market", "tf-idf scores": 0.19570004321446038}, {"Date": "2011-01-26T00:00:00", "vocab": "currency", "tf-idf scores": 0.18018148444120013}, {"Date": "2011-01-26T00:00:00", "vocab": "economic", "tf-idf scores": 0.16013186127376086}, {"Date": "2011-01-26T00:00:00", "vocab": "securities", "tf-idf scores": 0.1487009038786082}, {"Date": "2011-01-26T00:00:00", "vocab": "open", "tf-idf scores": 0.13931333776930785}, {"Date": "2011-01-26T00:00:00", "vocab": "inflation", "tf-idf scores": 0.12808272214228872}, {"Date": "2011-01-26T00:00:00", "vocab": "remained", "tf-idf scores": 0.1209354466387778}, {"Date": "2011-01-26T00:00:00", "vocab": "bank", "tf-idf scores": 0.11794934193390022}, {"Date": "2010-12-14T00:00:00", "vocab": "november", "tf-idf scores": 0.24216314055449722}, {"Date": "2010-12-14T00:00:00", "vocab": "continued", "tf-idf scores": 0.17203193004166678}, {"Date": "2010-12-14T00:00:00", "vocab": "economic", "tf-idf scores": 0.17207885929165007}, {"Date": "2010-12-14T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14831474209526846}, {"Date": "2010-12-14T00:00:00", "vocab": "remained", "tf-idf scores": 0.14833295393430285}, {"Date": "2010-12-14T00:00:00", "vocab": "october", "tf-idf scores": 0.13547668222355563}, {"Date": "2010-12-14T00:00:00", "vocab": "market", "tf-idf scores": 0.13049010124026977}, {"Date": "2010-12-14T00:00:00", "vocab": "securities", "tf-idf scores": 0.1273561604626134}, {"Date": "2010-12-14T00:00:00", "vocab": "prices", "tf-idf scores": 0.12516725971664208}, {"Date": "2010-12-14T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12458121735648399}, {"Date": "2010-11-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1775509140355682}, {"Date": "2010-11-03T00:00:00", "vocab": "remained", "tf-idf scores": 0.17762585028681022}, {"Date": "2010-11-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.15395565819653556}, {"Date": "2010-11-03T00:00:00", "vocab": "securities", "tf-idf scores": 0.14710792370560916}, {"Date": "2010-11-03T00:00:00", "vocab": "continued", "tf-idf scores": 0.12428322546864184}, {"Date": "2010-11-03T00:00:00", "vocab": "september", "tf-idf scores": 0.11969021380394558}, {"Date": "2010-11-03T00:00:00", "vocab": "price", "tf-idf scores": 0.11844262134091404}, {"Date": "2010-11-03T00:00:00", "vocab": "increase", "tf-idf scores": 0.10887638156744417}, {"Date": "2010-11-03T00:00:00", "vocab": "levels", "tf-idf scores": 0.10795571516615873}, {"Date": "2010-11-03T00:00:00", "vocab": "generally", "tf-idf scores": 0.10703389408628675}, {"Date": "2010-10-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1776321907964268}, {"Date": "2010-10-15T00:00:00", "vocab": "remained", "tf-idf scores": 0.17760638380032273}, {"Date": "2010-10-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.15392021640274026}, {"Date": "2010-10-15T00:00:00", "vocab": "securities", "tf-idf scores": 0.14718005872287762}, {"Date": "2010-10-15T00:00:00", "vocab": "continued", "tf-idf scores": 0.12429485574249316}, {"Date": "2010-10-15T00:00:00", "vocab": "september", "tf-idf scores": 0.11972601723355124}, {"Date": "2010-10-15T00:00:00", "vocab": "price", "tf-idf scores": 0.11843630656967756}, {"Date": "2010-10-15T00:00:00", "vocab": "increase", "tf-idf scores": 0.10892815959483347}, {"Date": "2010-10-15T00:00:00", "vocab": "levels", "tf-idf scores": 0.10797553464595588}, {"Date": "2010-10-15T00:00:00", "vocab": "generally", "tf-idf scores": 0.10704010098589183}, {"Date": "2010-09-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.2284380978353732}, {"Date": "2010-09-21T00:00:00", "vocab": "july", "tf-idf scores": 0.21762190398312956}, {"Date": "2010-09-21T00:00:00", "vocab": "august", "tf-idf scores": 0.203259525237507}, {"Date": "2010-09-21T00:00:00", "vocab": "remained", "tf-idf scores": 0.19038524201822948}, {"Date": "2010-09-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18405871553644115}, {"Date": "2010-09-21T00:00:00", "vocab": "continued", "tf-idf scores": 0.1332581494073888}, {"Date": "2010-09-21T00:00:00", "vocab": "prices", "tf-idf scores": 0.12115847302969367}, {"Date": "2010-09-21T00:00:00", "vocab": "intermeeting", "tf-idf scores": 0.11470476682776627}, {"Date": "2010-09-21T00:00:00", "vocab": "financial", "tf-idf scores": 0.10882388117838823}, {"Date": "2010-09-21T00:00:00", "vocab": "real", "tf-idf scores": 0.10836838771219942}, {"Date": "2010-08-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.2239494381874498}, {"Date": "2010-08-10T00:00:00", "vocab": "june", "tf-idf scores": 0.18755334364133874}, {"Date": "2010-08-10T00:00:00", "vocab": "remained", "tf-idf scores": 0.1493188882447245}, {"Date": "2010-08-10T00:00:00", "vocab": "second", "tf-idf scores": 0.14467330836089212}, {"Date": "2010-08-10T00:00:00", "vocab": "recent", "tf-idf scores": 0.1378182829316103}, {"Date": "2010-08-10T00:00:00", "vocab": "data", "tf-idf scores": 0.13329403959476707}, {"Date": "2010-08-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.13266333264233765}, {"Date": "2010-08-10T00:00:00", "vocab": "continued", "tf-idf scores": 0.1320957096752304}, {"Date": "2010-08-10T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13208319935391755}, {"Date": "2010-08-10T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1263955106910903}, {"Date": "2010-06-23T00:00:00", "vocab": "april", "tf-idf scores": 0.2268330391327891}, {"Date": "2010-06-23T00:00:00", "vocab": "economic", "tf-idf scores": 0.1866435111492139}, {"Date": "2010-06-23T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18668072308667025}, {"Date": "2010-06-23T00:00:00", "vocab": "financial", "tf-idf scores": 0.16737428322799291}, {"Date": "2010-06-23T00:00:00", "vocab": "continued", "tf-idf scores": 0.1399943434737985}, {"Date": "2010-06-23T00:00:00", "vocab": "prices", "tf-idf scores": 0.13018895584852447}, {"Date": "2010-06-23T00:00:00", "vocab": "recent", "tf-idf scores": 0.12966765973152605}, {"Date": "2010-06-23T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1245207624875701}, {"Date": "2010-06-23T00:00:00", "vocab": "securities", "tf-idf scores": 0.11135359270367842}, {"Date": "2010-06-23T00:00:00", "vocab": "european", "tf-idf scores": 0.11015551004205443}, {"Date": "2010-05-09T00:00:00", "vocab": "april", "tf-idf scores": 0.226919065319511}, {"Date": "2010-05-09T00:00:00", "vocab": "economic", "tf-idf scores": 0.18667004637021659}, {"Date": "2010-05-09T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18669606385793788}, {"Date": "2010-05-09T00:00:00", "vocab": "financial", "tf-idf scores": 0.16743105712572834}, {"Date": "2010-05-09T00:00:00", "vocab": "continued", "tf-idf scores": 0.14005914801490593}, {"Date": "2010-05-09T00:00:00", "vocab": "prices", "tf-idf scores": 0.1302106900353122}, {"Date": "2010-05-09T00:00:00", "vocab": "recent", "tf-idf scores": 0.12961670895228378}, {"Date": "2010-05-09T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1244375831185479}, {"Date": "2010-05-09T00:00:00", "vocab": "securities", "tf-idf scores": 0.11137036118202397}, {"Date": "2010-05-09T00:00:00", "vocab": "european", "tf-idf scores": 0.11017175532655676}, {"Date": "2010-04-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21501848583108232}, {"Date": "2010-04-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.20420710573877263}, {"Date": "2010-04-28T00:00:00", "vocab": "continued", "tf-idf scores": 0.134365708997363}, {"Date": "2010-04-28T00:00:00", "vocab": "sales", "tf-idf scores": 0.1320154982053771}, {"Date": "2010-04-28T00:00:00", "vocab": "market", "tf-idf scores": 0.12899986255764012}, {"Date": "2010-04-28T00:00:00", "vocab": "recovery", "tf-idf scores": 0.1262368327179625}, {"Date": "2010-04-28T00:00:00", "vocab": "recent", "tf-idf scores": 0.12360673669058189}, {"Date": "2010-04-28T00:00:00", "vocab": "strategy", "tf-idf scores": 0.1131536320669334}, {"Date": "2010-04-28T00:00:00", "vocab": "remained", "tf-idf scores": 0.11289308710964642}, {"Date": "2010-04-28T00:00:00", "vocab": "credit", "tf-idf scores": 0.11179232948948313}, {"Date": "2010-03-16T00:00:00", "vocab": "january", "tf-idf scores": 0.21236821347427726}, {"Date": "2010-03-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.19212921340912645}, {"Date": "2010-03-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.16885522566688696}, {"Date": "2010-03-16T00:00:00", "vocab": "market", "tf-idf scores": 0.1455536267096306}, {"Date": "2010-03-16T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1281385056066791}, {"Date": "2010-03-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.12812971353989872}, {"Date": "2010-03-16T00:00:00", "vocab": "remained", "tf-idf scores": 0.11647886425163691}, {"Date": "2010-03-16T00:00:00", "vocab": "credit", "tf-idf scores": 0.11434590362341891}, {"Date": "2010-03-16T00:00:00", "vocab": "recent", "tf-idf scores": 0.11068220330008649}, {"Date": "2010-03-16T00:00:00", "vocab": "agency", "tf-idf scores": 0.1066101485506292}, {"Date": "2010-01-27T00:00:00", "vocab": "market", "tf-idf scores": 0.20280026527960066}, {"Date": "2010-01-27T00:00:00", "vocab": "shall", "tf-idf scores": 0.19644936843473884}, {"Date": "2010-01-27T00:00:00", "vocab": "foreign", "tf-idf scores": 0.19310085326855592}, {"Date": "2010-01-27T00:00:00", "vocab": "securities", "tf-idf scores": 0.16365635172425702}, {"Date": "2010-01-27T00:00:00", "vocab": "currency", "tf-idf scores": 0.1462244149535311}, {"Date": "2010-01-27T00:00:00", "vocab": "open", "tf-idf scores": 0.12612488518284415}, {"Date": "2010-01-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.12551445204816478}, {"Date": "2010-01-27T00:00:00", "vocab": "inflation", "tf-idf scores": 0.12227831355804276}, {"Date": "2010-01-27T00:00:00", "vocab": "ioer", "tf-idf scores": 0.11848823001265948}, {"Date": "2010-01-27T00:00:00", "vocab": "bank", "tf-idf scores": 0.10670846246356569}, {"Date": "2009-12-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22382590101768882}, {"Date": "2009-12-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.21898494352398795}, {"Date": "2009-12-15T00:00:00", "vocab": "november", "tf-idf scores": 0.15547881859390725}, {"Date": "2009-12-15T00:00:00", "vocab": "continued", "tf-idf scores": 0.1362526997582547}, {"Date": "2009-12-15T00:00:00", "vocab": "market", "tf-idf scores": 0.1362816484523177}, {"Date": "2009-12-15T00:00:00", "vocab": "credit", "tf-idf scores": 0.12364270734864471}, {"Date": "2009-12-15T00:00:00", "vocab": "recent", "tf-idf scores": 0.11680307197791796}, {"Date": "2009-12-15T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11189648642057426}, {"Date": "2009-12-15T00:00:00", "vocab": "remained", "tf-idf scores": 0.11197900208044388}, {"Date": "2009-12-15T00:00:00", "vocab": "increased", "tf-idf scores": 0.10758279314559913}, {"Date": "2009-11-04T00:00:00", "vocab": "continued", "tf-idf scores": 0.18987197536898884}, {"Date": "2009-11-04T00:00:00", "vocab": "economic", "tf-idf scores": 0.17088945442248044}, {"Date": "2009-11-04T00:00:00", "vocab": "agency", "tf-idf scores": 0.14257373331413753}, {"Date": "2009-11-04T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13292122503078746}, {"Date": "2009-11-04T00:00:00", "vocab": "credit", "tf-idf scores": 0.13159939100339543}, {"Date": "2009-11-04T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1266151942948785}, {"Date": "2009-11-04T00:00:00", "vocab": "market", "tf-idf scores": 0.12658855798332344}, {"Date": "2009-11-04T00:00:00", "vocab": "recovery", "tf-idf scores": 0.12082614039169735}, {"Date": "2009-11-04T00:00:00", "vocab": "remained", "tf-idf scores": 0.12033173417530484}, {"Date": "2009-11-04T00:00:00", "vocab": "tools", "tf-idf scores": 0.11575796581048758}, {"Date": "2009-09-22T00:00:00", "vocab": "economic", "tf-idf scores": 0.17559927205338355}, {"Date": "2009-09-22T00:00:00", "vocab": "market", "tf-idf scores": 0.1634505337978973}, {"Date": "2009-09-22T00:00:00", "vocab": "august", "tf-idf scores": 0.1482638729899044}, {"Date": "2009-09-22T00:00:00", "vocab": "continued", "tf-idf scores": 0.13324701698483463}, {"Date": "2009-09-22T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1332109024184239}, {"Date": "2009-09-22T00:00:00", "vocab": "credit", "tf-idf scores": 0.13284522454908387}, {"Date": "2009-09-22T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12714113658501142}, {"Date": "2009-09-22T00:00:00", "vocab": "remained", "tf-idf scores": 0.12112559156568084}, {"Date": "2009-09-22T00:00:00", "vocab": "financial", "tf-idf scores": 0.11605146807435537}, {"Date": "2009-09-22T00:00:00", "vocab": "prices", "tf-idf scores": 0.11556199960202332}, {"Date": "2009-08-11T00:00:00", "vocab": "talf", "tf-idf scores": 0.16923576576429256}, {"Date": "2009-08-11T00:00:00", "vocab": "credit", "tf-idf scores": 0.168996867403745}, {"Date": "2009-08-11T00:00:00", "vocab": "second", "tf-idf scores": 0.15987584473379332}, {"Date": "2009-08-11T00:00:00", "vocab": "economic", "tf-idf scores": 0.1462823764556812}, {"Date": "2009-08-11T00:00:00", "vocab": "continued", "tf-idf scores": 0.13964862192674687}, {"Date": "2009-08-11T00:00:00", "vocab": "july", "tf-idf scores": 0.12677368116099502}, {"Date": "2009-08-11T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12639385420871907}, {"Date": "2009-08-11T00:00:00", "vocab": "market", "tf-idf scores": 0.1264225044990499}, {"Date": "2009-08-11T00:00:00", "vocab": "markets", "tf-idf scores": 0.11969653977986304}, {"Date": "2009-08-11T00:00:00", "vocab": "remained", "tf-idf scores": 0.11977754027226259}, {"Date": "2009-06-24T00:00:00", "vocab": "market", "tf-idf scores": 0.23805064339861354}, {"Date": "2009-06-24T00:00:00", "vocab": "april", "tf-idf scores": 0.15431769778216026}, {"Date": "2009-06-24T00:00:00", "vocab": "economic", "tf-idf scores": 0.14286042981056965}, {"Date": "2009-06-24T00:00:00", "vocab": "financial", "tf-idf scores": 0.1387683001134066}, {"Date": "2009-06-24T00:00:00", "vocab": "likely", "tf-idf scores": 0.13756456208502985}, {"Date": "2009-06-24T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12695133268227057}, {"Date": "2009-06-24T00:00:00", "vocab": "securities", "tf-idf scores": 0.12552301825838597}, {"Date": "2009-06-24T00:00:00", "vocab": "programs", "tf-idf scores": 0.12228355585064915}, {"Date": "2009-06-24T00:00:00", "vocab": "remained", "tf-idf scores": 0.12170793932566475}, {"Date": "2009-06-24T00:00:00", "vocab": "credit", "tf-idf scores": 0.11612845682249907}, {"Date": "2009-06-03T00:00:00", "vocab": "market", "tf-idf scores": 0.23809571887161743}, {"Date": "2009-06-03T00:00:00", "vocab": "april", "tf-idf scores": 0.1542765832151414}, {"Date": "2009-06-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.14281132491239065}, {"Date": "2009-06-03T00:00:00", "vocab": "financial", "tf-idf scores": 0.1388062961382945}, {"Date": "2009-06-03T00:00:00", "vocab": "likely", "tf-idf scores": 0.13753211126925177}, {"Date": "2009-06-03T00:00:00", "vocab": "conditions", "tf-idf scores": 0.12696676074831517}, {"Date": "2009-06-03T00:00:00", "vocab": "securities", "tf-idf scores": 0.1254994459188052}, {"Date": "2009-06-03T00:00:00", "vocab": "programs", "tf-idf scores": 0.12224402411650112}, {"Date": "2009-06-03T00:00:00", "vocab": "remained", "tf-idf scores": 0.12174501673680274}, {"Date": "2009-06-03T00:00:00", "vocab": "credit", "tf-idf scores": 0.11611024579178028}, {"Date": "2009-04-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.23529747043408467}, {"Date": "2009-04-29T00:00:00", "vocab": "financial", "tf-idf scores": 0.21099775554402478}, {"Date": "2009-04-29T00:00:00", "vocab": "credit", "tf-idf scores": 0.16607020829627908}, {"Date": "2009-04-29T00:00:00", "vocab": "march", "tf-idf scores": 0.15418061409758754}, {"Date": "2009-04-29T00:00:00", "vocab": "market", "tf-idf scores": 0.1372724517962709}, {"Date": "2009-04-29T00:00:00", "vocab": "securities", "tf-idf scores": 0.1329109017198317}, {"Date": "2009-04-29T00:00:00", "vocab": "conditions", "tf-idf scores": 0.13076427706805335}, {"Date": "2009-04-29T00:00:00", "vocab": "bank", "tf-idf scores": 0.10511159328813988}, {"Date": "2009-04-29T00:00:00", "vocab": "markets", "tf-idf scores": 0.10457790655615204}, {"Date": "2009-04-29T00:00:00", "vocab": "remained", "tf-idf scores": 0.09811078187098671}, {"Date": "2009-03-17T00:00:00", "vocab": "talf", "tf-idf scores": 0.24582292343018183}, {"Date": "2009-03-17T00:00:00", "vocab": "financial", "tf-idf scores": 0.15601018335431052}, {"Date": "2009-03-17T00:00:00", "vocab": "january", "tf-idf scores": 0.14306640248437197}, {"Date": "2009-03-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.13530072756438283}, {"Date": "2009-03-17T00:00:00", "vocab": "market", "tf-idf scores": 0.13528875982382202}, {"Date": "2009-03-17T00:00:00", "vocab": "purchase", "tf-idf scores": 0.13157160701986065}, {"Date": "2009-03-17T00:00:00", "vocab": "bank", "tf-idf scores": 0.129390282313004}, {"Date": "2009-03-17T00:00:00", "vocab": "purchases", "tf-idf scores": 0.12219326578605506}, {"Date": "2009-03-17T00:00:00", "vocab": "fourth", "tf-idf scores": 0.12008625502756752}, {"Date": "2009-03-17T00:00:00", "vocab": "billion", "tf-idf scores": 0.11627473356639412}, {"Date": "2009-02-07T00:00:00", "vocab": "talf", "tf-idf scores": 0.24586036242519013}, {"Date": "2009-02-07T00:00:00", "vocab": "financial", "tf-idf scores": 0.15594923322146703}, {"Date": "2009-02-07T00:00:00", "vocab": "january", "tf-idf scores": 0.1430090844884071}, {"Date": "2009-02-07T00:00:00", "vocab": "economic", "tf-idf scores": 0.13533758719951294}, {"Date": "2009-02-07T00:00:00", "vocab": "market", "tf-idf scores": 0.13529980320609072}, {"Date": "2009-02-07T00:00:00", "vocab": "purchase", "tf-idf scores": 0.13153609636528307}, {"Date": "2009-02-07T00:00:00", "vocab": "bank", "tf-idf scores": 0.12938625742499352}, {"Date": "2009-02-07T00:00:00", "vocab": "purchases", "tf-idf scores": 0.12220566388976177}, {"Date": "2009-02-07T00:00:00", "vocab": "fourth", "tf-idf scores": 0.1200756113011811}, {"Date": "2009-02-07T00:00:00", "vocab": "billion", "tf-idf scores": 0.11624228806515122}, {"Date": "2009-01-28T00:00:00", "vocab": "market", "tf-idf scores": 0.24588482757418376}, {"Date": "2009-01-28T00:00:00", "vocab": "shall", "tf-idf scores": 0.2143812104939659}, {"Date": "2009-01-28T00:00:00", "vocab": "foreign", "tf-idf scores": 0.20015462130801875}, {"Date": "2009-01-28T00:00:00", "vocab": "currency", "tf-idf scores": 0.17793899090546256}, {"Date": "2009-01-28T00:00:00", "vocab": "securities", "tf-idf scores": 0.15473260839444097}, {"Date": "2009-01-28T00:00:00", "vocab": "open", "tf-idf scores": 0.14108746143189677}, {"Date": "2009-01-28T00:00:00", "vocab": "credit", "tf-idf scores": 0.12980626299291395}, {"Date": "2009-01-28T00:00:00", "vocab": "financial", "tf-idf scores": 0.12759786090688985}, {"Date": "2009-01-28T00:00:00", "vocab": "programs", "tf-idf scores": 0.12622409340286458}, {"Date": "2009-01-28T00:00:00", "vocab": "chairman", "tf-idf scores": 0.11612590224568556}, {"Date": "2009-01-16T00:00:00", "vocab": "market", "tf-idf scores": 0.2458189764271451}, {"Date": "2009-01-16T00:00:00", "vocab": "shall", "tf-idf scores": 0.2143915073640813}, {"Date": "2009-01-16T00:00:00", "vocab": "foreign", "tf-idf scores": 0.20016002103172179}, {"Date": "2009-01-16T00:00:00", "vocab": "currency", "tf-idf scores": 0.1779463815488199}, {"Date": "2009-01-16T00:00:00", "vocab": "securities", "tf-idf scores": 0.15472747056337377}, {"Date": "2009-01-16T00:00:00", "vocab": "open", "tf-idf scores": 0.1410678717355203}, {"Date": "2009-01-16T00:00:00", "vocab": "credit", "tf-idf scores": 0.12979664405226296}, {"Date": "2009-01-16T00:00:00", "vocab": "financial", "tf-idf scores": 0.1275695648917982}, {"Date": "2009-01-16T00:00:00", "vocab": "programs", "tf-idf scores": 0.1262335049678656}, {"Date": "2009-01-16T00:00:00", "vocab": "chairman", "tf-idf scores": 0.1161285521110765}, {"Date": "2008-12-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.21304962647426046}, {"Date": "2008-12-16T00:00:00", "vocab": "market", "tf-idf scores": 0.17516096460096878}, {"Date": "2008-12-16T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1562935560524523}, {"Date": "2008-12-16T00:00:00", "vocab": "financial", "tf-idf scores": 0.15287452446251093}, {"Date": "2008-12-16T00:00:00", "vocab": "credit", "tf-idf scores": 0.13674684531581602}, {"Date": "2008-12-16T00:00:00", "vocab": "prices", "tf-idf scores": 0.1331901325285197}, {"Date": "2008-12-16T00:00:00", "vocab": "drops", "tf-idf scores": 0.13104186408618243}, {"Date": "2008-12-16T00:00:00", "vocab": "decline", "tf-idf scores": 0.12095908525443878}, {"Date": "2008-12-16T00:00:00", "vocab": "funds", "tf-idf scores": 0.11843409924750055}, {"Date": "2008-12-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11844182111035914}, {"Date": "2008-10-29T00:00:00", "vocab": "financial", "tf-idf scores": 0.22682897176277733}, {"Date": "2008-10-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.22021652416049467}, {"Date": "2008-10-29T00:00:00", "vocab": "market", "tf-idf scores": 0.2108488040618595}, {"Date": "2008-10-29T00:00:00", "vocab": "credit", "tf-idf scores": 0.13524982785532244}, {"Date": "2008-10-29T00:00:00", "vocab": "september", "tf-idf scores": 0.12387698798648447}, {"Date": "2008-10-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.12178946698792813}, {"Date": "2008-10-29T00:00:00", "vocab": "october", "tf-idf scores": 0.1151890984920427}, {"Date": "2008-10-29T00:00:00", "vocab": "liquidity", "tf-idf scores": 0.1105960002969055}, {"Date": "2008-10-29T00:00:00", "vocab": "growth", "tf-idf scores": 0.10358102273160573}, {"Date": "2008-10-29T00:00:00", "vocab": "conditions", "tf-idf scores": 0.10314706625779756}, {"Date": "2008-10-07T00:00:00", "vocab": "financial", "tf-idf scores": 0.22683937924440234}, {"Date": "2008-10-07T00:00:00", "vocab": "economic", "tf-idf scores": 0.2202229089401946}, {"Date": "2008-10-07T00:00:00", "vocab": "market", "tf-idf scores": 0.2107859071609847}, {"Date": "2008-10-07T00:00:00", "vocab": "credit", "tf-idf scores": 0.13532408885293437}, {"Date": "2008-10-07T00:00:00", "vocab": "september", "tf-idf scores": 0.12391625206796145}, {"Date": "2008-10-07T00:00:00", "vocab": "inflation", "tf-idf scores": 0.12179302293061664}, {"Date": "2008-10-07T00:00:00", "vocab": "october", "tf-idf scores": 0.11521225739851178}, {"Date": "2008-10-07T00:00:00", "vocab": "liquidity", "tf-idf scores": 0.11058991304277375}, {"Date": "2008-10-07T00:00:00", "vocab": "growth", "tf-idf scores": 0.10359009085301307}, {"Date": "2008-10-07T00:00:00", "vocab": "conditions", "tf-idf scores": 0.1031427569129856}, {"Date": "2008-09-29T00:00:00", "vocab": "financial", "tf-idf scores": 0.22686760476209825}, {"Date": "2008-09-29T00:00:00", "vocab": "economic", "tf-idf scores": 0.22019118046059183}, {"Date": "2008-09-29T00:00:00", "vocab": "market", "tf-idf scores": 0.21088124759811247}, {"Date": "2008-09-29T00:00:00", "vocab": "credit", "tf-idf scores": 0.13533232672150813}, {"Date": "2008-09-29T00:00:00", "vocab": "september", "tf-idf scores": 0.12388585995775288}, {"Date": "2008-09-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.12182151421404262}, {"Date": "2008-09-29T00:00:00", "vocab": "october", "tf-idf scores": 0.11521416888550455}, {"Date": "2008-09-29T00:00:00", "vocab": "liquidity", "tf-idf scores": 0.11059277209944882}, {"Date": "2008-09-29T00:00:00", "vocab": "growth", "tf-idf scores": 0.1035921936622735}, {"Date": "2008-09-29T00:00:00", "vocab": "conditions", "tf-idf scores": 0.10314666129611318}, {"Date": "2008-09-16T00:00:00", "vocab": "financial", "tf-idf scores": 0.23564913131712303}, {"Date": "2008-09-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.22708726650360173}, {"Date": "2008-09-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.21101224495676477}, {"Date": "2008-09-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.2035133043850611}, {"Date": "2008-09-16T00:00:00", "vocab": "august", "tf-idf scores": 0.19878593707608108}, {"Date": "2008-09-16T00:00:00", "vocab": "prices", "tf-idf scores": 0.17407134462599455}, {"Date": "2008-09-16T00:00:00", "vocab": "market", "tf-idf scores": 0.12811536097750711}, {"Date": "2008-09-16T00:00:00", "vocab": "strains", "tf-idf scores": 0.12647041512395396}, {"Date": "2008-09-16T00:00:00", "vocab": "credit", "tf-idf scores": 0.12185817544872926}, {"Date": "2008-09-16T00:00:00", "vocab": "real", "tf-idf scores": 0.12069201250609957}, {"Date": "2008-08-08T00:00:00", "vocab": "inflation", "tf-idf scores": 0.29195896689454}, {"Date": "2008-08-08T00:00:00", "vocab": "second", "tf-idf scores": 0.17941656286532537}, {"Date": "2008-08-08T00:00:00", "vocab": "prices", "tf-idf scores": 0.17882059809160009}, {"Date": "2008-08-08T00:00:00", "vocab": "growth", "tf-idf scores": 0.17169288620143316}, {"Date": "2008-08-08T00:00:00", "vocab": "financial", "tf-idf scores": 0.1651987278438133}, {"Date": "2008-08-08T00:00:00", "vocab": "june", "tf-idf scores": 0.13294235499520357}, {"Date": "2008-08-08T00:00:00", "vocab": "continued", "tf-idf scores": 0.12818317880489283}, {"Date": "2008-08-08T00:00:00", "vocab": "economic", "tf-idf scores": 0.12817481887964327}, {"Date": "2008-08-08T00:00:00", "vocab": "market", "tf-idf scores": 0.12108954386457488}, {"Date": "2008-08-08T00:00:00", "vocab": "remained", "tf-idf scores": 0.12109416886800328}, {"Date": "2008-07-24T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2920164480762409}, {"Date": "2008-07-24T00:00:00", "vocab": "second", "tf-idf scores": 0.1794070834674872}, {"Date": "2008-07-24T00:00:00", "vocab": "prices", "tf-idf scores": 0.17879845513676873}, {"Date": "2008-07-24T00:00:00", "vocab": "growth", "tf-idf scores": 0.17169012451097246}, {"Date": "2008-07-24T00:00:00", "vocab": "financial", "tf-idf scores": 0.16521133344514927}, {"Date": "2008-07-24T00:00:00", "vocab": "june", "tf-idf scores": 0.1329658042425827}, {"Date": "2008-07-24T00:00:00", "vocab": "continued", "tf-idf scores": 0.12822693391561457}, {"Date": "2008-07-24T00:00:00", "vocab": "economic", "tf-idf scores": 0.12823301909259197}, {"Date": "2008-07-24T00:00:00", "vocab": "market", "tf-idf scores": 0.12110449950137792}, {"Date": "2008-07-24T00:00:00", "vocab": "remained", "tf-idf scores": 0.1210658346209362}, {"Date": "2008-06-25T00:00:00", "vocab": "inflation", "tf-idf scores": 0.28886410677045804}, {"Date": "2008-06-25T00:00:00", "vocab": "april", "tf-idf scores": 0.26331637429970217}, {"Date": "2008-06-25T00:00:00", "vocab": "economic", "tf-idf scores": 0.18058109740707481}, {"Date": "2008-06-25T00:00:00", "vocab": "credit", "tf-idf scores": 0.17379941044996342}, {"Date": "2008-06-25T00:00:00", "vocab": "financial", "tf-idf scores": 0.16392861364280467}, {"Date": "2008-06-25T00:00:00", "vocab": "growth", "tf-idf scores": 0.145102154106056}, {"Date": "2008-06-25T00:00:00", "vocab": "prices", "tf-idf scores": 0.1450727744216754}, {"Date": "2008-06-25T00:00:00", "vocab": "recent", "tf-idf scores": 0.13848362948982487}, {"Date": "2008-06-25T00:00:00", "vocab": "remained", "tf-idf scores": 0.13844007566467623}, {"Date": "2008-06-25T00:00:00", "vocab": "market", "tf-idf scores": 0.12641822791769858}, {"Date": "2008-04-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.24881383247629144}, {"Date": "2008-04-30T00:00:00", "vocab": "march", "tf-idf scores": 0.20957665034364942}, {"Date": "2008-04-30T00:00:00", "vocab": "financial", "tf-idf scores": 0.196430843340632}, {"Date": "2008-04-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.17310428975823852}, {"Date": "2008-04-30T00:00:00", "vocab": "growth", "tf-idf scores": 0.15749958823368634}, {"Date": "2008-04-30T00:00:00", "vocab": "prices", "tf-idf scores": 0.15207589255590837}, {"Date": "2008-04-30T00:00:00", "vocab": "remained", "tf-idf scores": 0.14063734390962998}, {"Date": "2008-04-30T00:00:00", "vocab": "credit", "tf-idf scores": 0.13739382408221462}, {"Date": "2008-04-30T00:00:00", "vocab": "markets", "tf-idf scores": 0.1352288914582937}, {"Date": "2008-04-30T00:00:00", "vocab": "recent", "tf-idf scores": 0.12984345071647016}, {"Date": "2008-03-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23605174696642814}, {"Date": "2008-03-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.20239578753543883}, {"Date": "2008-03-18T00:00:00", "vocab": "prices", "tf-idf scores": 0.19648580780091}, {"Date": "2008-03-18T00:00:00", "vocab": "january", "tf-idf scores": 0.16044915124894946}, {"Date": "2008-03-18T00:00:00", "vocab": "financial", "tf-idf scores": 0.15652837836654163}, {"Date": "2008-03-18T00:00:00", "vocab": "credit", "tf-idf scores": 0.14800886365253743}, {"Date": "2008-03-18T00:00:00", "vocab": "real", "tf-idf scores": 0.136849590966607}, {"Date": "2008-03-18T00:00:00", "vocab": "market", "tf-idf scores": 0.12821652524206484}, {"Date": "2008-03-18T00:00:00", "vocab": "growth", "tf-idf scores": 0.12195320556550349}, {"Date": "2008-03-18T00:00:00", "vocab": "markets", "tf-idf scores": 0.12142199133165808}, {"Date": "2008-03-10T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23612323036216223}, {"Date": "2008-03-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.20234878644855864}, {"Date": "2008-03-10T00:00:00", "vocab": "prices", "tf-idf scores": 0.19646416208437603}, {"Date": "2008-03-10T00:00:00", "vocab": "january", "tf-idf scores": 0.16045966756630503}, {"Date": "2008-03-10T00:00:00", "vocab": "financial", "tf-idf scores": 0.156484103492275}, {"Date": "2008-03-10T00:00:00", "vocab": "credit", "tf-idf scores": 0.14807257378461552}, {"Date": "2008-03-10T00:00:00", "vocab": "real", "tf-idf scores": 0.1367994669001717}, {"Date": "2008-03-10T00:00:00", "vocab": "market", "tf-idf scores": 0.12820542665074353}, {"Date": "2008-03-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.12198613280382595}, {"Date": "2008-03-10T00:00:00", "vocab": "markets", "tf-idf scores": 0.1214849265097456}, {"Date": "2008-01-30T00:00:00", "vocab": "shall", "tf-idf scores": 0.25068645600975425}, {"Date": "2008-01-30T00:00:00", "vocab": "foreign", "tf-idf scores": 0.22830706294976025}, {"Date": "2008-01-30T00:00:00", "vocab": "currency", "tf-idf scores": 0.18561950029499538}, {"Date": "2008-01-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.1732058527156345}, {"Date": "2008-01-30T00:00:00", "vocab": "market", "tf-idf scores": 0.16530118824234355}, {"Date": "2008-01-30T00:00:00", "vocab": "fourth", "tf-idf scores": 0.15241456267109924}, {"Date": "2008-01-30T00:00:00", "vocab": "december", "tf-idf scores": 0.14799613077220009}, {"Date": "2008-01-30T00:00:00", "vocab": "growth", "tf-idf scores": 0.13841832429712386}, {"Date": "2008-01-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13782856168799004}, {"Date": "2008-01-30T00:00:00", "vocab": "open", "tf-idf scores": 0.13443893620078876}, {"Date": "2008-01-21T00:00:00", "vocab": "shall", "tf-idf scores": 0.2507010485796758}, {"Date": "2008-01-21T00:00:00", "vocab": "foreign", "tf-idf scores": 0.22827211691605456}, {"Date": "2008-01-21T00:00:00", "vocab": "currency", "tf-idf scores": 0.1856371027441457}, {"Date": "2008-01-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.17319777108759277}, {"Date": "2008-01-21T00:00:00", "vocab": "market", "tf-idf scores": 0.16537784878776138}, {"Date": "2008-01-21T00:00:00", "vocab": "fourth", "tf-idf scores": 0.15248445679369652}, {"Date": "2008-01-21T00:00:00", "vocab": "december", "tf-idf scores": 0.1480402931224842}, {"Date": "2008-01-21T00:00:00", "vocab": "growth", "tf-idf scores": 0.13844402848533874}, {"Date": "2008-01-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13775682559035482}, {"Date": "2008-01-21T00:00:00", "vocab": "open", "tf-idf scores": 0.13444151729770587}, {"Date": "2008-01-09T00:00:00", "vocab": "shall", "tf-idf scores": 0.2506761892348525}, {"Date": "2008-01-09T00:00:00", "vocab": "foreign", "tf-idf scores": 0.22828669444475455}, {"Date": "2008-01-09T00:00:00", "vocab": "currency", "tf-idf scores": 0.18570568122568912}, {"Date": "2008-01-09T00:00:00", "vocab": "economic", "tf-idf scores": 0.1731737501814502}, {"Date": "2008-01-09T00:00:00", "vocab": "market", "tf-idf scores": 0.16538149190380913}, {"Date": "2008-01-09T00:00:00", "vocab": "fourth", "tf-idf scores": 0.15245531108933202}, {"Date": "2008-01-09T00:00:00", "vocab": "december", "tf-idf scores": 0.14799188263121718}, {"Date": "2008-01-09T00:00:00", "vocab": "growth", "tf-idf scores": 0.1384297759064637}, {"Date": "2008-01-09T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13777860038862727}, {"Date": "2008-01-09T00:00:00", "vocab": "open", "tf-idf scores": 0.13448821463164246}, {"Date": "2007-12-11T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22137912038793417}, {"Date": "2007-12-11T00:00:00", "vocab": "growth", "tf-idf scores": 0.2084829904769906}, {"Date": "2007-12-11T00:00:00", "vocab": "financial", "tf-idf scores": 0.18844516113466606}, {"Date": "2007-12-11T00:00:00", "vocab": "october", "tf-idf scores": 0.1701074347334218}, {"Date": "2007-12-11T00:00:00", "vocab": "prices", "tf-idf scores": 0.15295932762615466}, {"Date": "2007-12-11T00:00:00", "vocab": "real", "tf-idf scores": 0.14776840061257981}, {"Date": "2007-12-11T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13144754758197938}, {"Date": "2007-12-11T00:00:00", "vocab": "credit", "tf-idf scores": 0.1278487653123483}, {"Date": "2007-12-11T00:00:00", "vocab": "economic", "tf-idf scores": 0.11766181768223323}, {"Date": "2007-12-11T00:00:00", "vocab": "core", "tf-idf scores": 0.11174800189337536}, {"Date": "2007-12-06T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2214258627410836}, {"Date": "2007-12-06T00:00:00", "vocab": "growth", "tf-idf scores": 0.20845372038804735}, {"Date": "2007-12-06T00:00:00", "vocab": "financial", "tf-idf scores": 0.1884669290586609}, {"Date": "2007-12-06T00:00:00", "vocab": "october", "tf-idf scores": 0.17010320197721235}, {"Date": "2007-12-06T00:00:00", "vocab": "prices", "tf-idf scores": 0.1528626204589052}, {"Date": "2007-12-06T00:00:00", "vocab": "real", "tf-idf scores": 0.14774784135645347}, {"Date": "2007-12-06T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1314460092085784}, {"Date": "2007-12-06T00:00:00", "vocab": "credit", "tf-idf scores": 0.12789841424878168}, {"Date": "2007-12-06T00:00:00", "vocab": "economic", "tf-idf scores": 0.11762025177014442}, {"Date": "2007-12-06T00:00:00", "vocab": "core", "tf-idf scores": 0.11179404070148821}, {"Date": "2007-10-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.20679778153987938}, {"Date": "2007-10-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2067878218680637}, {"Date": "2007-10-31T00:00:00", "vocab": "markets", "tf-idf scores": 0.19349415593137395}, {"Date": "2007-10-31T00:00:00", "vocab": "august", "tf-idf scores": 0.17598355741070887}, {"Date": "2007-10-31T00:00:00", "vocab": "september", "tf-idf scores": 0.16604307847359503}, {"Date": "2007-10-31T00:00:00", "vocab": "growth", "tf-idf scores": 0.16086612162340488}, {"Date": "2007-10-31T00:00:00", "vocab": "prices", "tf-idf scores": 0.14071042315207585}, {"Date": "2007-10-31T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1334658010694823}, {"Date": "2007-10-31T00:00:00", "vocab": "financial", "tf-idf scores": 0.1279058975514177}, {"Date": "2007-10-31T00:00:00", "vocab": "recent", "tf-idf scores": 0.1200799732119962}, {"Date": "2007-09-18T00:00:00", "vocab": "financial", "tf-idf scores": 0.22430834896999302}, {"Date": "2007-09-18T00:00:00", "vocab": "credit", "tf-idf scores": 0.19260053393474236}, {"Date": "2007-09-18T00:00:00", "vocab": "market", "tf-idf scores": 0.18762933503304027}, {"Date": "2007-09-18T00:00:00", "vocab": "july", "tf-idf scores": 0.18536327711901687}, {"Date": "2007-09-18T00:00:00", "vocab": "recent", "tf-idf scores": 0.17372195733911197}, {"Date": "2007-09-18T00:00:00", "vocab": "august", "tf-idf scores": 0.17023354641859365}, {"Date": "2007-09-18T00:00:00", "vocab": "growth", "tf-idf scores": 0.13964289700532687}, {"Date": "2007-09-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.13896391926914767}, {"Date": "2007-09-18T00:00:00", "vocab": "markets", "tf-idf scores": 0.13901311901089594}, {"Date": "2007-09-18T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13203019860780457}, {"Date": "2007-08-16T00:00:00", "vocab": "financial", "tf-idf scores": 0.2242948845719556}, {"Date": "2007-08-16T00:00:00", "vocab": "credit", "tf-idf scores": 0.19257197720468494}, {"Date": "2007-08-16T00:00:00", "vocab": "market", "tf-idf scores": 0.18764831975426768}, {"Date": "2007-08-16T00:00:00", "vocab": "july", "tf-idf scores": 0.18538907460108642}, {"Date": "2007-08-16T00:00:00", "vocab": "recent", "tf-idf scores": 0.17369072125495855}, {"Date": "2007-08-16T00:00:00", "vocab": "august", "tf-idf scores": 0.1702639129562059}, {"Date": "2007-08-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.13956677733094752}, {"Date": "2007-08-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.13895544520228234}, {"Date": "2007-08-16T00:00:00", "vocab": "markets", "tf-idf scores": 0.13904204864373196}, {"Date": "2007-08-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13202293458434033}, {"Date": "2007-08-10T00:00:00", "vocab": "financial", "tf-idf scores": 0.2243428657183332}, {"Date": "2007-08-10T00:00:00", "vocab": "credit", "tf-idf scores": 0.19257027641685726}, {"Date": "2007-08-10T00:00:00", "vocab": "market", "tf-idf scores": 0.18765446969285404}, {"Date": "2007-08-10T00:00:00", "vocab": "july", "tf-idf scores": 0.18537205734440096}, {"Date": "2007-08-10T00:00:00", "vocab": "recent", "tf-idf scores": 0.173700823275296}, {"Date": "2007-08-10T00:00:00", "vocab": "august", "tf-idf scores": 0.1702407037790998}, {"Date": "2007-08-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.13962201768565363}, {"Date": "2007-08-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.1389533485885955}, {"Date": "2007-08-10T00:00:00", "vocab": "markets", "tf-idf scores": 0.13903458431199825}, {"Date": "2007-08-10T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13208686969172476}, {"Date": "2007-08-07T00:00:00", "vocab": "growth", "tf-idf scores": 0.2949642990330431}, {"Date": "2007-08-07T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2349076886455886}, {"Date": "2007-08-07T00:00:00", "vocab": "second", "tf-idf scores": 0.21857126253437648}, {"Date": "2007-08-07T00:00:00", "vocab": "economic", "tf-idf scores": 0.14690214373759924}, {"Date": "2007-08-07T00:00:00", "vocab": "subprime", "tf-idf scores": 0.14066722292534312}, {"Date": "2007-08-07T00:00:00", "vocab": "credit", "tf-idf scores": 0.12722859883815393}, {"Date": "2007-08-07T00:00:00", "vocab": "likely", "tf-idf scores": 0.1247965811008671}, {"Date": "2007-08-07T00:00:00", "vocab": "moderate", "tf-idf scores": 0.12277556639443868}, {"Date": "2007-08-07T00:00:00", "vocab": "policy", "tf-idf scores": 0.11746129829943702}, {"Date": "2007-08-07T00:00:00", "vocab": "prices", "tf-idf scores": 0.1032171855517935}, {"Date": "2007-06-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.30187403254212036}, {"Date": "2007-06-28T00:00:00", "vocab": "recent", "tf-idf scores": 0.21762895185457073}, {"Date": "2007-06-28T00:00:00", "vocab": "growth", "tf-idf scores": 0.21149357362194537}, {"Date": "2007-06-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.16144802495926594}, {"Date": "2007-06-28T00:00:00", "vocab": "core", "tf-idf scores": 0.15871383864492586}, {"Date": "2007-06-28T00:00:00", "vocab": "pace", "tf-idf scores": 0.1544234547014446}, {"Date": "2007-06-28T00:00:00", "vocab": "april", "tf-idf scores": 0.1408014384264187}, {"Date": "2007-06-28T00:00:00", "vocab": "moderate", "tf-idf scores": 0.13305551693500328}, {"Date": "2007-06-28T00:00:00", "vocab": "spending", "tf-idf scores": 0.11937627248203873}, {"Date": "2007-06-28T00:00:00", "vocab": "fail", "tf-idf scores": 0.11916602396831415}, {"Date": "2007-05-09T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23562057082713905}, {"Date": "2007-05-09T00:00:00", "vocab": "economic", "tf-idf scores": 0.18850749012728174}, {"Date": "2007-05-09T00:00:00", "vocab": "growth", "tf-idf scores": 0.165643776472987}, {"Date": "2007-05-09T00:00:00", "vocab": "pace", "tf-idf scores": 0.14924076289555954}, {"Date": "2007-05-09T00:00:00", "vocab": "predominant", "tf-idf scores": 0.14398260669557505}, {"Date": "2007-05-09T00:00:00", "vocab": "appeared", "tf-idf scores": 0.134107536055601}, {"Date": "2007-05-09T00:00:00", "vocab": "fail", "tf-idf scores": 0.1333067857353762}, {"Date": "2007-05-09T00:00:00", "vocab": "march", "tf-idf scores": 0.1322995390652166}, {"Date": "2007-05-09T00:00:00", "vocab": "recent", "tf-idf scores": 0.12565689412729553}, {"Date": "2007-05-09T00:00:00", "vocab": "moderate", "tf-idf scores": 0.12264169953212696}, {"Date": "2007-03-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2807213691762106}, {"Date": "2007-03-21T00:00:00", "vocab": "recent", "tf-idf scores": 0.2147465865318181}, {"Date": "2007-03-21T00:00:00", "vocab": "growth", "tf-idf scores": 0.18250239385727882}, {"Date": "2007-03-21T00:00:00", "vocab": "subprime", "tf-idf scores": 0.15826668274474745}, {"Date": "2007-03-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.14869424439767417}, {"Date": "2007-03-21T00:00:00", "vocab": "investment", "tf-idf scores": 0.13274356159740064}, {"Date": "2007-03-21T00:00:00", "vocab": "spending", "tf-idf scores": 0.13214751878583972}, {"Date": "2007-03-21T00:00:00", "vocab": "likely", "tf-idf scores": 0.12391253811163958}, {"Date": "2007-03-21T00:00:00", "vocab": "pace", "tf-idf scores": 0.12394310977723341}, {"Date": "2007-03-21T00:00:00", "vocab": "governors", "tf-idf scores": 0.12069442608134069}, {"Date": "2007-01-31T00:00:00", "vocab": "shall", "tf-idf scores": 0.2865921543409967}, {"Date": "2007-01-31T00:00:00", "vocab": "foreign", "tf-idf scores": 0.23399148775376002}, {"Date": "2007-01-31T00:00:00", "vocab": "currency", "tf-idf scores": 0.19652306281388168}, {"Date": "2007-01-31T00:00:00", "vocab": "market", "tf-idf scores": 0.1619494183642586}, {"Date": "2007-01-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1529808246039529}, {"Date": "2007-01-31T00:00:00", "vocab": "open", "tf-idf scores": 0.14916025083382564}, {"Date": "2007-01-31T00:00:00", "vocab": "growth", "tf-idf scores": 0.14012846728972572}, {"Date": "2007-01-31T00:00:00", "vocab": "bank", "tf-idf scores": 0.1355582761307581}, {"Date": "2007-01-31T00:00:00", "vocab": "chairman", "tf-idf scores": 0.1274268568466175}, {"Date": "2007-01-31T00:00:00", "vocab": "accounts", "tf-idf scores": 0.1265866706607257}, {"Date": "2006-12-12T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22705760419146148}, {"Date": "2006-12-12T00:00:00", "vocab": "growth", "tf-idf scores": 0.19431291386631958}, {"Date": "2006-12-12T00:00:00", "vocab": "economic", "tf-idf scores": 0.18497650271250837}, {"Date": "2006-12-12T00:00:00", "vocab": "october", "tf-idf scores": 0.17725492364616552}, {"Date": "2006-12-12T00:00:00", "vocab": "spending", "tf-idf scores": 0.12620813999845673}, {"Date": "2006-12-12T00:00:00", "vocab": "activity", "tf-idf scores": 0.11770853211785845}, {"Date": "2006-12-12T00:00:00", "vocab": "remained", "tf-idf scores": 0.11775162477737179}, {"Date": "2006-12-12T00:00:00", "vocab": "core", "tf-idf scores": 0.11772241427312112}, {"Date": "2006-12-12T00:00:00", "vocab": "moderate", "tf-idf scores": 0.11256756943080573}, {"Date": "2006-12-12T00:00:00", "vocab": "recent", "tf-idf scores": 0.10934023342616006}, {"Date": "2006-10-25T00:00:00", "vocab": "inflation", "tf-idf scores": 0.26816365955813104}, {"Date": "2006-10-25T00:00:00", "vocab": "growth", "tf-idf scores": 0.17960541245057385}, {"Date": "2006-10-25T00:00:00", "vocab": "remained", "tf-idf scores": 0.15444383561567684}, {"Date": "2006-10-25T00:00:00", "vocab": "september", "tf-idf scores": 0.15163965996185833}, {"Date": "2006-10-25T00:00:00", "vocab": "prices", "tf-idf scores": 0.14698109743030272}, {"Date": "2006-10-25T00:00:00", "vocab": "economic", "tf-idf scores": 0.14632626515447314}, {"Date": "2006-10-25T00:00:00", "vocab": "continued", "tf-idf scores": 0.13819831559151624}, {"Date": "2006-10-25T00:00:00", "vocab": "core", "tf-idf scores": 0.13126927139786976}, {"Date": "2006-10-25T00:00:00", "vocab": "likely", "tf-idf scores": 0.12194345877349842}, {"Date": "2006-10-25T00:00:00", "vocab": "recent", "tf-idf scores": 0.12192737470758772}, {"Date": "2006-09-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23602036928435402}, {"Date": "2006-09-20T00:00:00", "vocab": "growth", "tf-idf scores": 0.18966083204497808}, {"Date": "2006-09-20T00:00:00", "vocab": "recent", "tf-idf scores": 0.16053604770354954}, {"Date": "2006-09-20T00:00:00", "vocab": "prices", "tf-idf scores": 0.1517311142625542}, {"Date": "2006-09-20T00:00:00", "vocab": "july", "tf-idf scores": 0.14394284348013037}, {"Date": "2006-09-20T00:00:00", "vocab": "august", "tf-idf scores": 0.14232750993465368}, {"Date": "2006-09-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.1416499384458055}, {"Date": "2006-09-20T00:00:00", "vocab": "energy", "tf-idf scores": 0.1333394728396284}, {"Date": "2006-09-20T00:00:00", "vocab": "increases", "tf-idf scores": 0.12548855581215468}, {"Date": "2006-09-20T00:00:00", "vocab": "pace", "tf-idf scores": 0.12279740682817997}, {"Date": "2006-08-08T00:00:00", "vocab": "inflation", "tf-idf scores": 0.26564073616214895}, {"Date": "2006-08-08T00:00:00", "vocab": "growth", "tf-idf scores": 0.21164299867844882}, {"Date": "2006-08-08T00:00:00", "vocab": "second", "tf-idf scores": 0.18875487836550403}, {"Date": "2006-08-08T00:00:00", "vocab": "june", "tf-idf scores": 0.1851944437754495}, {"Date": "2006-08-08T00:00:00", "vocab": "prices", "tf-idf scores": 0.1840360171689436}, {"Date": "2006-08-08T00:00:00", "vocab": "continued", "tf-idf scores": 0.16488724042770575}, {"Date": "2006-08-08T00:00:00", "vocab": "energy", "tf-idf scores": 0.1385862683378025}, {"Date": "2006-08-08T00:00:00", "vocab": "quarter", "tf-idf scores": 0.11966142086098751}, {"Date": "2006-08-08T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11909508069553354}, {"Date": "2006-08-08T00:00:00", "vocab": "economic", "tf-idf scores": 0.11908289919021355}, {"Date": "2006-06-29T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2909312164392814}, {"Date": "2006-06-29T00:00:00", "vocab": "growth", "tf-idf scores": 0.24213489035181857}, {"Date": "2006-06-29T00:00:00", "vocab": "prices", "tf-idf scores": 0.16693986605705846}, {"Date": "2006-06-29T00:00:00", "vocab": "quarter", "tf-idf scores": 0.15029148099540818}, {"Date": "2006-06-29T00:00:00", "vocab": "core", "tf-idf scores": 0.12536242919468374}, {"Date": "2006-06-29T00:00:00", "vocab": "tendency", "tf-idf scores": 0.12228576997701145}, {"Date": "2006-06-29T00:00:00", "vocab": "april", "tf-idf scores": 0.12128896901025968}, {"Date": "2006-06-29T00:00:00", "vocab": "energy", "tf-idf scores": 0.10902266058860052}, {"Date": "2006-06-29T00:00:00", "vocab": "second", "tf-idf scores": 0.10469645365162089}, {"Date": "2006-06-29T00:00:00", "vocab": "consumer", "tf-idf scores": 0.09977769039056501}, {"Date": "2006-05-10T00:00:00", "vocab": "march", "tf-idf scores": 0.27160881009434806}, {"Date": "2006-05-10T00:00:00", "vocab": "inflation", "tf-idf scores": 0.24578323551471992}, {"Date": "2006-05-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.23905924100608417}, {"Date": "2006-05-10T00:00:00", "vocab": "prices", "tf-idf scores": 0.17742367287853467}, {"Date": "2006-05-10T00:00:00", "vocab": "quarter", "tf-idf scores": 0.1388494184991038}, {"Date": "2006-05-10T00:00:00", "vocab": "pace", "tf-idf scores": 0.1305886723160659}, {"Date": "2006-05-10T00:00:00", "vocab": "energy", "tf-idf scores": 0.12401114752165686}, {"Date": "2006-05-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.1229113411935784}, {"Date": "2006-05-10T00:00:00", "vocab": "spending", "tf-idf scores": 0.1152428760746687}, {"Date": "2006-05-10T00:00:00", "vocab": "firming", "tf-idf scores": 0.1084741494634456}, {"Date": "2006-03-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1901712674018996}, {"Date": "2006-03-28T00:00:00", "vocab": "prices", "tf-idf scores": 0.1826881121492284}, {"Date": "2006-03-28T00:00:00", "vocab": "february", "tf-idf scores": 0.17235782210276562}, {"Date": "2006-03-28T00:00:00", "vocab": "growth", "tf-idf scores": 0.16607036964232408}, {"Date": "2006-03-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.1570683735845585}, {"Date": "2006-03-28T00:00:00", "vocab": "january", "tf-idf scores": 0.14427873498600266}, {"Date": "2006-03-28T00:00:00", "vocab": "increases", "tf-idf scores": 0.13518372315123534}, {"Date": "2006-03-28T00:00:00", "vocab": "appeared", "tf-idf scores": 0.13288910489043224}, {"Date": "2006-03-28T00:00:00", "vocab": "market", "tf-idf scores": 0.13225279957930738}, {"Date": "2006-03-28T00:00:00", "vocab": "fourth", "tf-idf scores": 0.1185951952155257}, {"Date": "2006-01-31T00:00:00", "vocab": "shall", "tf-idf scores": 0.29827330417531567}, {"Date": "2006-01-31T00:00:00", "vocab": "foreign", "tf-idf scores": 0.24725324295163867}, {"Date": "2006-01-31T00:00:00", "vocab": "currency", "tf-idf scores": 0.21201307680123926}, {"Date": "2006-01-31T00:00:00", "vocab": "market", "tf-idf scores": 0.1618753319907337}, {"Date": "2006-01-31T00:00:00", "vocab": "chairman", "tf-idf scores": 0.15925997090117}, {"Date": "2006-01-31T00:00:00", "vocab": "open", "tf-idf scores": 0.15350950493439053}, {"Date": "2006-01-31T00:00:00", "vocab": "bank", "tf-idf scores": 0.13548753456574145}, {"Date": "2006-01-31T00:00:00", "vocab": "new", "tf-idf scores": 0.11696165573029929}, {"Date": "2006-01-31T00:00:00", "vocab": "york", "tf-idf scores": 0.11592912896929691}, {"Date": "2006-01-31T00:00:00", "vocab": "inflation", "tf-idf scores": 0.10796310816701213}, {"Date": "2005-12-13T00:00:00", "vocab": "inflation", "tf-idf scores": 0.22755430346822664}, {"Date": "2005-12-13T00:00:00", "vocab": "policy", "tf-idf scores": 0.20024811872985937}, {"Date": "2005-12-13T00:00:00", "vocab": "energy", "tf-idf scores": 0.17453887447964445}, {"Date": "2005-12-13T00:00:00", "vocab": "growth", "tf-idf scores": 0.17370098592041275}, {"Date": "2005-12-13T00:00:00", "vocab": "prices", "tf-idf scores": 0.17377739534645}, {"Date": "2005-12-13T00:00:00", "vocab": "november", "tf-idf scores": 0.12930701386404694}, {"Date": "2005-12-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.12744617132308683}, {"Date": "2005-12-13T00:00:00", "vocab": "firming", "tf-idf scores": 0.1124932504632263}, {"Date": "2005-12-13T00:00:00", "vocab": "market", "tf-idf scores": 0.1092749985603219}, {"Date": "2005-12-13T00:00:00", "vocab": "meeting", "tf-idf scores": 0.10923907679976583}, {"Date": "2005-11-01T00:00:00", "vocab": "hurricane", "tf-idf scores": 0.25542136083416817}, {"Date": "2005-11-01T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23964855246289665}, {"Date": "2005-11-01T00:00:00", "vocab": "hurricanes", "tf-idf scores": 0.1851840347425791}, {"Date": "2005-11-01T00:00:00", "vocab": "rebuilding", "tf-idf scores": 0.17943416404871448}, {"Date": "2005-11-01T00:00:00", "vocab": "remained", "tf-idf scores": 0.1711859867316097}, {"Date": "2005-11-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.1455682679867071}, {"Date": "2005-11-01T00:00:00", "vocab": "growth", "tf-idf scores": 0.1376029975302835}, {"Date": "2005-11-01T00:00:00", "vocab": "policy", "tf-idf scores": 0.13697963232945104}, {"Date": "2005-11-01T00:00:00", "vocab": "energy", "tf-idf scores": 0.12953996166566806}, {"Date": "2005-11-01T00:00:00", "vocab": "price", "tf-idf scores": 0.12843162242197712}, {"Date": "2005-09-20T00:00:00", "vocab": "hurricane", "tf-idf scores": 0.35362769543007816}, {"Date": "2005-09-20T00:00:00", "vocab": "gulf", "tf-idf scores": 0.20024159535635086}, {"Date": "2005-09-20T00:00:00", "vocab": "inflation", "tf-idf scores": 0.19830189351806182}, {"Date": "2005-09-20T00:00:00", "vocab": "katrina", "tf-idf scores": 0.18413480702323212}, {"Date": "2005-09-20T00:00:00", "vocab": "probably", "tf-idf scores": 0.14951458838460846}, {"Date": "2005-09-20T00:00:00", "vocab": "coast", "tf-idf scores": 0.14625718941288707}, {"Date": "2005-09-20T00:00:00", "vocab": "energy", "tf-idf scores": 0.12172718702054298}, {"Date": "2005-09-20T00:00:00", "vocab": "growth", "tf-idf scores": 0.12120879732838172}, {"Date": "2005-09-20T00:00:00", "vocab": "august", "tf-idf scores": 0.11369748159337059}, {"Date": "2005-09-20T00:00:00", "vocab": "prices", "tf-idf scores": 0.11256115909051546}, {"Date": "2005-08-09T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23818323197774904}, {"Date": "2005-08-09T00:00:00", "vocab": "growth", "tf-idf scores": 0.17687818813373446}, {"Date": "2005-08-09T00:00:00", "vocab": "pace", "tf-idf scores": 0.17611225708803346}, {"Date": "2005-08-09T00:00:00", "vocab": "second", "tf-idf scores": 0.16602517533877956}, {"Date": "2005-08-09T00:00:00", "vocab": "remained", "tf-idf scores": 0.15535816791958354}, {"Date": "2005-08-09T00:00:00", "vocab": "policy", "tf-idf scores": 0.14497618793261866}, {"Date": "2005-08-09T00:00:00", "vocab": "recent", "tf-idf scores": 0.14496595690579298}, {"Date": "2005-08-09T00:00:00", "vocab": "june", "tf-idf scores": 0.14496398742137553}, {"Date": "2005-08-09T00:00:00", "vocab": "core", "tf-idf scores": 0.13387684402319944}, {"Date": "2005-08-09T00:00:00", "vocab": "continued", "tf-idf scores": 0.11390106380081047}, {"Date": "2005-06-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.24719106944615737}, {"Date": "2005-06-30T00:00:00", "vocab": "growth", "tf-idf scores": 0.15512744200281156}, {"Date": "2005-06-30T00:00:00", "vocab": "labor", "tf-idf scores": 0.1545106644406435}, {"Date": "2005-06-30T00:00:00", "vocab": "remained", "tf-idf scores": 0.14671975703051363}, {"Date": "2005-06-30T00:00:00", "vocab": "prices", "tf-idf scores": 0.13962767506024026}, {"Date": "2005-06-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.13899706573707826}, {"Date": "2005-06-30T00:00:00", "vocab": "market", "tf-idf scores": 0.13136229119506343}, {"Date": "2005-06-30T00:00:00", "vocab": "price", "tf-idf scores": 0.13129413033043763}, {"Date": "2005-06-30T00:00:00", "vocab": "recent", "tf-idf scores": 0.13133657322645356}, {"Date": "2005-06-30T00:00:00", "vocab": "governors", "tf-idf scores": 0.12421122907476477}, {"Date": "2005-05-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.23308596548645685}, {"Date": "2005-05-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.20979117884859907}, {"Date": "2005-05-03T00:00:00", "vocab": "growth", "tf-idf scores": 0.1950602285910372}, {"Date": "2005-05-03T00:00:00", "vocab": "prices", "tf-idf scores": 0.187265743400631}, {"Date": "2005-05-03T00:00:00", "vocab": "energy", "tf-idf scores": 0.18024174161421389}, {"Date": "2005-05-03T00:00:00", "vocab": "policy", "tf-idf scores": 0.14768192092512997}, {"Date": "2005-05-03T00:00:00", "vocab": "march", "tf-idf scores": 0.1439480514293808}, {"Date": "2005-05-03T00:00:00", "vocab": "recent", "tf-idf scores": 0.13212428875133653}, {"Date": "2005-05-03T00:00:00", "vocab": "price", "tf-idf scores": 0.1243824544169968}, {"Date": "2005-05-03T00:00:00", "vocab": "remained", "tf-idf scores": 0.11656619105907827}, {"Date": "2005-03-22T00:00:00", "vocab": "inflation", "tf-idf scores": 0.2683811636904564}, {"Date": "2005-03-22T00:00:00", "vocab": "policy", "tf-idf scores": 0.1708200113116812}, {"Date": "2005-03-22T00:00:00", "vocab": "prices", "tf-idf scores": 0.16333776704325192}, {"Date": "2005-03-22T00:00:00", "vocab": "growth", "tf-idf scores": 0.1551895882233289}, {"Date": "2005-03-22T00:00:00", "vocab": "labor", "tf-idf scores": 0.15452602083067263}, {"Date": "2005-03-22T00:00:00", "vocab": "consumer", "tf-idf scores": 0.14635627707016902}, {"Date": "2005-03-22T00:00:00", "vocab": "business", "tf-idf scores": 0.12196344882257205}, {"Date": "2005-03-22T00:00:00", "vocab": "economic", "tf-idf scores": 0.12198225534888064}, {"Date": "2005-03-22T00:00:00", "vocab": "price", "tf-idf scores": 0.11389533138014667}, {"Date": "2005-03-22T00:00:00", "vocab": "likely", "tf-idf scores": 0.10573855640995233}, {"Date": "2005-02-02T00:00:00", "vocab": "shall", "tf-idf scores": 0.29552459755731325}, {"Date": "2005-02-02T00:00:00", "vocab": "foreign", "tf-idf scores": 0.25522671501119837}, {"Date": "2005-02-02T00:00:00", "vocab": "currency", "tf-idf scores": 0.21072158383990613}, {"Date": "2005-02-02T00:00:00", "vocab": "market", "tf-idf scores": 0.17863991300424195}, {"Date": "2005-02-02T00:00:00", "vocab": "open", "tf-idf scores": 0.15153406458262544}, {"Date": "2005-02-02T00:00:00", "vocab": "chairman", "tf-idf scores": 0.14789266376221555}, {"Date": "2005-02-02T00:00:00", "vocab": "inflation", "tf-idf scores": 0.12066391194760663}, {"Date": "2005-02-02T00:00:00", "vocab": "operations", "tf-idf scores": 0.1102715393051942}, {"Date": "2005-02-02T00:00:00", "vocab": "paragraph", "tf-idf scores": 0.10724707136450878}, {"Date": "2005-02-02T00:00:00", "vocab": "fourth", "tf-idf scores": 0.10646382359966593}, {"Date": "2004-12-14T00:00:00", "vocab": "november", "tf-idf scores": 0.17818554777820558}, {"Date": "2004-12-14T00:00:00", "vocab": "recent", "tf-idf scores": 0.17571878539303118}, {"Date": "2004-12-14T00:00:00", "vocab": "growth", "tf-idf scores": 0.16806343236098326}, {"Date": "2004-12-14T00:00:00", "vocab": "inflation", "tf-idf scores": 0.16735156694949585}, {"Date": "2004-12-14T00:00:00", "vocab": "pace", "tf-idf scores": 0.14221898640646338}, {"Date": "2004-12-14T00:00:00", "vocab": "minutes", "tf-idf scores": 0.14110656967042454}, {"Date": "2004-12-14T00:00:00", "vocab": "economic", "tf-idf scores": 0.13392318530081956}, {"Date": "2004-12-14T00:00:00", "vocab": "prices", "tf-idf scores": 0.12608718835712004}, {"Date": "2004-12-14T00:00:00", "vocab": "policy", "tf-idf scores": 0.12547700799352318}, {"Date": "2004-12-14T00:00:00", "vocab": "productivity", "tf-idf scores": 0.12230707951526162}, {"Date": "2004-11-10T00:00:00", "vocab": "pace", "tf-idf scores": 0.1763098762629547}, {"Date": "2004-11-10T00:00:00", "vocab": "members", "tf-idf scores": 0.16797197493498878}, {"Date": "2004-11-10T00:00:00", "vocab": "likely", "tf-idf scores": 0.1567193041860498}, {"Date": "2004-11-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.1371323449986431}, {"Date": "2004-11-10T00:00:00", "vocab": "september", "tf-idf scores": 0.13714674252713205}, {"Date": "2004-11-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.12794312911551228}, {"Date": "2004-11-10T00:00:00", "vocab": "recent", "tf-idf scores": 0.12731580679563703}, {"Date": "2004-11-10T00:00:00", "vocab": "spending", "tf-idf scores": 0.12734059865097994}, {"Date": "2004-11-10T00:00:00", "vocab": "business", "tf-idf scores": 0.11752707624877334}, {"Date": "2004-11-10T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11758752851248186}, {"Date": "2004-09-21T00:00:00", "vocab": "july", "tf-idf scores": 0.24137365101963848}, {"Date": "2004-09-21T00:00:00", "vocab": "growth", "tf-idf scores": 0.1696798726406879}, {"Date": "2004-09-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.16890324675674656}, {"Date": "2004-09-21T00:00:00", "vocab": "prices", "tf-idf scores": 0.13578204354150702}, {"Date": "2004-09-21T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13514148135864487}, {"Date": "2004-09-21T00:00:00", "vocab": "policy", "tf-idf scores": 0.1351540860895502}, {"Date": "2004-09-21T00:00:00", "vocab": "policymakers", "tf-idf scores": 0.1338598857397438}, {"Date": "2004-09-21T00:00:00", "vocab": "labor", "tf-idf scores": 0.12666369213079468}, {"Date": "2004-09-21T00:00:00", "vocab": "pace", "tf-idf scores": 0.12670549564000688}, {"Date": "2004-09-21T00:00:00", "vocab": "spending", "tf-idf scores": 0.11827132092817545}, {"Date": "2004-08-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.2758068034413206}, {"Date": "2004-08-10T00:00:00", "vocab": "members", "tf-idf scores": 0.17629254194549526}, {"Date": "2004-08-10T00:00:00", "vocab": "june", "tf-idf scores": 0.16829955300526905}, {"Date": "2004-08-10T00:00:00", "vocab": "spending", "tf-idf scores": 0.15812087781487816}, {"Date": "2004-08-10T00:00:00", "vocab": "energy", "tf-idf scores": 0.15108071903598008}, {"Date": "2004-08-10T00:00:00", "vocab": "consumer", "tf-idf scores": 0.14979079712954563}, {"Date": "2004-08-10T00:00:00", "vocab": "pace", "tf-idf scores": 0.14982932373239993}, {"Date": "2004-08-10T00:00:00", "vocab": "recent", "tf-idf scores": 0.14977624562604383}, {"Date": "2004-08-10T00:00:00", "vocab": "removed", "tf-idf scores": 0.1356931309834409}, {"Date": "2004-08-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.13320368404670557}, {"Date": "2004-06-30T00:00:00", "vocab": "policy", "tf-idf scores": 0.17019398655944346}, {"Date": "2004-06-30T00:00:00", "vocab": "members", "tf-idf scores": 0.16356710901625052}, {"Date": "2004-06-30T00:00:00", "vocab": "inflation", "tf-idf scores": 0.16208791255795046}, {"Date": "2004-06-30T00:00:00", "vocab": "recent", "tf-idf scores": 0.15396445736270753}, {"Date": "2004-06-30T00:00:00", "vocab": "growth", "tf-idf scores": 0.14651043413398449}, {"Date": "2004-06-30T00:00:00", "vocab": "consumer", "tf-idf scores": 0.14588083840818208}, {"Date": "2004-06-30T00:00:00", "vocab": "price", "tf-idf scores": 0.12966196692475096}, {"Date": "2004-06-30T00:00:00", "vocab": "increases", "tf-idf scores": 0.12430125836277599}, {"Date": "2004-06-30T00:00:00", "vocab": "spending", "tf-idf scores": 0.12155212964358153}, {"Date": "2004-06-30T00:00:00", "vocab": "april", "tf-idf scores": 0.11818592725867834}, {"Date": "2004-05-04T00:00:00", "vocab": "march", "tf-idf scores": 0.1851165947438652}, {"Date": "2004-05-04T00:00:00", "vocab": "growth", "tf-idf scores": 0.17830754105363025}, {"Date": "2004-05-04T00:00:00", "vocab": "members", "tf-idf scores": 0.1705438227342517}, {"Date": "2004-05-04T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14371567173960306}, {"Date": "2004-05-04T00:00:00", "vocab": "price", "tf-idf scores": 0.14377704295974442}, {"Date": "2004-05-04T00:00:00", "vocab": "business", "tf-idf scores": 0.13526466205252685}, {"Date": "2004-05-04T00:00:00", "vocab": "quarter", "tf-idf scores": 0.12740795601339905}, {"Date": "2004-05-04T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12686377745603458}, {"Date": "2004-05-04T00:00:00", "vocab": "increases", "tf-idf scores": 0.12095018206781302}, {"Date": "2004-05-04T00:00:00", "vocab": "market", "tf-idf scores": 0.11841530978323582}, {"Date": "2004-03-16T00:00:00", "vocab": "members", "tf-idf scores": 0.23495324046007354}, {"Date": "2004-03-16T00:00:00", "vocab": "january", "tf-idf scores": 0.2052838092495677}, {"Date": "2004-03-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.19064707415047483}, {"Date": "2004-03-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18116256906018094}, {"Date": "2004-03-16T00:00:00", "vocab": "policy", "tf-idf scores": 0.16389763889779688}, {"Date": "2004-03-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.15527518046020866}, {"Date": "2004-03-16T00:00:00", "vocab": "spending", "tf-idf scores": 0.12083333614251013}, {"Date": "2004-03-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.1121956706358979}, {"Date": "2004-03-16T00:00:00", "vocab": "remained", "tf-idf scores": 0.1121582414651024}, {"Date": "2004-03-16T00:00:00", "vocab": "consumer", "tf-idf scores": 0.10357729415597819}, {"Date": "2004-01-28T00:00:00", "vocab": "shall", "tf-idf scores": 0.2671272123265108}, {"Date": "2004-01-28T00:00:00", "vocab": "foreign", "tf-idf scores": 0.21287371735675684}, {"Date": "2004-01-28T00:00:00", "vocab": "currency", "tf-idf scores": 0.18935207684142455}, {"Date": "2004-01-28T00:00:00", "vocab": "members", "tf-idf scores": 0.1835387938526493}, {"Date": "2004-01-28T00:00:00", "vocab": "market", "tf-idf scores": 0.15488455791062583}, {"Date": "2004-01-28T00:00:00", "vocab": "york", "tf-idf scores": 0.1330636020121519}, {"Date": "2004-01-28T00:00:00", "vocab": "open", "tf-idf scores": 0.12833907389312793}, {"Date": "2004-01-28T00:00:00", "vocab": "fourth", "tf-idf scores": 0.127726413780625}, {"Date": "2004-01-28T00:00:00", "vocab": "business", "tf-idf scores": 0.12005908646050213}, {"Date": "2004-01-28T00:00:00", "vocab": "bank", "tf-idf scores": 0.11281431828806696}, {"Date": "2003-12-09T00:00:00", "vocab": "members", "tf-idf scores": 0.2535688558656287}, {"Date": "2003-12-09T00:00:00", "vocab": "october", "tf-idf scores": 0.1793053191326407}, {"Date": "2003-12-09T00:00:00", "vocab": "spending", "tf-idf scores": 0.15713279532081986}, {"Date": "2003-12-09T00:00:00", "vocab": "economic", "tf-idf scores": 0.1492559688637871}, {"Date": "2003-12-09T00:00:00", "vocab": "growth", "tf-idf scores": 0.14204372216386005}, {"Date": "2003-12-09T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14139993128796743}, {"Date": "2003-12-09T00:00:00", "vocab": "governors", "tf-idf scores": 0.1263170097128904}, {"Date": "2003-12-09T00:00:00", "vocab": "policy", "tf-idf scores": 0.11781883269272238}, {"Date": "2003-12-09T00:00:00", "vocab": "recent", "tf-idf scores": 0.11784692524060833}, {"Date": "2003-12-09T00:00:00", "vocab": "gains", "tf-idf scores": 0.10909475023591764}, {"Date": "2003-10-28T00:00:00", "vocab": "members", "tf-idf scores": 0.22447711922642877}, {"Date": "2003-10-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.20476370373753525}, {"Date": "2003-10-28T00:00:00", "vocab": "business", "tf-idf scores": 0.1868981247853719}, {"Date": "2003-10-28T00:00:00", "vocab": "august", "tf-idf scores": 0.1844650849700248}, {"Date": "2003-10-28T00:00:00", "vocab": "september", "tf-idf scores": 0.1800240038068772}, {"Date": "2003-10-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14243172462750456}, {"Date": "2003-10-28T00:00:00", "vocab": "recent", "tf-idf scores": 0.1246597051683199}, {"Date": "2003-10-28T00:00:00", "vocab": "growth", "tf-idf scores": 0.11620305091946143}, {"Date": "2003-10-28T00:00:00", "vocab": "consumer", "tf-idf scores": 0.115774096751631}, {"Date": "2003-10-28T00:00:00", "vocab": "indications", "tf-idf scores": 0.10786689174440868}, {"Date": "2003-09-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.21471477499775599}, {"Date": "2003-09-16T00:00:00", "vocab": "members", "tf-idf scores": 0.20827016991076092}, {"Date": "2003-09-16T00:00:00", "vocab": "business", "tf-idf scores": 0.20645416717425918}, {"Date": "2003-09-16T00:00:00", "vocab": "august", "tf-idf scores": 0.1400471732184142}, {"Date": "2003-09-16T00:00:00", "vocab": "july", "tf-idf scores": 0.12594722649362416}, {"Date": "2003-09-16T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12389007099599435}, {"Date": "2003-09-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.1161193159782786}, {"Date": "2003-09-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11567342588217769}, {"Date": "2003-09-16T00:00:00", "vocab": "tax", "tf-idf scores": 0.10914104700576156}, {"Date": "2003-09-16T00:00:00", "vocab": "final", "tf-idf scores": 0.10898484181500066}, {"Date": "2003-09-15T00:00:00", "vocab": "business", "tf-idf scores": 0.1940258481010123}, {"Date": "2003-09-15T00:00:00", "vocab": "members", "tf-idf scores": 0.18642108293430915}, {"Date": "2003-09-15T00:00:00", "vocab": "june", "tf-idf scores": 0.1724620041775212}, {"Date": "2003-09-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.16629726303019968}, {"Date": "2003-09-15T00:00:00", "vocab": "consumer", "tf-idf scores": 0.14791092479942233}, {"Date": "2003-09-15T00:00:00", "vocab": "disinflation", "tf-idf scores": 0.13322184130172032}, {"Date": "2003-09-15T00:00:00", "vocab": "policy", "tf-idf scores": 0.1293730790660527}, {"Date": "2003-09-15T00:00:00", "vocab": "growth", "tf-idf scores": 0.12064963534032298}, {"Date": "2003-09-15T00:00:00", "vocab": "remained", "tf-idf scores": 0.12014378941367693}, {"Date": "2003-09-15T00:00:00", "vocab": "recent", "tf-idf scores": 0.11094308581775689}, {"Date": "2003-08-12T00:00:00", "vocab": "business", "tf-idf scores": 0.19407960769383206}, {"Date": "2003-08-12T00:00:00", "vocab": "members", "tf-idf scores": 0.1864179447515781}, {"Date": "2003-08-12T00:00:00", "vocab": "june", "tf-idf scores": 0.17246057179797036}, {"Date": "2003-08-12T00:00:00", "vocab": "economic", "tf-idf scores": 0.16631022474790144}, {"Date": "2003-08-12T00:00:00", "vocab": "consumer", "tf-idf scores": 0.14786735608897786}, {"Date": "2003-08-12T00:00:00", "vocab": "disinflation", "tf-idf scores": 0.13322606639802195}, {"Date": "2003-08-12T00:00:00", "vocab": "policy", "tf-idf scores": 0.12942291386525337}, {"Date": "2003-08-12T00:00:00", "vocab": "growth", "tf-idf scores": 0.1206532390856668}, {"Date": "2003-08-12T00:00:00", "vocab": "remained", "tf-idf scores": 0.12012514115326177}, {"Date": "2003-08-12T00:00:00", "vocab": "recent", "tf-idf scores": 0.11091816892636286}, {"Date": "2003-06-25T00:00:00", "vocab": "members", "tf-idf scores": 0.2954692863151836}, {"Date": "2003-06-25T00:00:00", "vocab": "economic", "tf-idf scores": 0.21239704413966393}, {"Date": "2003-06-25T00:00:00", "vocab": "policy", "tf-idf scores": 0.14642171665989548}, {"Date": "2003-06-25T00:00:00", "vocab": "business", "tf-idf scores": 0.12449191745621115}, {"Date": "2003-06-25T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12449198498066619}, {"Date": "2003-06-25T00:00:00", "vocab": "april", "tf-idf scores": 0.1201615853394818}, {"Date": "2003-06-25T00:00:00", "vocab": "growth", "tf-idf scores": 0.11765681439939858}, {"Date": "2003-06-25T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11716210781782664}, {"Date": "2003-06-25T00:00:00", "vocab": "low", "tf-idf scores": 0.11325564003394618}, {"Date": "2003-06-25T00:00:00", "vocab": "continued", "tf-idf scores": 0.10982671155145815}, {"Date": "2003-05-06T00:00:00", "vocab": "economic", "tf-idf scores": 0.2983865349248085}, {"Date": "2003-05-06T00:00:00", "vocab": "members", "tf-idf scores": 0.18063116611944244}, {"Date": "2003-05-06T00:00:00", "vocab": "march", "tf-idf scores": 0.17229145397252188}, {"Date": "2003-05-06T00:00:00", "vocab": "disinflation", "tf-idf scores": 0.17205403752606996}, {"Date": "2003-05-06T00:00:00", "vocab": "growth", "tf-idf scores": 0.1456015584913561}, {"Date": "2003-05-06T00:00:00", "vocab": "business", "tf-idf scores": 0.144986924158806}, {"Date": "2003-05-06T00:00:00", "vocab": "persisting", "tf-idf scores": 0.12073076810603156}, {"Date": "2003-05-06T00:00:00", "vocab": "activity", "tf-idf scores": 0.11936400190655333}, {"Date": "2003-05-06T00:00:00", "vocab": "iraqi", "tf-idf scores": 0.11489847549872932}, {"Date": "2003-05-06T00:00:00", "vocab": "war", "tf-idf scores": 0.11263152090625814}, {"Date": "2003-04-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.19534685238768673}, {"Date": "2003-04-16T00:00:00", "vocab": "war", "tf-idf scores": 0.18433505906527736}, {"Date": "2003-04-16T00:00:00", "vocab": "iraq", "tf-idf scores": 0.15781864919643318}, {"Date": "2003-04-16T00:00:00", "vocab": "members", "tf-idf scores": 0.15013979478755937}, {"Date": "2003-04-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.14016391737083395}, {"Date": "2003-04-16T00:00:00", "vocab": "business", "tf-idf scores": 0.139567810957085}, {"Date": "2003-04-16T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1302138350810027}, {"Date": "2003-04-16T00:00:00", "vocab": "continued", "tf-idf scores": 0.12097136918543647}, {"Date": "2003-04-16T00:00:00", "vocab": "likely", "tf-idf scores": 0.12093586646999152}, {"Date": "2003-04-16T00:00:00", "vocab": "february", "tf-idf scores": 0.10575705949193351}, {"Date": "2003-04-08T00:00:00", "vocab": "economic", "tf-idf scores": 0.1952914851455422}, {"Date": "2003-04-08T00:00:00", "vocab": "war", "tf-idf scores": 0.18428554704413969}, {"Date": "2003-04-08T00:00:00", "vocab": "iraq", "tf-idf scores": 0.15782177006110015}, {"Date": "2003-04-08T00:00:00", "vocab": "members", "tf-idf scores": 0.15016899901764297}, {"Date": "2003-04-08T00:00:00", "vocab": "growth", "tf-idf scores": 0.1401288463571454}, {"Date": "2003-04-08T00:00:00", "vocab": "business", "tf-idf scores": 0.13952888822767953}, {"Date": "2003-04-08T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1302771129559396}, {"Date": "2003-04-08T00:00:00", "vocab": "continued", "tf-idf scores": 0.12092339276772326}, {"Date": "2003-04-08T00:00:00", "vocab": "likely", "tf-idf scores": 0.12097443818474123}, {"Date": "2003-04-08T00:00:00", "vocab": "february", "tf-idf scores": 0.10578944345978794}, {"Date": "2003-04-01T00:00:00", "vocab": "economic", "tf-idf scores": 0.19536650441492226}, {"Date": "2003-04-01T00:00:00", "vocab": "war", "tf-idf scores": 0.1843103723137229}, {"Date": "2003-04-01T00:00:00", "vocab": "iraq", "tf-idf scores": 0.1577999396022427}, {"Date": "2003-04-01T00:00:00", "vocab": "members", "tf-idf scores": 0.15016256754819166}, {"Date": "2003-04-01T00:00:00", "vocab": "growth", "tf-idf scores": 0.14016059358844032}, {"Date": "2003-04-01T00:00:00", "vocab": "business", "tf-idf scores": 0.13956664735639732}, {"Date": "2003-04-01T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1302115959492323}, {"Date": "2003-04-01T00:00:00", "vocab": "continued", "tf-idf scores": 0.12096386491509582}, {"Date": "2003-04-01T00:00:00", "vocab": "likely", "tf-idf scores": 0.12091865230441025}, {"Date": "2003-04-01T00:00:00", "vocab": "february", "tf-idf scores": 0.10574811972929211}, {"Date": "2003-03-25T00:00:00", "vocab": "economic", "tf-idf scores": 0.1953415085894862}, {"Date": "2003-03-25T00:00:00", "vocab": "war", "tf-idf scores": 0.18432428226395223}, {"Date": "2003-03-25T00:00:00", "vocab": "iraq", "tf-idf scores": 0.15776595414271424}, {"Date": "2003-03-25T00:00:00", "vocab": "members", "tf-idf scores": 0.1501385427753442}, {"Date": "2003-03-25T00:00:00", "vocab": "growth", "tf-idf scores": 0.14012819967777776}, {"Date": "2003-03-25T00:00:00", "vocab": "business", "tf-idf scores": 0.13954739177324296}, {"Date": "2003-03-25T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13023967405145223}, {"Date": "2003-03-25T00:00:00", "vocab": "continued", "tf-idf scores": 0.12088519800253768}, {"Date": "2003-03-25T00:00:00", "vocab": "likely", "tf-idf scores": 0.12090248891640311}, {"Date": "2003-03-25T00:00:00", "vocab": "february", "tf-idf scores": 0.10580529837934478}, {"Date": "2003-03-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.1953629173804316}, {"Date": "2003-03-18T00:00:00", "vocab": "war", "tf-idf scores": 0.18433284496544336}, {"Date": "2003-03-18T00:00:00", "vocab": "iraq", "tf-idf scores": 0.15782755987375705}, {"Date": "2003-03-18T00:00:00", "vocab": "members", "tf-idf scores": 0.1501285306477947}, {"Date": "2003-03-18T00:00:00", "vocab": "growth", "tf-idf scores": 0.14009953393309382}, {"Date": "2003-03-18T00:00:00", "vocab": "business", "tf-idf scores": 0.13952781106586712}, {"Date": "2003-03-18T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13023404737978359}, {"Date": "2003-03-18T00:00:00", "vocab": "continued", "tf-idf scores": 0.12097362457581443}, {"Date": "2003-03-18T00:00:00", "vocab": "likely", "tf-idf scores": 0.12090566376573944}, {"Date": "2003-03-18T00:00:00", "vocab": "february", "tf-idf scores": 0.10581236409354872}, {"Date": "2003-01-29T00:00:00", "vocab": "shall", "tf-idf scores": 0.2632619431974498}, {"Date": "2003-01-29T00:00:00", "vocab": "foreign", "tf-idf scores": 0.2209271223366767}, {"Date": "2003-01-29T00:00:00", "vocab": "market", "tf-idf scores": 0.19837200029097987}, {"Date": "2003-01-29T00:00:00", "vocab": "currency", "tf-idf scores": 0.1890555236328144}, {"Date": "2003-01-29T00:00:00", "vocab": "open", "tf-idf scores": 0.15851181138617548}, {"Date": "2003-01-29T00:00:00", "vocab": "chairman", "tf-idf scores": 0.12772791832794395}, {"Date": "2003-01-29T00:00:00", "vocab": "members", "tf-idf scores": 0.12285188252377681}, {"Date": "2003-01-29T00:00:00", "vocab": "bank", "tf-idf scores": 0.12230724221764204}, {"Date": "2003-01-29T00:00:00", "vocab": "business", "tf-idf scores": 0.12174164100490008}, {"Date": "2003-01-29T00:00:00", "vocab": "operations", "tf-idf scores": 0.11904905627364502}, {"Date": "2002-12-10T00:00:00", "vocab": "economic", "tf-idf scores": 0.28533841147342326}, {"Date": "2002-12-10T00:00:00", "vocab": "members", "tf-idf scores": 0.15425384039412904}, {"Date": "2002-12-10T00:00:00", "vocab": "growth", "tf-idf scores": 0.13309380174298055}, {"Date": "2002-12-10T00:00:00", "vocab": "continued", "tf-idf scores": 0.13254863340179615}, {"Date": "2002-12-10T00:00:00", "vocab": "november", "tf-idf scores": 0.1265944647688475}, {"Date": "2002-12-10T00:00:00", "vocab": "october", "tf-idf scores": 0.12533584664868558}, {"Date": "2002-12-10T00:00:00", "vocab": "improvement", "tf-idf scores": 0.12384914197159137}, {"Date": "2002-12-10T00:00:00", "vocab": "business", "tf-idf scores": 0.12232447539494316}, {"Date": "2002-12-10T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12230773669636351}, {"Date": "2002-12-10T00:00:00", "vocab": "remained", "tf-idf scores": 0.12232842480546495}, {"Date": "2002-11-06T00:00:00", "vocab": "economic", "tf-idf scores": 0.2356652558984021}, {"Date": "2002-11-06T00:00:00", "vocab": "members", "tf-idf scores": 0.22289118944463496}, {"Date": "2002-11-06T00:00:00", "vocab": "market", "tf-idf scores": 0.14734040025922754}, {"Date": "2002-11-06T00:00:00", "vocab": "alternatives", "tf-idf scores": 0.14557147554021502}, {"Date": "2002-11-06T00:00:00", "vocab": "study", "tf-idf scores": 0.13735402053857915}, {"Date": "2002-11-06T00:00:00", "vocab": "september", "tf-idf scores": 0.12602724441910967}, {"Date": "2002-11-06T00:00:00", "vocab": "policy", "tf-idf scores": 0.12522574961115893}, {"Date": "2002-11-06T00:00:00", "vocab": "ginnie", "tf-idf scores": 0.11803215285275402}, {"Date": "2002-11-06T00:00:00", "vocab": "business", "tf-idf scores": 0.11789410007098676}, {"Date": "2002-11-06T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11783077265706864}, {"Date": "2002-09-24T00:00:00", "vocab": "economic", "tf-idf scores": 0.19496285668781135}, {"Date": "2002-09-24T00:00:00", "vocab": "growth", "tf-idf scores": 0.16915166097200726}, {"Date": "2002-09-24T00:00:00", "vocab": "august", "tf-idf scores": 0.1669772317595942}, {"Date": "2002-09-24T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1595067666994999}, {"Date": "2002-09-24T00:00:00", "vocab": "spending", "tf-idf scores": 0.15957202964810538}, {"Date": "2002-09-24T00:00:00", "vocab": "business", "tf-idf scores": 0.14183950457792002}, {"Date": "2002-09-24T00:00:00", "vocab": "july", "tf-idf scores": 0.13514725786001341}, {"Date": "2002-09-24T00:00:00", "vocab": "capital", "tf-idf scores": 0.11721691782773379}, {"Date": "2002-09-24T00:00:00", "vocab": "activity", "tf-idf scores": 0.11526616490162479}, {"Date": "2002-09-24T00:00:00", "vocab": "continued", "tf-idf scores": 0.1152405675561165}, {"Date": "2002-08-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.20571559958159835}, {"Date": "2002-08-13T00:00:00", "vocab": "members", "tf-idf scores": 0.12971813385154826}, {"Date": "2002-08-13T00:00:00", "vocab": "growth", "tf-idf scores": 0.12920911818735148}, {"Date": "2002-08-13T00:00:00", "vocab": "business", "tf-idf scores": 0.12861880062303713}, {"Date": "2002-08-13T00:00:00", "vocab": "continued", "tf-idf scores": 0.12001001418727002}, {"Date": "2002-08-13T00:00:00", "vocab": "june", "tf-idf scores": 0.12003365408989428}, {"Date": "2002-08-13T00:00:00", "vocab": "equity", "tf-idf scores": 0.11591759747153924}, {"Date": "2002-08-13T00:00:00", "vocab": "foreseeable", "tf-idf scores": 0.11580146474192156}, {"Date": "2002-08-13T00:00:00", "vocab": "financial", "tf-idf scores": 0.11247950686593902}, {"Date": "2002-08-13T00:00:00", "vocab": "activity", "tf-idf scores": 0.11141764169735825}, {"Date": "2002-06-26T00:00:00", "vocab": "april", "tf-idf scores": 0.2027274476030131}, {"Date": "2002-06-26T00:00:00", "vocab": "members", "tf-idf scores": 0.1726432826671456}, {"Date": "2002-06-26T00:00:00", "vocab": "economic", "tf-idf scores": 0.1539559287301162}, {"Date": "2002-06-26T00:00:00", "vocab": "inflation", "tf-idf scores": 0.154036172021039}, {"Date": "2002-06-26T00:00:00", "vocab": "growth", "tf-idf scores": 0.13746915204776208}, {"Date": "2002-06-26T00:00:00", "vocab": "spending", "tf-idf scores": 0.12837346127682855}, {"Date": "2002-06-26T00:00:00", "vocab": "strength", "tf-idf scores": 0.11874740264844903}, {"Date": "2002-06-26T00:00:00", "vocab": "forecasts", "tf-idf scores": 0.11186623094942559}, {"Date": "2002-06-26T00:00:00", "vocab": "liquidation", "tf-idf scores": 0.106341315263574}, {"Date": "2002-06-26T00:00:00", "vocab": "policy", "tf-idf scores": 0.1026493609745193}, {"Date": "2002-05-07T00:00:00", "vocab": "members", "tf-idf scores": 0.1853002927757834}, {"Date": "2002-05-07T00:00:00", "vocab": "economic", "tf-idf scores": 0.15750549068452}, {"Date": "2002-05-07T00:00:00", "vocab": "inflation", "tf-idf scores": 0.13998023982341967}, {"Date": "2002-05-07T00:00:00", "vocab": "policy", "tf-idf scores": 0.1312418066760987}, {"Date": "2002-05-07T00:00:00", "vocab": "growth", "tf-idf scores": 0.12307184926419704}, {"Date": "2002-05-07T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12253000764180451}, {"Date": "2002-05-07T00:00:00", "vocab": "final", "tf-idf scores": 0.11538599501478967}, {"Date": "2002-05-07T00:00:00", "vocab": "activity", "tf-idf scores": 0.11372757069222783}, {"Date": "2002-05-07T00:00:00", "vocab": "continued", "tf-idf scores": 0.11379406823299551}, {"Date": "2002-05-07T00:00:00", "vocab": "spending", "tf-idf scores": 0.11371306677698433}, {"Date": "2002-03-19T00:00:00", "vocab": "economic", "tf-idf scores": 0.23700748604859084}, {"Date": "2002-03-19T00:00:00", "vocab": "members", "tf-idf scores": 0.19423544600354414}, {"Date": "2002-03-19T00:00:00", "vocab": "business", "tf-idf scores": 0.1555605115899967}, {"Date": "2002-03-19T00:00:00", "vocab": "inventory", "tf-idf scores": 0.12995069487312452}, {"Date": "2002-03-19T00:00:00", "vocab": "february", "tf-idf scores": 0.12635095489994705}, {"Date": "2002-03-19T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1259255567595875}, {"Date": "2002-03-19T00:00:00", "vocab": "outlook", "tf-idf scores": 0.11905800789125226}, {"Date": "2002-03-19T00:00:00", "vocab": "policy", "tf-idf scores": 0.11851109735450063}, {"Date": "2002-03-19T00:00:00", "vocab": "january", "tf-idf scores": 0.11748361273675526}, {"Date": "2002-03-19T00:00:00", "vocab": "growth", "tf-idf scores": 0.1115698529907093}, {"Date": "2002-01-30T00:00:00", "vocab": "shall", "tf-idf scores": 0.27264124765668474}, {"Date": "2002-01-30T00:00:00", "vocab": "foreign", "tf-idf scores": 0.25905971539392914}, {"Date": "2002-01-30T00:00:00", "vocab": "currency", "tf-idf scores": 0.21847934367484834}, {"Date": "2002-01-30T00:00:00", "vocab": "market", "tf-idf scores": 0.19648090226701176}, {"Date": "2002-01-30T00:00:00", "vocab": "economic", "tf-idf scores": 0.1473781609034669}, {"Date": "2002-01-30T00:00:00", "vocab": "open", "tf-idf scores": 0.14360580077006702}, {"Date": "2002-01-30T00:00:00", "vocab": "operations", "tf-idf scores": 0.11794724556133347}, {"Date": "2002-01-30T00:00:00", "vocab": "bank", "tf-idf scores": 0.11670199322740932}, {"Date": "2002-01-30T00:00:00", "vocab": "chairman", "tf-idf scores": 0.10544550483886687}, {"Date": "2002-01-30T00:00:00", "vocab": "business", "tf-idf scores": 0.10275309393109894}, {"Date": "2001-12-11T00:00:00", "vocab": "economic", "tf-idf scores": 0.2544084149076473}, {"Date": "2001-12-11T00:00:00", "vocab": "members", "tf-idf scores": 0.18663045677367154}, {"Date": "2001-12-11T00:00:00", "vocab": "september", "tf-idf scores": 0.13192180870805825}, {"Date": "2001-12-11T00:00:00", "vocab": "terrorist", "tf-idf scores": 0.13076671716343255}, {"Date": "2001-12-11T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12340384445959918}, {"Date": "2001-12-11T00:00:00", "vocab": "remained", "tf-idf scores": 0.12334840740700644}, {"Date": "2001-12-11T00:00:00", "vocab": "october", "tf-idf scores": 0.10829279389154725}, {"Date": "2001-12-11T00:00:00", "vocab": "business", "tf-idf scores": 0.10791208729915082}, {"Date": "2001-12-11T00:00:00", "vocab": "easing", "tf-idf scores": 0.10322580857497785}, {"Date": "2001-12-11T00:00:00", "vocab": "governors", "tf-idf scores": 0.10139012590532141}, {"Date": "2001-11-06T00:00:00", "vocab": "members", "tf-idf scores": 0.18148391095260438}, {"Date": "2001-11-06T00:00:00", "vocab": "economic", "tf-idf scores": 0.1635607863940723}, {"Date": "2001-11-06T00:00:00", "vocab": "weakness", "tf-idf scores": 0.15250057807069406}, {"Date": "2001-11-06T00:00:00", "vocab": "terrorist", "tf-idf scores": 0.1387522650157134}, {"Date": "2001-11-06T00:00:00", "vocab": "business", "tf-idf scores": 0.13081930131709804}, {"Date": "2001-11-06T00:00:00", "vocab": "consumer", "tf-idf scores": 0.13079310675376143}, {"Date": "2001-11-06T00:00:00", "vocab": "september", "tf-idf scores": 0.1271850366381342}, {"Date": "2001-11-06T00:00:00", "vocab": "economy", "tf-idf scores": 0.11753488199685559}, {"Date": "2001-11-06T00:00:00", "vocab": "activity", "tf-idf scores": 0.11444375520046608}, {"Date": "2001-11-06T00:00:00", "vocab": "policy", "tf-idf scores": 0.11450714097198608}, {"Date": "2001-10-02T00:00:00", "vocab": "terrorist", "tf-idf scores": 0.37846127452880146}, {"Date": "2001-10-02T00:00:00", "vocab": "september", "tf-idf scores": 0.19224909096089557}, {"Date": "2001-10-02T00:00:00", "vocab": "business", "tf-idf scores": 0.15105872538967058}, {"Date": "2001-10-02T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1441733552735681}, {"Date": "2001-10-02T00:00:00", "vocab": "economic", "tf-idf scores": 0.1373266267409174}, {"Date": "2001-10-02T00:00:00", "vocab": "august", "tf-idf scores": 0.12941048352238788}, {"Date": "2001-10-02T00:00:00", "vocab": "attacks", "tf-idf scores": 0.11357139130520454}, {"Date": "2001-10-02T00:00:00", "vocab": "spending", "tf-idf scores": 0.10992005884928333}, {"Date": "2001-10-02T00:00:00", "vocab": "july", "tf-idf scores": 0.10471851626596701}, {"Date": "2001-10-02T00:00:00", "vocab": "weakness", "tf-idf scores": 0.10062918742442756}, {"Date": "2001-09-17T00:00:00", "vocab": "business", "tf-idf scores": 0.1876124750423746}, {"Date": "2001-09-17T00:00:00", "vocab": "members", "tf-idf scores": 0.18136412066142954}, {"Date": "2001-09-17T00:00:00", "vocab": "growth", "tf-idf scores": 0.15703724511917624}, {"Date": "2001-09-17T00:00:00", "vocab": "economic", "tf-idf scores": 0.14849836873045863}, {"Date": "2001-09-17T00:00:00", "vocab": "weakness", "tf-idf scores": 0.13541624225132298}, {"Date": "2001-09-17T00:00:00", "vocab": "governors", "tf-idf scores": 0.12562227660570052}, {"Date": "2001-09-17T00:00:00", "vocab": "votes", "tf-idf scores": 0.11785623187258305}, {"Date": "2001-09-17T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1172639835203037}, {"Date": "2001-09-17T00:00:00", "vocab": "recent", "tf-idf scores": 0.11723231570783241}, {"Date": "2001-09-17T00:00:00", "vocab": "second", "tf-idf scores": 0.11636095346093998}, {"Date": "2001-09-13T00:00:00", "vocab": "business", "tf-idf scores": 0.18761973581876207}, {"Date": "2001-09-13T00:00:00", "vocab": "members", "tf-idf scores": 0.18133974684794263}, {"Date": "2001-09-13T00:00:00", "vocab": "growth", "tf-idf scores": 0.15704737734845015}, {"Date": "2001-09-13T00:00:00", "vocab": "economic", "tf-idf scores": 0.1485624539200531}, {"Date": "2001-09-13T00:00:00", "vocab": "weakness", "tf-idf scores": 0.13536458358165673}, {"Date": "2001-09-13T00:00:00", "vocab": "governors", "tf-idf scores": 0.125639089746622}, {"Date": "2001-09-13T00:00:00", "vocab": "votes", "tf-idf scores": 0.1178429993088103}, {"Date": "2001-09-13T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11729106270687077}, {"Date": "2001-09-13T00:00:00", "vocab": "recent", "tf-idf scores": 0.1172108363954108}, {"Date": "2001-09-13T00:00:00", "vocab": "second", "tf-idf scores": 0.11633484784074528}, {"Date": "2001-08-21T00:00:00", "vocab": "business", "tf-idf scores": 0.18759411921645813}, {"Date": "2001-08-21T00:00:00", "vocab": "members", "tf-idf scores": 0.18135225880726683}, {"Date": "2001-08-21T00:00:00", "vocab": "growth", "tf-idf scores": 0.15702671372352764}, {"Date": "2001-08-21T00:00:00", "vocab": "economic", "tf-idf scores": 0.14852705655248885}, {"Date": "2001-08-21T00:00:00", "vocab": "weakness", "tf-idf scores": 0.13540813602132942}, {"Date": "2001-08-21T00:00:00", "vocab": "governors", "tf-idf scores": 0.12565830219394172}, {"Date": "2001-08-21T00:00:00", "vocab": "votes", "tf-idf scores": 0.11784795390274588}, {"Date": "2001-08-21T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11728423255991197}, {"Date": "2001-08-21T00:00:00", "vocab": "recent", "tf-idf scores": 0.11729816001739937}, {"Date": "2001-08-21T00:00:00", "vocab": "second", "tf-idf scores": 0.11633072772881235}, {"Date": "2001-06-27T00:00:00", "vocab": "economic", "tf-idf scores": 0.23192899921303195}, {"Date": "2001-06-27T00:00:00", "vocab": "growth", "tf-idf scores": 0.17289292735434605}, {"Date": "2001-06-27T00:00:00", "vocab": "easing", "tf-idf scores": 0.1602178100166822}, {"Date": "2001-06-27T00:00:00", "vocab": "members", "tf-idf scores": 0.1585595784387456}, {"Date": "2001-06-27T00:00:00", "vocab": "activity", "tf-idf scores": 0.157135710814717}, {"Date": "2001-06-27T00:00:00", "vocab": "business", "tf-idf scores": 0.14963204620170786}, {"Date": "2001-06-27T00:00:00", "vocab": "weakness", "tf-idf scores": 0.1395526516526896}, {"Date": "2001-06-27T00:00:00", "vocab": "april", "tf-idf scores": 0.13643402841935492}, {"Date": "2001-06-27T00:00:00", "vocab": "continued", "tf-idf scores": 0.13472513689456125}, {"Date": "2001-06-27T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11975154792449128}, {"Date": "2001-05-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.20971896223485617}, {"Date": "2001-05-15T00:00:00", "vocab": "members", "tf-idf scores": 0.19635837571020603}, {"Date": "2001-05-15T00:00:00", "vocab": "growth", "tf-idf scores": 0.1804874008037455}, {"Date": "2001-05-15T00:00:00", "vocab": "easing", "tf-idf scores": 0.18035080392981703}, {"Date": "2001-05-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.14983067857217675}, {"Date": "2001-05-15T00:00:00", "vocab": "business", "tf-idf scores": 0.13476689485766366}, {"Date": "2001-05-15T00:00:00", "vocab": "march", "tf-idf scores": 0.1261615336512882}, {"Date": "2001-05-15T00:00:00", "vocab": "year", "tf-idf scores": 0.12037728795348551}, {"Date": "2001-05-15T00:00:00", "vocab": "spending", "tf-idf scores": 0.10490338885609}, {"Date": "2001-05-15T00:00:00", "vocab": "weakness", "tf-idf scores": 0.09980266860595345}, {"Date": "2001-04-18T00:00:00", "vocab": "january", "tf-idf scores": 0.2243022821001534}, {"Date": "2001-04-18T00:00:00", "vocab": "members", "tf-idf scores": 0.2060203973479912}, {"Date": "2001-04-18T00:00:00", "vocab": "business", "tf-idf scores": 0.1964379373932714}, {"Date": "2001-04-18T00:00:00", "vocab": "economic", "tf-idf scores": 0.18853825373235897}, {"Date": "2001-04-18T00:00:00", "vocab": "growth", "tf-idf scores": 0.1736244996670761}, {"Date": "2001-04-18T00:00:00", "vocab": "consumer", "tf-idf scores": 0.16496871557562076}, {"Date": "2001-04-18T00:00:00", "vocab": "easing", "tf-idf scores": 0.1472002476223072}, {"Date": "2001-04-18T00:00:00", "vocab": "expansion", "tf-idf scores": 0.13852258986756674}, {"Date": "2001-04-18T00:00:00", "vocab": "relatively", "tf-idf scores": 0.1269259331109534}, {"Date": "2001-04-18T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11784619270805993}, {"Date": "2001-04-11T00:00:00", "vocab": "january", "tf-idf scores": 0.22427807551291187}, {"Date": "2001-04-11T00:00:00", "vocab": "members", "tf-idf scores": 0.20610833265462727}, {"Date": "2001-04-11T00:00:00", "vocab": "business", "tf-idf scores": 0.1963898310217342}, {"Date": "2001-04-11T00:00:00", "vocab": "economic", "tf-idf scores": 0.1885570083042107}, {"Date": "2001-04-11T00:00:00", "vocab": "growth", "tf-idf scores": 0.17360325948082736}, {"Date": "2001-04-11T00:00:00", "vocab": "consumer", "tf-idf scores": 0.16502313987628445}, {"Date": "2001-04-11T00:00:00", "vocab": "easing", "tf-idf scores": 0.147259454629181}, {"Date": "2001-04-11T00:00:00", "vocab": "expansion", "tf-idf scores": 0.1385182572448747}, {"Date": "2001-04-11T00:00:00", "vocab": "relatively", "tf-idf scores": 0.12692933715107457}, {"Date": "2001-04-11T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11784409090623493}, {"Date": "2001-03-20T00:00:00", "vocab": "january", "tf-idf scores": 0.22433971619882254}, {"Date": "2001-03-20T00:00:00", "vocab": "members", "tf-idf scores": 0.2060531944343837}, {"Date": "2001-03-20T00:00:00", "vocab": "business", "tf-idf scores": 0.19641345500324534}, {"Date": "2001-03-20T00:00:00", "vocab": "economic", "tf-idf scores": 0.18858653560130062}, {"Date": "2001-03-20T00:00:00", "vocab": "growth", "tf-idf scores": 0.17361663474967026}, {"Date": "2001-03-20T00:00:00", "vocab": "consumer", "tf-idf scores": 0.16502853443507534}, {"Date": "2001-03-20T00:00:00", "vocab": "easing", "tf-idf scores": 0.14722875685001532}, {"Date": "2001-03-20T00:00:00", "vocab": "expansion", "tf-idf scores": 0.13854968237565882}, {"Date": "2001-03-20T00:00:00", "vocab": "relatively", "tf-idf scores": 0.12691163970583047}, {"Date": "2001-03-20T00:00:00", "vocab": "conditions", "tf-idf scores": 0.11790923251567736}, {"Date": "2001-01-31T00:00:00", "vocab": "shall", "tf-idf scores": 0.23783508311190907}, {"Date": "2001-01-31T00:00:00", "vocab": "foreign", "tf-idf scores": 0.21821128104824694}, {"Date": "2001-01-31T00:00:00", "vocab": "market", "tf-idf scores": 0.19092251886272874}, {"Date": "2001-01-31T00:00:00", "vocab": "currency", "tf-idf scores": 0.18380825211620613}, {"Date": "2001-01-31T00:00:00", "vocab": "members", "tf-idf scores": 0.16122560783814}, {"Date": "2001-01-31T00:00:00", "vocab": "open", "tf-idf scores": 0.13311084085768213}, {"Date": "2001-01-31T00:00:00", "vocab": "operations", "tf-idf scores": 0.12345709599983835}, {"Date": "2001-01-31T00:00:00", "vocab": "business", "tf-idf scores": 0.12079234451778766}, {"Date": "2001-01-31T00:00:00", "vocab": "economic", "tf-idf scores": 0.1207847891344029}, {"Date": "2001-01-31T00:00:00", "vocab": "securities", "tf-idf scores": 0.11448126042494475}, {"Date": "2001-01-03T00:00:00", "vocab": "growth", "tf-idf scores": 0.2529135513696266}, {"Date": "2001-01-03T00:00:00", "vocab": "economic", "tf-idf scores": 0.21679356778639347}, {"Date": "2001-01-03T00:00:00", "vocab": "expansion", "tf-idf scores": 0.17262450484753994}, {"Date": "2001-01-03T00:00:00", "vocab": "october", "tf-idf scores": 0.1719973188141204}, {"Date": "2001-01-03T00:00:00", "vocab": "members", "tf-idf scores": 0.1411248547810888}, {"Date": "2001-01-03T00:00:00", "vocab": "business", "tf-idf scores": 0.12595805330495757}, {"Date": "2001-01-03T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12596013342438467}, {"Date": "2001-01-03T00:00:00", "vocab": "november", "tf-idf scores": 0.12410952686849899}, {"Date": "2001-01-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11895279132272542}, {"Date": "2001-01-03T00:00:00", "vocab": "risks", "tf-idf scores": 0.11247269143399528}, {"Date": "2000-12-19T00:00:00", "vocab": "growth", "tf-idf scores": 0.25291537361611}, {"Date": "2000-12-19T00:00:00", "vocab": "economic", "tf-idf scores": 0.2168006394228249}, {"Date": "2000-12-19T00:00:00", "vocab": "expansion", "tf-idf scores": 0.1726330382655655}, {"Date": "2000-12-19T00:00:00", "vocab": "october", "tf-idf scores": 0.1720087042091317}, {"Date": "2000-12-19T00:00:00", "vocab": "members", "tf-idf scores": 0.14110528939789807}, {"Date": "2000-12-19T00:00:00", "vocab": "business", "tf-idf scores": 0.1259415431324343}, {"Date": "2000-12-19T00:00:00", "vocab": "consumer", "tf-idf scores": 0.12595824792218455}, {"Date": "2000-12-19T00:00:00", "vocab": "november", "tf-idf scores": 0.12415679730519312}, {"Date": "2000-12-19T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11892444209520583}, {"Date": "2000-12-19T00:00:00", "vocab": "risks", "tf-idf scores": 0.1124105473961893}, {"Date": "2000-11-15T00:00:00", "vocab": "growth", "tf-idf scores": 0.2984542378773542}, {"Date": "2000-11-15T00:00:00", "vocab": "inflation", "tf-idf scores": 0.18776645617307952}, {"Date": "2000-11-15T00:00:00", "vocab": "economic", "tf-idf scores": 0.15646730568289538}, {"Date": "2000-11-15T00:00:00", "vocab": "expansion", "tf-idf scores": 0.15628041994076058}, {"Date": "2000-11-15T00:00:00", "vocab": "members", "tf-idf scores": 0.1499520213560436}, {"Date": "2000-11-15T00:00:00", "vocab": "october", "tf-idf scores": 0.12365857295142656}, {"Date": "2000-11-15T00:00:00", "vocab": "prices", "tf-idf scores": 0.11790651447993808}, {"Date": "2000-11-15T00:00:00", "vocab": "consumer", "tf-idf scores": 0.11730170233630875}, {"Date": "2000-11-15T00:00:00", "vocab": "energy", "tf-idf scores": 0.11051227397605465}, {"Date": "2000-11-15T00:00:00", "vocab": "direction", "tf-idf scores": 0.10206808943002124}, {"Date": "2000-10-03T00:00:00", "vocab": "growth", "tf-idf scores": 0.2534391614277661}, {"Date": "2000-10-03T00:00:00", "vocab": "august", "tf-idf scores": 0.22897689516171657}, {"Date": "2000-10-03T00:00:00", "vocab": "july", "tf-idf scores": 0.17814053717487266}, {"Date": "2000-10-03T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1588836294710116}, {"Date": "2000-10-03T00:00:00", "vocab": "prices", "tf-idf scores": 0.15024949644014293}, {"Date": "2000-10-03T00:00:00", "vocab": "recent", "tf-idf scores": 0.14958452590526616}, {"Date": "2000-10-03T00:00:00", "vocab": "members", "tf-idf scores": 0.1414904938996594}, {"Date": "2000-10-03T00:00:00", "vocab": "expansion", "tf-idf scores": 0.13188668245044163}, {"Date": "2000-10-03T00:00:00", "vocab": "gains", "tf-idf scores": 0.1297619407962027}, {"Date": "2000-10-03T00:00:00", "vocab": "somewhat", "tf-idf scores": 0.11319699333119186}, {"Date": "2000-08-22T00:00:00", "vocab": "productivity", "tf-idf scores": 0.24446973733501456}, {"Date": "2000-08-22T00:00:00", "vocab": "growth", "tf-idf scores": 0.23673567963542758}, {"Date": "2000-08-22T00:00:00", "vocab": "prices", "tf-idf scores": 0.1756690787360517}, {"Date": "2000-08-22T00:00:00", "vocab": "members", "tf-idf scores": 0.15339076346355396}, {"Date": "2000-08-22T00:00:00", "vocab": "consumer", "tf-idf scores": 0.15211870565871716}, {"Date": "2000-08-22T00:00:00", "vocab": "recent", "tf-idf scores": 0.1445284643529559}, {"Date": "2000-08-22T00:00:00", "vocab": "demand", "tf-idf scores": 0.13994710816040196}, {"Date": "2000-08-22T00:00:00", "vocab": "inflation", "tf-idf scores": 0.1369122754511422}, {"Date": "2000-08-22T00:00:00", "vocab": "labor", "tf-idf scores": 0.1140816159511267}, {"Date": "2000-08-22T00:00:00", "vocab": "june", "tf-idf scores": 0.10644510409094599}, {"Date": "2000-06-28T00:00:00", "vocab": "growth", "tf-idf scores": 0.22171761752449587}, {"Date": "2000-06-28T00:00:00", "vocab": "members", "tf-idf scores": 0.17503801106100972}, {"Date": "2000-06-28T00:00:00", "vocab": "consumer", "tf-idf scores": 0.1577575098488144}, {"Date": "2000-06-28T00:00:00", "vocab": "prices", "tf-idf scores": 0.15045699098291151}, {"Date": "2000-06-28T00:00:00", "vocab": "ranges", "tf-idf scores": 0.14528036213467055}, {"Date": "2000-06-28T00:00:00", "vocab": "expansion", "tf-idf scores": 0.13899505575307414}, {"Date": "2000-06-28T00:00:00", "vocab": "april", "tf-idf scores": 0.12940622115162914}, {"Date": "2000-06-28T00:00:00", "vocab": "economic", "tf-idf scores": 0.12621605134496372}, {"Date": "2000-06-28T00:00:00", "vocab": "inflation", "tf-idf scores": 0.11828790231710484}, {"Date": "2000-06-28T00:00:00", "vocab": "indications", "tf-idf scores": 0.11146986668693806}, {"Date": "2000-05-16T00:00:00", "vocab": "growth", "tf-idf scores": 0.2254573535563378}, {"Date": "2000-05-16T00:00:00", "vocab": "demand", "tf-idf scores": 0.17433878548485118}, {"Date": "2000-05-16T00:00:00", "vocab": "members", "tf-idf scores": 0.1539822831064848}, {"Date": "2000-05-16T00:00:00", "vocab": "april", "tf-idf scores": 0.14730182963957394}, {"Date": "2000-05-16T00:00:00", "vocab": "labor", "tf-idf scores": 0.13474949582172502}, {"Date": "2000-05-16T00:00:00", "vocab": "indications", "tf-idf scores": 0.10881674293404817}, {"Date": "2000-05-16T00:00:00", "vocab": "prices", "tf-idf scores": 0.10823129188089962}, {"Date": "2000-05-16T00:00:00", "vocab": "economic", "tf-idf scores": 0.10781859776043679}, {"Date": "2000-05-16T00:00:00", "vocab": "inflation", "tf-idf scores": 0.10778333062550045}, {"Date": "2000-05-16T00:00:00", "vocab": "recent", "tf-idf scores": 0.10778506939360813}, {"Date": "2000-03-21T00:00:00", "vocab": "growth", "tf-idf scores": 0.18827868829269798}, {"Date": "2000-03-21T00:00:00", "vocab": "members", "tf-idf scores": 0.14801504747371863}, {"Date": "2000-03-21T00:00:00", "vocab": "acceleration", "tf-idf scores": 0.1365435360929158}, {"Date": "2000-03-21T00:00:00", "vocab": "aggregate", "tf-idf scores": 0.13476951338067056}, {"Date": "2000-03-21T00:00:00", "vocab": "demand", "tf-idf scores": 0.13335252792848257}, {"Date": "2000-03-21T00:00:00", "vocab": "february", "tf-idf scores": 0.12362938040389974}, {"Date": "2000-03-21T00:00:00", "vocab": "inflation", "tf-idf scores": 0.12225219714157802}, {"Date": "2000-03-21T00:00:00", "vocab": "collateral", "tf-idf scores": 0.11892877365916595}, {"Date": "2000-03-21T00:00:00", "vocab": "century", "tf-idf scores": 0.11806679774003047}, {"Date": "2000-03-21T00:00:00", "vocab": "january", "tf-idf scores": 0.11641662354074554}, {"Date": "2000-02-02T00:00:00", "vocab": "shall", "tf-idf scores": 0.25627932534955505}, {"Date": "2000-02-02T00:00:00", "vocab": "foreign", "tf-idf scores": 0.2225671552389485}, {"Date": "2000-02-02T00:00:00", "vocab": "ranges", "tf-idf scores": 0.2209960731401622}, {"Date": "2000-02-02T00:00:00", "vocab": "currency", "tf-idf scores": 0.1833480357259736}, {"Date": "2000-02-02T00:00:00", "vocab": "growth", "tf-idf scores": 0.16869198610389388}, {"Date": "2000-02-02T00:00:00", "vocab": "market", "tf-idf scores": 0.155394480409516}, {"Date": "2000-02-02T00:00:00", "vocab": "bank", "tf-idf scores": 0.12649740276763885}, {"Date": "2000-02-02T00:00:00", "vocab": "members", "tf-idf scores": 0.12286449062698462}, {"Date": "2000-02-02T00:00:00", "vocab": "open", "tf-idf scores": 0.1180859278659384}, {"Date": "2000-02-02T00:00:00", "vocab": "new", "tf-idf scores": 0.11337527556268391}]}}, {"mode": "vega-lite"});
</script>



#### 3.3.2 **Conclusion**

There are 3 periods of rate rise:
1. June 2004 to July 2006
2. Dec 2015 to Dec 2018
3. March 2022 to August 2023

There are periods of rate cuts:
1. Till Dec 2001
2. Sept 2007 to Dec 2008
3. July 2019 to April 2020
4. Since Sept 2024

From the heatmap, we can confirm the hypothesis that A **significant shift in the language used to describe the economic outlook signals a high probability of changes in policy direct in the incoming FOMC meetings**:
1. When "inflation" became the highest tf-idf words in 2005,and in early 2015, and in late 2021, there are policy rate rise within half a year
2. Words such as "war", "pandemic", and "growth" often appeared before rate cuts, showing that the exogenous negative shocks and concerns over economic growth can often post significant pressure on the Fed decision. The Fed would then choose to cut the rate

### 3.3.3 Setiment Test

#### 3.3.3.1 Function to process data
This part will create a function to process the cleaned text and analyze the sentiment score.

The sentiment score is referred to the ['Loughran-McDonald Master Dictionary w/ Sentiment Word Lists'](https://sraf.nd.edu/loughranmcdonald-master-dictionary/).

The sentiment score is now integer, with greater number marking positive.


```python
from collections import Counter
lm_dict = pd.read_csv('/content/drive/MyDrive/text_analysis/Financial_dic.csv')
lm_words = set(lm_dict['Word'])

# Preprocess text
def preprocess(text):
    return [word.upper() for word in text.split() if word.isalpha()]

# Match words to dictionary
def sentiment_analysis(text):
    words = preprocess(text)
    word_count = Counter(words)
    sentiment_counts = {cat: 0 for cat in ['Positive', 'Negative', 'Uncertainty']}

    for word in words:
        if word in lm_words:
            for cat in sentiment_counts.keys():
                if word in lm_dict[lm_dict[cat] > 0]['Word'].values:
                    sentiment_counts[cat] += word_count[word]
    return sentiment_counts


def analyze_sentiment_statements(df):
    sentiment_scores = []
    for index, row in df.iterrows():
        text = row['clean no_stop text']
        sentiment_counts = sentiment_analysis(text)  # Get sentiment counts

        # Calculate a single sentiment score (e.g., positive - negative)
        sentiment_score = sentiment_counts['Positive'] - sentiment_counts['Negative']

        sentiment_scores.append([row['Date'], sentiment_score])

    sentiment_df = pd.DataFrame(sentiment_scores, columns=['Date', 'Sentiment Score'])
    return sentiment_df

```

#### 3.3.3.2 Applying function to the text
Due to over 30 minutes processing time, I have save the processed result into a csv file in my Google Doc


```python
statements_sentiment = analyze_sentiment_statements(statements)
```


```python
minutes_sentiment = analyze_sentiment_statements(minutes)
```


```python
# the following part is to read the data quickly withnot requiring the program to process the data again
'''
minutes_sentiment = pd.read_csv('/content/drive/MyDrive/text_analysis/minutes_sentiment_big.csv')
statements_sentiment = pd.read_csv('/content/drive/MyDrive/text_analysis/statements_sentiment.csv')
statements_sentiment["Date"] = pd.to_datetime(statements_sentiment["Date"])
minutes_sentiment["Date"] = pd.to_datetime(minutes_sentiment["Date"])
'''
```




    '\nminutes_sentiment = pd.read_csv(\'/content/drive/MyDrive/text_analysis/minutes_sentiment_big.csv\')\nstatements_sentiment = pd.read_csv(\'/content/drive/MyDrive/text_analysis/statements_sentiment.csv\')\nstatements_sentiment["Date"] = pd.to_datetime(statements_sentiment["Date"])\nminutes_sentiment["Date"] = pd.to_datetime(minutes_sentiment["Date"])\n'



**Result Preview:**


```python
statements_sentiment.head()
```





  <div id="df-bc8fae10-7a54-4b8a-9756-490f0f60d0c4" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Sentiment Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-11-07</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-09-18</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-07-31</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-06-12</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-05-01</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-bc8fae10-7a54-4b8a-9756-490f0f60d0c4')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-bc8fae10-7a54-4b8a-9756-490f0f60d0c4 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-bc8fae10-7a54-4b8a-9756-490f0f60d0c4');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-3d99abe9-a7fb-45e3-be8e-9737639e662f">
  <button class="colab-df-quickchart" onclick="quickchart('df-3d99abe9-a7fb-45e3-be8e-9737639e662f')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-3d99abe9-a7fb-45e3-be8e-9737639e662f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
minutes_sentiment.head()
```





  <div id="df-96918a22-bf90-4706-bda7-48fab922dbc3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Sentiment Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-11-07</td>
      <td>-250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-09-18</td>
      <td>-336</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-07-31</td>
      <td>-163</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-06-12</td>
      <td>-57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-05-01</td>
      <td>-115</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-96918a22-bf90-4706-bda7-48fab922dbc3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-96918a22-bf90-4706-bda7-48fab922dbc3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-96918a22-bf90-4706-bda7-48fab922dbc3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-c190a3a8-2dc9-4f7c-96d6-d589bc833d82">
  <button class="colab-df-quickchart" onclick="quickchart('df-c190a3a8-2dc9-4f7c-96d6-d589bc833d82')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c190a3a8-2dc9-4f7c-96d6-d589bc833d82 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Merge interest rates with statements
minutes_sentiment_date = pd.merge(minutes_sentiment, final_rates, left_on='Date', right_on='Date', how='left')

# Merge interest rates with minutes
statements_sentiment_date = pd.merge(statements_sentiment, final_rates, left_on='Date', right_on='Date', how='left')
```


```python
minutes_sentiment_date.head()
```





  <div id="df-5034030d-1038-40a2-85fd-f975bca55e81" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Sentiment Score</th>
      <th>Policy Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-11-07</td>
      <td>-250</td>
      <td>4.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-09-18</td>
      <td>-336</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-07-31</td>
      <td>-163</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-06-12</td>
      <td>-57</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-05-01</td>
      <td>-115</td>
      <td>5.25</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5034030d-1038-40a2-85fd-f975bca55e81')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5034030d-1038-40a2-85fd-f975bca55e81 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5034030d-1038-40a2-85fd-f975bca55e81');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-1da1bb30-7a2f-4088-98e7-551f2a108a12">
  <button class="colab-df-quickchart" onclick="quickchart('df-1da1bb30-7a2f-4088-98e7-551f2a108a12')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-1da1bb30-7a2f-4088-98e7-551f2a108a12 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
statements_sentiment_date.head()
```





  <div id="df-d8c12b39-4a95-4fba-847e-0a5afe936b25" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Sentiment Score</th>
      <th>Policy Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-11-07</td>
      <td>2</td>
      <td>4.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-09-18</td>
      <td>7</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-07-31</td>
      <td>5</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-06-12</td>
      <td>5</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-05-01</td>
      <td>2</td>
      <td>5.25</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d8c12b39-4a95-4fba-847e-0a5afe936b25')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d8c12b39-4a95-4fba-847e-0a5afe936b25 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d8c12b39-4a95-4fba-847e-0a5afe936b25');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b61f0e6e-500c-4804-880d-e1f9cc62610c">
  <button class="colab-df-quickchart" onclick="quickchart('df-b61f0e6e-500c-4804-880d-e1f9cc62610c')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b61f0e6e-500c-4804-880d-e1f9cc62610c button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




#### 3.3.3.3 Data Visualization


```python
fig5 = make_subplots(specs=[[{"secondary_y": True}]])

fig5.add_trace(
    go.Scatter(
        x=statements_sentiment_date["Date"],
        y=statements_sentiment_date["Sentiment Score"],
        name="Sentiment Score of the Statement'",
    ),
    secondary_y=False,
)

fig5.add_trace(
    go.Scatter(
        x=statements_sentiment_date["Date"],
        y=statements_sentiment_date["Policy Rate"],
        name="Policy Rate",
    ),
    secondary_y=True,
)

fig5.update_layout(title_text=f"Sentiment Score vs. Policy Rate (statement information)")

fig5.update_xaxes(title_text="Date")

fig5.update_yaxes(title_text="Sentiment Score of the Statement", secondary_y=False)
fig5.update_yaxes(title_text="Policy Rate", secondary_y=True)

fig5.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="0f5562e5-96f1-4718-9833-7bc595ac33d3" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("0f5562e5-96f1-4718-9833-7bc595ac33d3")) {                    Plotly.newPlot(                        "0f5562e5-96f1-4718-9833-7bc595ac33d3",                        [{"name":"Sentiment Score of the Statement'","x":["2024-11-07T00:00:00","2024-09-18T00:00:00","2024-07-31T00:00:00","2024-06-12T00:00:00","2024-05-01T00:00:00","2024-03-20T00:00:00","2024-01-31T00:00:00","2023-12-13T00:00:00","2023-11-01T00:00:00","2023-09-20T00:00:00","2023-07-26T00:00:00","2023-06-14T00:00:00","2023-05-03T00:00:00","2023-03-22T00:00:00","2023-02-01T00:00:00","2022-12-14T00:00:00","2022-11-02T00:00:00","2022-09-21T00:00:00","2022-07-27T00:00:00","2022-06-15T00:00:00","2022-05-04T00:00:00","2022-03-16T00:00:00","2022-01-26T00:00:00","2021-12-15T00:00:00","2021-11-03T00:00:00","2021-09-22T00:00:00","2021-07-28T00:00:00","2021-06-16T00:00:00","2021-04-28T00:00:00","2021-03-17T00:00:00","2021-01-27T00:00:00","2020-12-16T00:00:00","2020-11-05T00:00:00","2020-09-16T00:00:00","2020-07-29T00:00:00","2020-06-10T00:00:00","2020-04-29T00:00:00","2020-03-23T00:00:00","2020-03-15T00:00:00","2020-03-03T00:00:00","2020-01-29T00:00:00","2019-12-11T00:00:00","2019-10-30T00:00:00","2019-10-04T00:00:00","2019-09-18T00:00:00","2019-07-31T00:00:00","2019-06-19T00:00:00","2019-05-01T00:00:00","2019-03-20T00:00:00","2019-01-30T00:00:00","2018-12-19T00:00:00","2018-11-08T00:00:00","2018-09-26T00:00:00","2018-08-01T00:00:00","2018-06-13T00:00:00","2018-05-02T00:00:00","2018-03-21T00:00:00","2018-01-31T00:00:00","2017-12-13T00:00:00","2017-11-01T00:00:00","2017-09-20T00:00:00","2017-07-26T00:00:00","2017-06-14T00:00:00","2017-05-03T00:00:00","2017-03-15T00:00:00","2017-02-01T00:00:00","2016-12-14T00:00:00","2016-11-02T00:00:00","2016-09-21T00:00:00","2016-07-27T00:00:00","2016-06-15T00:00:00","2016-04-27T00:00:00","2016-03-16T00:00:00","2016-01-27T00:00:00","2015-12-16T00:00:00","2015-10-28T00:00:00","2015-09-17T00:00:00","2015-07-29T00:00:00","2015-06-17T00:00:00","2015-04-29T00:00:00","2015-03-18T00:00:00","2015-01-28T00:00:00","2014-12-17T00:00:00","2014-10-29T00:00:00","2014-09-17T00:00:00","2014-07-30T00:00:00","2014-06-18T00:00:00","2014-04-30T00:00:00","2014-03-19T00:00:00","2014-01-29T00:00:00","2013-12-18T00:00:00","2013-10-30T00:00:00","2013-09-18T00:00:00","2013-07-31T00:00:00","2013-06-19T00:00:00","2013-05-01T00:00:00","2013-03-20T00:00:00","2013-01-30T00:00:00","2012-12-12T00:00:00","2012-10-24T00:00:00","2012-09-13T00:00:00","2012-08-01T00:00:00","2012-06-20T00:00:00","2012-04-25T00:00:00","2012-03-13T00:00:00","2012-01-25T00:00:00","2011-12-13T00:00:00","2011-11-02T00:00:00","2011-09-21T00:00:00","2011-08-09T00:00:00","2011-06-22T00:00:00","2011-04-27T00:00:00","2011-03-15T00:00:00","2011-01-26T00:00:00","2010-12-14T00:00:00","2010-11-03T00:00:00","2010-09-21T00:00:00","2010-08-10T00:00:00","2010-06-23T00:00:00","2010-05-09T00:00:00","2010-04-28T00:00:00","2010-03-16T00:00:00","2010-01-27T00:00:00","2009-12-15T00:00:00","2009-11-04T00:00:00","2009-09-22T00:00:00","2009-08-11T00:00:00","2009-06-24T00:00:00","2009-04-29T00:00:00","2009-03-17T00:00:00","2009-01-28T00:00:00","2008-12-16T00:00:00","2008-10-29T00:00:00","2008-10-07T00:00:00","2008-09-16T00:00:00","2008-08-08T00:00:00","2008-06-25T00:00:00","2008-04-30T00:00:00","2008-03-18T00:00:00","2008-03-10T00:00:00","2008-01-30T00:00:00","2008-01-21T00:00:00","2007-12-11T00:00:00","2007-10-31T00:00:00","2007-09-18T00:00:00","2007-08-16T00:00:00","2007-08-10T00:00:00","2007-08-07T00:00:00","2007-06-28T00:00:00","2007-05-09T00:00:00","2007-03-21T00:00:00","2007-01-31T00:00:00","2006-12-12T00:00:00","2006-10-25T00:00:00","2006-09-20T00:00:00","2006-08-08T00:00:00","2006-06-29T00:00:00","2006-05-10T00:00:00","2006-03-28T00:00:00","2006-01-31T00:00:00","2005-12-13T00:00:00","2005-11-01T00:00:00","2005-09-20T00:00:00","2005-08-09T00:00:00","2005-06-30T00:00:00","2005-05-03T00:00:00","2005-03-22T00:00:00","2005-02-02T00:00:00","2004-12-14T00:00:00","2004-11-10T00:00:00","2004-09-21T00:00:00","2004-08-10T00:00:00","2004-06-30T00:00:00","2004-05-04T00:00:00","2004-03-16T00:00:00","2004-01-28T00:00:00","2003-12-09T00:00:00","2003-10-28T00:00:00","2003-09-16T00:00:00","2003-08-12T00:00:00","2003-06-25T00:00:00","2003-05-06T00:00:00","2003-03-18T00:00:00","2003-01-29T00:00:00","2002-12-10T00:00:00","2002-11-06T00:00:00","2002-09-24T00:00:00","2002-08-13T00:00:00","2002-06-26T00:00:00","2002-05-07T00:00:00","2002-03-19T00:00:00","2002-01-30T00:00:00","2001-12-11T00:00:00","2001-11-06T00:00:00","2001-10-02T00:00:00","2001-09-17T00:00:00","2001-08-21T00:00:00","2001-06-27T00:00:00","2001-05-15T00:00:00","2001-04-18T00:00:00","2001-03-20T00:00:00","2001-01-31T00:00:00","2001-01-03T00:00:00","2000-12-19T00:00:00","2000-11-15T00:00:00","2000-10-03T00:00:00","2000-08-22T00:00:00","2000-06-28T00:00:00","2000-05-16T00:00:00","2000-03-21T00:00:00","2000-02-02T00:00:00"],"y":[2,7,5,5,2,4,4,-1,0,-2,-1,-1,-1,0,0,0,0,1,1,0,-1,1,4,8,12,31,32,21,14,8,5,4,0,2,4,2,0,7,13,1,4,9,9,-2,16,9,9,2,2,2,5,5,5,10,5,5,12,3,-2,0,2,5,5,4,6,6,2,10,1,-5,-5,-1,-3,-6,6,-2,-2,-2,-2,-10,-5,2,9,13,24,15,15,11,7,-5,-15,-3,8,-7,-6,-4,-4,-9,4,6,5,-1,-3,3,2,1,-1,-2,-4,-9,-8,6,3,2,-1,-1,3,1,1,-3,3,5,4,2,3,2,4,4,-2,0,-1,-5,-5,-12,-6,-2,-2,-4,-5,0,-1,-4,-1,-3,-3,-2,0,-3,0,-3,-2,2,-1,-1,0,0,2,2,3,2,3,0,-1,5,3,2,4,4,4,3,5,4,2,2,-2,0,0,-1,-1,-1,1,4,2,2,0,0,1,-1,2,2,3,-1,0,2,1,1,-3,-2,-1,-6,-6,-2,1,-7,-1,3,1,2,-2,-2,-2],"type":"scatter","xaxis":"x","yaxis":"y"},{"name":"Policy Rate","x":["2024-11-07T00:00:00","2024-09-18T00:00:00","2024-07-31T00:00:00","2024-06-12T00:00:00","2024-05-01T00:00:00","2024-03-20T00:00:00","2024-01-31T00:00:00","2023-12-13T00:00:00","2023-11-01T00:00:00","2023-09-20T00:00:00","2023-07-26T00:00:00","2023-06-14T00:00:00","2023-05-03T00:00:00","2023-03-22T00:00:00","2023-02-01T00:00:00","2022-12-14T00:00:00","2022-11-02T00:00:00","2022-09-21T00:00:00","2022-07-27T00:00:00","2022-06-15T00:00:00","2022-05-04T00:00:00","2022-03-16T00:00:00","2022-01-26T00:00:00","2021-12-15T00:00:00","2021-11-03T00:00:00","2021-09-22T00:00:00","2021-07-28T00:00:00","2021-06-16T00:00:00","2021-04-28T00:00:00","2021-03-17T00:00:00","2021-01-27T00:00:00","2020-12-16T00:00:00","2020-11-05T00:00:00","2020-09-16T00:00:00","2020-07-29T00:00:00","2020-06-10T00:00:00","2020-04-29T00:00:00","2020-03-23T00:00:00","2020-03-15T00:00:00","2020-03-03T00:00:00","2020-01-29T00:00:00","2019-12-11T00:00:00","2019-10-30T00:00:00","2019-10-04T00:00:00","2019-09-18T00:00:00","2019-07-31T00:00:00","2019-06-19T00:00:00","2019-05-01T00:00:00","2019-03-20T00:00:00","2019-01-30T00:00:00","2018-12-19T00:00:00","2018-11-08T00:00:00","2018-09-26T00:00:00","2018-08-01T00:00:00","2018-06-13T00:00:00","2018-05-02T00:00:00","2018-03-21T00:00:00","2018-01-31T00:00:00","2017-12-13T00:00:00","2017-11-01T00:00:00","2017-09-20T00:00:00","2017-07-26T00:00:00","2017-06-14T00:00:00","2017-05-03T00:00:00","2017-03-15T00:00:00","2017-02-01T00:00:00","2016-12-14T00:00:00","2016-11-02T00:00:00","2016-09-21T00:00:00","2016-07-27T00:00:00","2016-06-15T00:00:00","2016-04-27T00:00:00","2016-03-16T00:00:00","2016-01-27T00:00:00","2015-12-16T00:00:00","2015-10-28T00:00:00","2015-09-17T00:00:00","2015-07-29T00:00:00","2015-06-17T00:00:00","2015-04-29T00:00:00","2015-03-18T00:00:00","2015-01-28T00:00:00","2014-12-17T00:00:00","2014-10-29T00:00:00","2014-09-17T00:00:00","2014-07-30T00:00:00","2014-06-18T00:00:00","2014-04-30T00:00:00","2014-03-19T00:00:00","2014-01-29T00:00:00","2013-12-18T00:00:00","2013-10-30T00:00:00","2013-09-18T00:00:00","2013-07-31T00:00:00","2013-06-19T00:00:00","2013-05-01T00:00:00","2013-03-20T00:00:00","2013-01-30T00:00:00","2012-12-12T00:00:00","2012-10-24T00:00:00","2012-09-13T00:00:00","2012-08-01T00:00:00","2012-06-20T00:00:00","2012-04-25T00:00:00","2012-03-13T00:00:00","2012-01-25T00:00:00","2011-12-13T00:00:00","2011-11-02T00:00:00","2011-09-21T00:00:00","2011-08-09T00:00:00","2011-06-22T00:00:00","2011-04-27T00:00:00","2011-03-15T00:00:00","2011-01-26T00:00:00","2010-12-14T00:00:00","2010-11-03T00:00:00","2010-09-21T00:00:00","2010-08-10T00:00:00","2010-06-23T00:00:00","2010-05-09T00:00:00","2010-04-28T00:00:00","2010-03-16T00:00:00","2010-01-27T00:00:00","2009-12-15T00:00:00","2009-11-04T00:00:00","2009-09-22T00:00:00","2009-08-11T00:00:00","2009-06-24T00:00:00","2009-04-29T00:00:00","2009-03-17T00:00:00","2009-01-28T00:00:00","2008-12-16T00:00:00","2008-10-29T00:00:00","2008-10-07T00:00:00","2008-09-16T00:00:00","2008-08-08T00:00:00","2008-06-25T00:00:00","2008-04-30T00:00:00","2008-03-18T00:00:00","2008-03-10T00:00:00","2008-01-30T00:00:00","2008-01-21T00:00:00","2007-12-11T00:00:00","2007-10-31T00:00:00","2007-09-18T00:00:00","2007-08-16T00:00:00","2007-08-10T00:00:00","2007-08-07T00:00:00","2007-06-28T00:00:00","2007-05-09T00:00:00","2007-03-21T00:00:00","2007-01-31T00:00:00","2006-12-12T00:00:00","2006-10-25T00:00:00","2006-09-20T00:00:00","2006-08-08T00:00:00","2006-06-29T00:00:00","2006-05-10T00:00:00","2006-03-28T00:00:00","2006-01-31T00:00:00","2005-12-13T00:00:00","2005-11-01T00:00:00","2005-09-20T00:00:00","2005-08-09T00:00:00","2005-06-30T00:00:00","2005-05-03T00:00:00","2005-03-22T00:00:00","2005-02-02T00:00:00","2004-12-14T00:00:00","2004-11-10T00:00:00","2004-09-21T00:00:00","2004-08-10T00:00:00","2004-06-30T00:00:00","2004-05-04T00:00:00","2004-03-16T00:00:00","2004-01-28T00:00:00","2003-12-09T00:00:00","2003-10-28T00:00:00","2003-09-16T00:00:00","2003-08-12T00:00:00","2003-06-25T00:00:00","2003-05-06T00:00:00","2003-03-18T00:00:00","2003-01-29T00:00:00","2002-12-10T00:00:00","2002-11-06T00:00:00","2002-09-24T00:00:00","2002-08-13T00:00:00","2002-06-26T00:00:00","2002-05-07T00:00:00","2002-03-19T00:00:00","2002-01-30T00:00:00","2001-12-11T00:00:00","2001-11-06T00:00:00","2001-10-02T00:00:00","2001-09-17T00:00:00","2001-08-21T00:00:00","2001-06-27T00:00:00","2001-05-15T00:00:00","2001-04-18T00:00:00","2001-03-20T00:00:00","2001-01-31T00:00:00","2001-01-03T00:00:00","2000-12-19T00:00:00","2000-11-15T00:00:00","2000-10-03T00:00:00","2000-08-22T00:00:00","2000-06-28T00:00:00","2000-05-16T00:00:00","2000-03-21T00:00:00","2000-02-02T00:00:00"],"y":[4.75,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.0,5.0,4.75,4.5,4.25,3.75,3.0,2.25,1.5,0.75,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.5,1.5,1.5,1.75,1.75,2.0,2.25,2.25,2.25,2.25,2.25,2.0,2.0,1.75,1.75,1.5,1.5,1.25,1.25,1.0,1.0,1.0,1.0,0.75,0.75,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,2.0,2.0,2.0,2.0,2.25,3.0,3.0,4.25,4.25,4.5,4.75,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.0,4.75,4.5,4.25,4.0,3.75,3.5,3.25,3.0,2.75,2.5,2.25,2.0,1.75,1.5,1.25,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.25,1.25,1.25,1.25,1.25,1.75,1.75,1.75,1.75,1.75,1.75,1.75,2.0,2.5,3.0,3.5,3.75,4.0,4.5,5.0,5.5,6.0,6.5,6.5,6.5,6.5,6.5,6.5,6.0,5.75],"type":"scatter","xaxis":"x","yaxis":"y2"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.94],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Sentiment Score of the Statement"}},"yaxis2":{"anchor":"x","overlaying":"y","side":"right","title":{"text":"Policy Rate"}},"title":{"text":"Sentiment Score vs. Policy Rate (statement information)"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('0f5562e5-96f1-4718-9833-7bc595ac33d3');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```python
fig6 = make_subplots(specs=[[{"secondary_y": True}]])

fig6.add_trace(
    go.Scatter(
        x=minutes_sentiment_date["Date"],
        y=minutes_sentiment_date["Sentiment Score"],
        name="Sentiment Score of the Minutes'",
    ),
    secondary_y=False,
)

fig6.add_trace(
    go.Scatter(
        x=minutes_sentiment_date["Date"],
        y=minutes_sentiment_date["Policy Rate"],
        name="Policy Rate",
    ),
    secondary_y=True,
)

fig6.update_layout(title_text=f"Sentiment Score vs. Policy Rate (minutes information)")

fig6.update_xaxes(title_text="Date")

fig6.update_yaxes(title_text="Sentiment Score of the Minutes", secondary_y=False)
fig6.update_yaxes(title_text="Policy Rate", secondary_y=True)

fig6.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="40f438ab-7b00-4352-afa6-1b9d3f981291" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("40f438ab-7b00-4352-afa6-1b9d3f981291")) {                    Plotly.newPlot(                        "40f438ab-7b00-4352-afa6-1b9d3f981291",                        [{"name":"Sentiment Score of the Minutes'","x":["2024-11-07T00:00:00","2024-09-18T00:00:00","2024-07-31T00:00:00","2024-06-12T00:00:00","2024-05-01T00:00:00","2024-03-20T00:00:00","2024-01-31T00:00:00","2023-12-13T00:00:00","2023-11-01T00:00:00","2023-09-20T00:00:00","2023-07-26T00:00:00","2023-06-14T00:00:00","2023-05-03T00:00:00","2023-03-22T00:00:00","2023-02-01T00:00:00","2022-12-14T00:00:00","2022-11-02T00:00:00","2022-09-21T00:00:00","2022-07-27T00:00:00","2022-06-15T00:00:00","2022-05-04T00:00:00","2022-03-16T00:00:00","2022-01-26T00:00:00","2021-12-15T00:00:00","2021-11-03T00:00:00","2021-09-22T00:00:00","2021-07-28T00:00:00","2021-06-16T00:00:00","2021-04-28T00:00:00","2021-03-17T00:00:00","2021-01-27T00:00:00","2020-12-16T00:00:00","2020-11-05T00:00:00","2020-09-16T00:00:00","2020-07-29T00:00:00","2020-06-10T00:00:00","2020-04-29T00:00:00","2020-03-15T00:00:00","2020-03-03T00:00:00","2020-01-29T00:00:00","2019-12-11T00:00:00","2019-10-30T00:00:00","2019-10-04T00:00:00","2019-09-18T00:00:00","2019-07-31T00:00:00","2019-06-19T00:00:00","2019-05-01T00:00:00","2019-03-20T00:00:00","2019-01-30T00:00:00","2018-12-19T00:00:00","2018-11-08T00:00:00","2018-09-26T00:00:00","2018-08-01T00:00:00","2018-06-13T00:00:00","2018-05-02T00:00:00","2018-03-21T00:00:00","2018-01-31T00:00:00","2017-12-13T00:00:00","2017-11-01T00:00:00","2017-09-20T00:00:00","2017-07-26T00:00:00","2017-06-14T00:00:00","2017-05-03T00:00:00","2017-03-15T00:00:00","2017-02-01T00:00:00","2016-12-14T00:00:00","2016-11-02T00:00:00","2016-09-21T00:00:00","2016-07-27T00:00:00","2016-06-15T00:00:00","2016-04-27T00:00:00","2016-03-16T00:00:00","2016-01-27T00:00:00","2015-12-16T00:00:00","2015-10-28T00:00:00","2015-09-17T00:00:00","2015-07-29T00:00:00","2015-06-17T00:00:00","2015-04-29T00:00:00","2015-03-18T00:00:00","2015-01-28T00:00:00","2014-12-17T00:00:00","2014-10-29T00:00:00","2014-09-17T00:00:00","2014-07-30T00:00:00","2014-06-18T00:00:00","2014-04-30T00:00:00","2014-03-19T00:00:00","2014-03-04T00:00:00","2014-01-29T00:00:00","2013-12-18T00:00:00","2013-10-30T00:00:00","2013-10-16T00:00:00","2013-09-18T00:00:00","2013-07-31T00:00:00","2013-06-19T00:00:00","2013-05-01T00:00:00","2013-03-20T00:00:00","2013-01-30T00:00:00","2012-12-12T00:00:00","2012-10-24T00:00:00","2012-09-13T00:00:00","2012-08-01T00:00:00","2012-06-20T00:00:00","2012-04-25T00:00:00","2012-03-13T00:00:00","2012-01-25T00:00:00","2011-12-13T00:00:00","2011-11-28T00:00:00","2011-11-02T00:00:00","2011-09-21T00:00:00","2011-08-09T00:00:00","2011-08-01T00:00:00","2011-06-22T00:00:00","2011-04-27T00:00:00","2011-03-15T00:00:00","2011-01-26T00:00:00","2010-12-14T00:00:00","2010-11-03T00:00:00","2010-10-15T00:00:00","2010-09-21T00:00:00","2010-08-10T00:00:00","2010-06-23T00:00:00","2010-05-09T00:00:00","2010-04-28T00:00:00","2010-03-16T00:00:00","2010-01-27T00:00:00","2009-12-15T00:00:00","2009-11-04T00:00:00","2009-09-22T00:00:00","2009-08-11T00:00:00","2009-06-24T00:00:00","2009-06-03T00:00:00","2009-04-29T00:00:00","2009-03-17T00:00:00","2009-02-07T00:00:00","2009-01-28T00:00:00","2009-01-16T00:00:00","2008-12-16T00:00:00","2008-10-29T00:00:00","2008-10-07T00:00:00","2008-09-29T00:00:00","2008-09-16T00:00:00","2008-08-08T00:00:00","2008-07-24T00:00:00","2008-06-25T00:00:00","2008-04-30T00:00:00","2008-03-18T00:00:00","2008-03-10T00:00:00","2008-01-30T00:00:00","2008-01-21T00:00:00","2008-01-09T00:00:00","2007-12-11T00:00:00","2007-12-06T00:00:00","2007-10-31T00:00:00","2007-09-18T00:00:00","2007-08-16T00:00:00","2007-08-10T00:00:00","2007-08-07T00:00:00","2007-06-28T00:00:00","2007-05-09T00:00:00","2007-03-21T00:00:00","2007-01-31T00:00:00","2006-12-12T00:00:00","2006-10-25T00:00:00","2006-09-20T00:00:00","2006-08-08T00:00:00","2006-06-29T00:00:00","2006-05-10T00:00:00","2006-03-28T00:00:00","2006-01-31T00:00:00","2005-12-13T00:00:00","2005-11-01T00:00:00","2005-09-20T00:00:00","2005-08-09T00:00:00","2005-06-30T00:00:00","2005-05-03T00:00:00","2005-03-22T00:00:00","2005-02-02T00:00:00","2004-12-14T00:00:00","2004-11-10T00:00:00","2004-09-21T00:00:00","2004-08-10T00:00:00","2004-06-30T00:00:00","2004-05-04T00:00:00","2004-03-16T00:00:00","2004-01-28T00:00:00","2003-12-09T00:00:00","2003-10-28T00:00:00","2003-09-16T00:00:00","2003-09-15T00:00:00","2003-08-12T00:00:00","2003-06-25T00:00:00","2003-05-06T00:00:00","2003-04-16T00:00:00","2003-04-08T00:00:00","2003-04-01T00:00:00","2003-03-25T00:00:00","2003-03-18T00:00:00","2003-01-29T00:00:00","2002-12-10T00:00:00","2002-11-06T00:00:00","2002-09-24T00:00:00","2002-08-13T00:00:00","2002-06-26T00:00:00","2002-05-07T00:00:00","2002-03-19T00:00:00","2002-01-30T00:00:00","2001-12-11T00:00:00","2001-11-06T00:00:00","2001-10-02T00:00:00","2001-09-17T00:00:00","2001-09-13T00:00:00","2001-08-21T00:00:00","2001-06-27T00:00:00","2001-05-15T00:00:00","2001-04-18T00:00:00","2001-04-11T00:00:00","2001-03-20T00:00:00","2001-01-31T00:00:00","2001-01-03T00:00:00","2000-12-19T00:00:00","2000-11-15T00:00:00","2000-10-03T00:00:00","2000-08-22T00:00:00","2000-06-28T00:00:00","2000-05-16T00:00:00","2000-03-21T00:00:00","2000-02-02T00:00:00"],"y":[-250,-336,-163,-57,-115,-222,33,-443,-182,-214,-497,-499,-1086,-372,-567,-560,-661,-437,-688,-435,-227,-126,96,-107,-6,-30,412,230,185,355,202,28,-218,68,-281,-563,-742,-618,-618,59,130,-496,-496,-381,-104,-237,-150,-342,-396,-506,-67,52,24,98,42,-220,-68,-172,-304,-383,-233,-524,-622,-131,-221,-345,-103,-688,-478,-701,-230,-408,-949,-492,-490,-713,-223,-185,-475,-356,-863,-396,-471,-306,-402,-681,-395,-1000,-1000,-869,-728,-579,-579,-454,-347,-318,-398,-379,-443,-671,-336,-457,-379,-873,-656,-237,-708,-437,-437,-323,-610,-582,-582,-590,-193,-96,-229,-162,-352,-352,-246,-308,-198,-198,-112,-170,-350,-284,-240,-173,-277,-543,-543,-157,-356,-356,-607,-607,-1400,-817,-817,-817,-410,-287,-287,-256,-387,-473,-473,-369,-369,-369,-285,-285,-283,-175,-175,-175,-127,-145,-174,-135,62,-85,-219,-70,-47,-50,147,43,-3,114,11,11,34,55,81,117,523,63,20,-25,-59,191,100,37,95,93,91,100,14,14,-167,-148,-67,-67,-67,-67,-67,-31,-39,-134,23,-66,34,-67,80,-166,-384,-446,-469,-413,-413,-413,-627,-422,-352,-352,-352,-302,-292,-292,-95,116,-40,-65,26,36,88],"type":"scatter","xaxis":"x","yaxis":"y"},{"name":"Policy Rate","x":["2024-11-07T00:00:00","2024-09-18T00:00:00","2024-07-31T00:00:00","2024-06-12T00:00:00","2024-05-01T00:00:00","2024-03-20T00:00:00","2024-01-31T00:00:00","2023-12-13T00:00:00","2023-11-01T00:00:00","2023-09-20T00:00:00","2023-07-26T00:00:00","2023-06-14T00:00:00","2023-05-03T00:00:00","2023-03-22T00:00:00","2023-02-01T00:00:00","2022-12-14T00:00:00","2022-11-02T00:00:00","2022-09-21T00:00:00","2022-07-27T00:00:00","2022-06-15T00:00:00","2022-05-04T00:00:00","2022-03-16T00:00:00","2022-01-26T00:00:00","2021-12-15T00:00:00","2021-11-03T00:00:00","2021-09-22T00:00:00","2021-07-28T00:00:00","2021-06-16T00:00:00","2021-04-28T00:00:00","2021-03-17T00:00:00","2021-01-27T00:00:00","2020-12-16T00:00:00","2020-11-05T00:00:00","2020-09-16T00:00:00","2020-07-29T00:00:00","2020-06-10T00:00:00","2020-04-29T00:00:00","2020-03-15T00:00:00","2020-03-03T00:00:00","2020-01-29T00:00:00","2019-12-11T00:00:00","2019-10-30T00:00:00","2019-10-04T00:00:00","2019-09-18T00:00:00","2019-07-31T00:00:00","2019-06-19T00:00:00","2019-05-01T00:00:00","2019-03-20T00:00:00","2019-01-30T00:00:00","2018-12-19T00:00:00","2018-11-08T00:00:00","2018-09-26T00:00:00","2018-08-01T00:00:00","2018-06-13T00:00:00","2018-05-02T00:00:00","2018-03-21T00:00:00","2018-01-31T00:00:00","2017-12-13T00:00:00","2017-11-01T00:00:00","2017-09-20T00:00:00","2017-07-26T00:00:00","2017-06-14T00:00:00","2017-05-03T00:00:00","2017-03-15T00:00:00","2017-02-01T00:00:00","2016-12-14T00:00:00","2016-11-02T00:00:00","2016-09-21T00:00:00","2016-07-27T00:00:00","2016-06-15T00:00:00","2016-04-27T00:00:00","2016-03-16T00:00:00","2016-01-27T00:00:00","2015-12-16T00:00:00","2015-10-28T00:00:00","2015-09-17T00:00:00","2015-07-29T00:00:00","2015-06-17T00:00:00","2015-04-29T00:00:00","2015-03-18T00:00:00","2015-01-28T00:00:00","2014-12-17T00:00:00","2014-10-29T00:00:00","2014-09-17T00:00:00","2014-07-30T00:00:00","2014-06-18T00:00:00","2014-04-30T00:00:00","2014-03-19T00:00:00","2014-03-04T00:00:00","2014-01-29T00:00:00","2013-12-18T00:00:00","2013-10-30T00:00:00","2013-10-16T00:00:00","2013-09-18T00:00:00","2013-07-31T00:00:00","2013-06-19T00:00:00","2013-05-01T00:00:00","2013-03-20T00:00:00","2013-01-30T00:00:00","2012-12-12T00:00:00","2012-10-24T00:00:00","2012-09-13T00:00:00","2012-08-01T00:00:00","2012-06-20T00:00:00","2012-04-25T00:00:00","2012-03-13T00:00:00","2012-01-25T00:00:00","2011-12-13T00:00:00","2011-11-28T00:00:00","2011-11-02T00:00:00","2011-09-21T00:00:00","2011-08-09T00:00:00","2011-08-01T00:00:00","2011-06-22T00:00:00","2011-04-27T00:00:00","2011-03-15T00:00:00","2011-01-26T00:00:00","2010-12-14T00:00:00","2010-11-03T00:00:00","2010-10-15T00:00:00","2010-09-21T00:00:00","2010-08-10T00:00:00","2010-06-23T00:00:00","2010-05-09T00:00:00","2010-04-28T00:00:00","2010-03-16T00:00:00","2010-01-27T00:00:00","2009-12-15T00:00:00","2009-11-04T00:00:00","2009-09-22T00:00:00","2009-08-11T00:00:00","2009-06-24T00:00:00","2009-06-03T00:00:00","2009-04-29T00:00:00","2009-03-17T00:00:00","2009-02-07T00:00:00","2009-01-28T00:00:00","2009-01-16T00:00:00","2008-12-16T00:00:00","2008-10-29T00:00:00","2008-10-07T00:00:00","2008-09-29T00:00:00","2008-09-16T00:00:00","2008-08-08T00:00:00","2008-07-24T00:00:00","2008-06-25T00:00:00","2008-04-30T00:00:00","2008-03-18T00:00:00","2008-03-10T00:00:00","2008-01-30T00:00:00","2008-01-21T00:00:00","2008-01-09T00:00:00","2007-12-11T00:00:00","2007-12-06T00:00:00","2007-10-31T00:00:00","2007-09-18T00:00:00","2007-08-16T00:00:00","2007-08-10T00:00:00","2007-08-07T00:00:00","2007-06-28T00:00:00","2007-05-09T00:00:00","2007-03-21T00:00:00","2007-01-31T00:00:00","2006-12-12T00:00:00","2006-10-25T00:00:00","2006-09-20T00:00:00","2006-08-08T00:00:00","2006-06-29T00:00:00","2006-05-10T00:00:00","2006-03-28T00:00:00","2006-01-31T00:00:00","2005-12-13T00:00:00","2005-11-01T00:00:00","2005-09-20T00:00:00","2005-08-09T00:00:00","2005-06-30T00:00:00","2005-05-03T00:00:00","2005-03-22T00:00:00","2005-02-02T00:00:00","2004-12-14T00:00:00","2004-11-10T00:00:00","2004-09-21T00:00:00","2004-08-10T00:00:00","2004-06-30T00:00:00","2004-05-04T00:00:00","2004-03-16T00:00:00","2004-01-28T00:00:00","2003-12-09T00:00:00","2003-10-28T00:00:00","2003-09-16T00:00:00","2003-09-15T00:00:00","2003-08-12T00:00:00","2003-06-25T00:00:00","2003-05-06T00:00:00","2003-04-16T00:00:00","2003-04-08T00:00:00","2003-04-01T00:00:00","2003-03-25T00:00:00","2003-03-18T00:00:00","2003-01-29T00:00:00","2002-12-10T00:00:00","2002-11-06T00:00:00","2002-09-24T00:00:00","2002-08-13T00:00:00","2002-06-26T00:00:00","2002-05-07T00:00:00","2002-03-19T00:00:00","2002-01-30T00:00:00","2001-12-11T00:00:00","2001-11-06T00:00:00","2001-10-02T00:00:00","2001-09-17T00:00:00","2001-09-13T00:00:00","2001-08-21T00:00:00","2001-06-27T00:00:00","2001-05-15T00:00:00","2001-04-18T00:00:00","2001-04-11T00:00:00","2001-03-20T00:00:00","2001-01-31T00:00:00","2001-01-03T00:00:00","2000-12-19T00:00:00","2000-11-15T00:00:00","2000-10-03T00:00:00","2000-08-22T00:00:00","2000-06-28T00:00:00","2000-05-16T00:00:00","2000-03-21T00:00:00","2000-02-02T00:00:00"],"y":[4.75,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.0,5.0,4.75,4.5,4.25,3.75,3.0,2.25,1.5,0.75,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.5,1.5,1.5,1.75,1.75,2.0,2.25,2.25,2.25,2.25,2.25,2.0,2.0,1.75,1.75,1.5,1.5,1.25,1.25,1.0,1.0,1.0,1.0,0.75,0.75,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.25,3.0,3.0,4.25,4.25,4.25,4.5,4.5,4.75,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.0,4.75,4.5,4.25,4.0,3.75,3.5,3.25,3.0,2.75,2.5,2.25,2.0,1.75,1.5,1.25,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.75,1.75,1.75,1.75,1.75,1.75,1.75,2.0,2.5,3.0,3.5,3.5,3.75,4.0,4.5,5.0,5.0,5.5,6.0,6.5,6.5,6.5,6.5,6.5,6.5,6.0,5.75],"type":"scatter","xaxis":"x","yaxis":"y2"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.94],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Sentiment Score of the Minutes"}},"yaxis2":{"anchor":"x","overlaying":"y","side":"right","title":{"text":"Policy Rate"}},"title":{"text":"Sentiment Score vs. Policy Rate (minutes information)"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('40f438ab-7b00-4352-afa6-1b9d3f981291');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


#### 3.3.3 **Conclusion**

The visualizations of this part **approve** the original hypothesis and show that **an increase in the use of positive language is associated with a reduced likelihood of an imminent rate cut or increased likelihood of rate rise.** <br>

Minutes' texts display a more illustrative result than statements' result, the findings are the following:
1. The positive trend of the text can foreshadow or appear concurrently with the policy action of rate rise
2. The negative trend of the text can foreshadow or appear concurrently with the policy action of rate cut

Conjecture: This may imply that the Fed would consider the economic resilient when taking tightening actions. The Fed may move to take monetary easing when they perceive the negative factor within the market.


# **PART 4**: Conclusion and Policy Implication

## 4.1 Conclusion to the overall project

**Conclusion:** <br>
After three rounds of test, this project reaches the following conclusion regarding the hypotheses:
1. An increase in the use of positive language is associated with a reduced likelihood of an imminent rate cut or increased likelihood of rate hikes
2. significant shift in the language used to describe the economic outlook signals a high probability of changes in policy direct in the incoming FOMC meetings.

The confirmation of the hypotheses reflects that the policy information from the FOMC statements is meaningful and useful to project the policy intentions and potential directions.



**Policy Implication**: <br>
From a policy analyst’s viewpoint, the findings on the FOMC’s communication underline the importance of monitoring and evaluating the language used in public releases. Analysts should focus on:
1. Market Sentiment Analysis: Utilize sentiment analysis and computational tools to identify patterns in tone shifts (e.g., from optimism to caution) and correlate these with subsequent monetary policy decisions and market reactions.
2. Predictive Modeling: Develop predictive models to analyze the potential impacts of language changes in FOMC statements on financial markets, helping to forecast the implications of various policy scenarios.
3. Policy Communication Assessment: Regularly assess how clearly the FOMC’s language conveys its economic outlook and policy intentions. This can help to improve transparency and reduce market speculation.

From a policymaker’s perspective, the findings stress the strategic importance of language in achieving monetary policy objectives. Policymakers should consider:
1. Strategic Communication: Craft language carefully to send deliberate signals to the market, ensuring alignment with monetary policy objectives. Policymakers should be mindful of how subtle tone changes can influence market sentiment and expectations.
2. Gradual Signaling: Use incremental shifts in language to prepare markets for policy changes, reducing the likelihood of sudden disruptions while maintaining credibility.

By adopting these tailored approaches, analysts can enhance their evaluation frameworks, while policymakers can strengthen their communication strategies to ensure alignment with macroeconomic goals and market stability.

**Future Improvement**: <br>
1. Advanced Text Analysis:
  - Apply more sophisticated natural language processing (NLP) techniques, such as transformer-based models (e.g., BERT or GPT), to detect nuanced language shifts.
  - Incorporate semantic analysis to better understand implied policy intentions.
2. Policy Simulation Tool:
  - Develop an interactive simulation to test alternative FOMC statements and predict potential market responses

# Part 5: reference and citation
citation list:
1. Board of Governors of the Federal Reserve System (US), Federal Funds Target Rate (DISCONTINUED) [DFEDTAR], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DFEDTAR, December 16, 2024.
2. Board of Governors of the Federal Reserve System (US), Federal Funds Target Range - Lower Limit [DFEDTARL], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DFEDTARL, December 16, 2024.



```python

```
