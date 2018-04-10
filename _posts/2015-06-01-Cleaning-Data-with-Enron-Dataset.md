---
layout: post
title: "Data Cleaning Tutorial with the Enron Dataset"
description: "Tools, techniques, and tips to clean data using the enron dataset"
date: 2015-06-01
tags: [python, data cleaning, pandas, enron]
---
In this notebook I'm going to look at the basics of cleaning data with Python. I will be using a dataset of people involved in the Enron scandel. I first saw this data set in the Intro to Machine Learning class at Udacity.<!--more-->


```python
# Basic imports that we'll use
import pandas as pd
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
```

# Loading the data

The first step is to load the data. It's saved in as a pickle file. Pickle files are used to serialize and de-serialize files. Serializing is a process of converting a Python data object, like a list or dictionary, into a stream of characters.


```python
path = 'C:/Users/HMISYS/Documents/GitHub/jss367.github.io/datasets/Enron/'
file = 'final_project_dataset.pkl'
with open(path + file, 'rb') as f:
    enron_data = pickle.load(f)
```

# Exploring the data

Now let's look at the data. First we'll see what type it is.


```python
print("The dataset is a", type(enron_data))
```

    The dataset is a <class 'dict'>
    

OK, we have a dictionary. Let's see what different entities we have in the dataset


```python
print("There are {} entities in the dataset.\n".format(len(enron_data)))
print(enron_data.keys())
```

    There are 146 entities in the dataset.
    
    dict_keys(['METTS MARK', 'BAXTER JOHN C', 'ELLIOTT STEVEN', 'CORDES WILLIAM R', 'HANNON KEVIN P', 'MORDAUNT KRISTINA M', 'MEYER ROCKFORD G', 'MCMAHON JEFFREY', 'HAEDICKE MARK E', 'PIPER GREGORY F', 'HUMPHREY GENE E', 'NOLES JAMES L', 'BLACHMAN JEREMY M', 'SUNDE MARTIN', 'GIBBS DANA R', 'LOWRY CHARLES P', 'COLWELL WESLEY', 'MULLER MARK S', 'JACKSON CHARLENE R', 'WESTFAHL RICHARD K', 'WALTERS GARETH W', 'WALLS JR ROBERT H', 'KITCHEN LOUISE', 'CHAN RONNIE', 'BELFER ROBERT', 'SHANKMAN JEFFREY A', 'WODRASKA JOHN', 'BERGSIEKER RICHARD P', 'URQUHART JOHN A', 'BIBI PHILIPPE A', 'RIEKER PAULA H', 'WHALEY DAVID A', 'BECK SALLY W', 'HAUG DAVID L', 'ECHOLS JOHN B', 'MENDELSOHN JOHN', 'HICKERSON GARY J', 'CLINE KENNETH W', 'LEWIS RICHARD', 'HAYES ROBERT E', 'KOPPER MICHAEL J', 'LEFF DANIEL P', 'LAVORATO JOHN J', 'BERBERIAN DAVID', 'DETMERING TIMOTHY J', 'WAKEHAM JOHN', 'POWERS WILLIAM', 'GOLD JOSEPH', 'BANNANTINE JAMES M', 'DUNCAN JOHN H', 'SHAPIRO RICHARD S', 'SHERRIFF JOHN R', 'SHELBY REX', 'LEMAISTRE CHARLES', 'DEFFNER JOSEPH M', 'KISHKILL JOSEPH G', 'WHALLEY LAWRENCE G', 'MCCONNELL MICHAEL S', 'PIRO JIM', 'DELAINEY DAVID W', 'SULLIVAN-SHAKLOVITZ COLLEEN', 'WROBEL BRUCE', 'LINDHOLM TOD A', 'MEYER JEROME J', 'LAY KENNETH L', 'BUTTS ROBERT H', 'OLSON CINDY K', 'MCDONALD REBECCA', 'CUMBERLAND MICHAEL S', 'GAHN ROBERT S', 'BADUM JAMES P', 'HERMANN ROBERT J', 'FALLON JAMES B', 'GATHMANN WILLIAM D', 'HORTON STANLEY C', 'BOWEN JR RAYMOND M', 'GILLIS JOHN', 'FITZGERALD JAY L', 'MORAN MICHAEL P', 'REDMOND BRIAN L', 'BAZELIDES PHILIP J', 'BELDEN TIMOTHY N', 'DIMICHELE RICHARD G', 'DURAN WILLIAM D', 'THORN TERENCE H', 'FASTOW ANDREW S', 'FOY JOE', 'CALGER CHRISTOPHER F', 'RICE KENNETH D', 'KAMINSKI WINCENTY J', 'LOCKHART EUGENE E', 'COX DAVID', 'OVERDYKE JR JERE C', 'PEREIRA PAULO V. FERRAZ', 'STABLER FRANK', 'SKILLING JEFFREY K', 'BLAKE JR. NORMAN P', 'SHERRICK JEFFREY B', 'PRENTICE JAMES', 'GRAY RODNEY', 'THE TRAVEL AGENCY IN THE PARK', 'UMANOFF ADAM S', 'KEAN STEVEN J', 'TOTAL', 'FOWLER PEGGY', 'WASAFF GEORGE', 'WHITE JR THOMAS E', 'CHRISTODOULOU DIOMEDES', 'ALLEN PHILLIP K', 'SHARP VICTORIA T', 'JAEDICKE ROBERT', 'WINOKUR JR. HERBERT S', 'BROWN MICHAEL', 'MCCLELLAN GEORGE', 'HUGHES JAMES A', 'REYNOLDS LAWRENCE', 'PICKERING MARK R', 'BHATNAGAR SANJAY', 'CARTER REBECCA C', 'BUCHANAN HAROLD G', 'YEAP SOON', 'MURRAY JULIA H', 'GARLAND C KEVIN', 'DODSON KEITH', 'YEAGER F SCOTT', 'HIRKO JOSEPH', 'DIETRICH JANET R', 'DERRICK JR. JAMES V', 'FREVERT MARK A', 'PAI LOU L', 'HAYSLETT RODERICK J', 'BAY FRANKLIN R', 'MCCARTY DANNY J', 'FUGH JOHN L', 'SCRIMSHAW MATTHEW', 'KOENIG MARK E', 'SAVAGE FRANK', 'IZZO LAWRENCE L', 'TILNEY ELIZABETH A', 'MARTIN AMANDA K', 'BUY RICHARD B', 'GRAMM WENDY L', 'CAUSEY RICHARD A', 'TAYLOR MITCHELL S', 'DONAHUE JR JEFFREY M', 'GLISAN JR BEN F'])
    

The keys are different people who worked at Enron. I see several names familiar from the Enron scandel, including Kennth Lay, Jeffry Skilling, Andrew Fastow, and Cliff Baxter (who's listed as John C Baxter). There's also an entity named "The Travel Agency in the Park". From footnote j in the original document (http://news.findlaw.com/hdocs/docs/enron/enron61702insiderpay.pdf), this business was co-owned by the sister of Enron's former Chairman. It may be of interest to investigators, but it will mess up the machine learning algorithms as it's not an employee, so I will remove it.

Now let's see what information we have about each person. We'll start with Ken Lay.


```python
enron_data['LAY KENNETH L']
```




    {'bonus': 7000000,
     'deferral_payments': 202911,
     'deferred_income': -300000,
     'director_fees': 'NaN',
     'email_address': 'kenneth.lay@enron.com',
     'exercised_stock_options': 34348384,
     'expenses': 99832,
     'from_messages': 36,
     'from_poi_to_this_person': 123,
     'from_this_person_to_poi': 16,
     'loan_advances': 81525000,
     'long_term_incentive': 3600000,
     'other': 10359729,
     'poi': True,
     'restricted_stock': 14761694,
     'restricted_stock_deferred': 'NaN',
     'salary': 1072321,
     'shared_receipt_with_poi': 2411,
     'to_messages': 4273,
     'total_payments': 103559793,
     'total_stock_value': 49110078}



There are checksums built into the dataset, like total_payments and total_stock_value. We should be able to calculate these from the other values to double check the values.


We can also query for a specific value, like this


```python
print("Jeff Skilling's total payments were ${:,.0f}.".format(enron_data['SKILLING JEFFREY K']['total_payments']))
```

    Jeff Skilling's total payments were $8,682,716.
    

Before we go any further, we'll put the data into a pandas dataframe to make it easier to work with. We'll make a dataframe with all the values and a separate 


```python
# The keys of the dictionary are the people, so we'll want them to be the rows of the dataframe
# To do this, pass orient='index'
df = pd.DataFrame.from_dict(enron_data, orient='index')
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>email_address</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALLEN PHILLIP K</th>
      <td>201955</td>
      <td>2902</td>
      <td>2869717</td>
      <td>4484442</td>
      <td>NaN</td>
      <td>4175000</td>
      <td>phillip.allen@enron.com</td>
      <td>-126027</td>
      <td>-3081055</td>
      <td>1729541</td>
      <td>...</td>
      <td>47</td>
      <td>1729541</td>
      <td>2195</td>
      <td>152</td>
      <td>65</td>
      <td>False</td>
      <td>304805</td>
      <td>1407</td>
      <td>126027</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BADUM JAMES P</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>178980</td>
      <td>182466</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>257817</td>
      <td>...</td>
      <td>NaN</td>
      <td>257817</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BANNANTINE JAMES M</th>
      <td>477</td>
      <td>566</td>
      <td>NaN</td>
      <td>916197</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>james.bannantine@enron.com</td>
      <td>-560222</td>
      <td>-5104</td>
      <td>5243487</td>
      <td>...</td>
      <td>39</td>
      <td>4046157</td>
      <td>29</td>
      <td>864523</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>465</td>
      <td>1757552</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>267102</td>
      <td>NaN</td>
      <td>1295738</td>
      <td>5634343</td>
      <td>NaN</td>
      <td>1200000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1386055</td>
      <td>10623258</td>
      <td>...</td>
      <td>NaN</td>
      <td>6680544</td>
      <td>NaN</td>
      <td>2660303</td>
      <td>NaN</td>
      <td>False</td>
      <td>1586055</td>
      <td>NaN</td>
      <td>3942714</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>239671</td>
      <td>NaN</td>
      <td>260455</td>
      <td>827696</td>
      <td>NaN</td>
      <td>400000</td>
      <td>frank.bay@enron.com</td>
      <td>-82782</td>
      <td>-201641</td>
      <td>63014</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>145796</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



We can already see missing values, and it looks like they're entered as NaN, which Python will see as a string and not recognized as a null value. We can do a quick replacement on those



```python
df = df.replace('NaN', np.nan)
```

Pandas makes summarizing the data simple


```python
df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>expenses</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.500000e+01</td>
      <td>86.000000</td>
      <td>3.900000e+01</td>
      <td>1.250000e+02</td>
      <td>4.000000e+00</td>
      <td>8.200000e+01</td>
      <td>1.800000e+01</td>
      <td>4.900000e+01</td>
      <td>1.260000e+02</td>
      <td>9.500000e+01</td>
      <td>86.000000</td>
      <td>1.020000e+02</td>
      <td>86.000000</td>
      <td>9.300000e+01</td>
      <td>86.000000</td>
      <td>6.600000e+01</td>
      <td>86.000000</td>
      <td>1.100000e+02</td>
      <td>1.700000e+01</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.621943e+05</td>
      <td>2073.860465</td>
      <td>1.642674e+06</td>
      <td>5.081526e+06</td>
      <td>4.196250e+07</td>
      <td>2.374235e+06</td>
      <td>1.664106e+05</td>
      <td>-1.140475e+06</td>
      <td>6.773957e+06</td>
      <td>1.087289e+05</td>
      <td>64.895349</td>
      <td>5.987054e+06</td>
      <td>608.790698</td>
      <td>9.190650e+05</td>
      <td>41.232558</td>
      <td>1.470361e+06</td>
      <td>1176.465116</td>
      <td>2.321741e+06</td>
      <td>1.668049e+05</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.716369e+06</td>
      <td>2582.700981</td>
      <td>5.161930e+06</td>
      <td>2.906172e+07</td>
      <td>4.708321e+07</td>
      <td>1.071333e+07</td>
      <td>4.201494e+06</td>
      <td>4.025406e+06</td>
      <td>3.895777e+07</td>
      <td>5.335348e+05</td>
      <td>86.979244</td>
      <td>3.106201e+07</td>
      <td>1841.033949</td>
      <td>4.589253e+06</td>
      <td>100.073111</td>
      <td>5.942759e+06</td>
      <td>1178.317641</td>
      <td>1.251828e+07</td>
      <td>3.198914e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.770000e+02</td>
      <td>57.000000</td>
      <td>-1.025000e+05</td>
      <td>1.480000e+02</td>
      <td>4.000000e+05</td>
      <td>7.000000e+04</td>
      <td>-7.576788e+06</td>
      <td>-2.799289e+07</td>
      <td>-4.409300e+04</td>
      <td>1.480000e+02</td>
      <td>0.000000</td>
      <td>3.285000e+03</td>
      <td>12.000000</td>
      <td>2.000000e+00</td>
      <td>0.000000</td>
      <td>6.922300e+04</td>
      <td>2.000000</td>
      <td>-2.604490e+06</td>
      <td>3.285000e+03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.118160e+05</td>
      <td>541.250000</td>
      <td>8.157300e+04</td>
      <td>3.944750e+05</td>
      <td>1.600000e+06</td>
      <td>4.312500e+05</td>
      <td>-3.896218e+05</td>
      <td>-6.948620e+05</td>
      <td>4.945102e+05</td>
      <td>2.261400e+04</td>
      <td>10.000000</td>
      <td>5.278862e+05</td>
      <td>22.750000</td>
      <td>1.215000e+03</td>
      <td>1.000000</td>
      <td>2.812500e+05</td>
      <td>249.750000</td>
      <td>2.540180e+05</td>
      <td>9.878400e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.599960e+05</td>
      <td>1211.000000</td>
      <td>2.274490e+05</td>
      <td>1.101393e+06</td>
      <td>4.176250e+07</td>
      <td>7.693750e+05</td>
      <td>-1.469750e+05</td>
      <td>-1.597920e+05</td>
      <td>1.102872e+06</td>
      <td>4.695000e+04</td>
      <td>35.000000</td>
      <td>1.310814e+06</td>
      <td>41.000000</td>
      <td>5.238200e+04</td>
      <td>8.000000</td>
      <td>4.420350e+05</td>
      <td>740.500000</td>
      <td>4.517400e+05</td>
      <td>1.085790e+05</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.121170e+05</td>
      <td>2634.750000</td>
      <td>1.002672e+06</td>
      <td>2.093263e+06</td>
      <td>8.212500e+07</td>
      <td>1.200000e+06</td>
      <td>-7.500975e+04</td>
      <td>-3.834600e+04</td>
      <td>2.949847e+06</td>
      <td>7.995250e+04</td>
      <td>72.250000</td>
      <td>2.547724e+06</td>
      <td>145.500000</td>
      <td>3.620960e+05</td>
      <td>24.750000</td>
      <td>9.386720e+05</td>
      <td>1888.250000</td>
      <td>1.002370e+06</td>
      <td>1.137840e+05</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.670423e+07</td>
      <td>15149.000000</td>
      <td>3.208340e+07</td>
      <td>3.098866e+08</td>
      <td>8.392500e+07</td>
      <td>9.734362e+07</td>
      <td>1.545629e+07</td>
      <td>-8.330000e+02</td>
      <td>4.345095e+08</td>
      <td>5.235198e+06</td>
      <td>528.000000</td>
      <td>3.117640e+08</td>
      <td>14368.000000</td>
      <td>4.266759e+07</td>
      <td>609.000000</td>
      <td>4.852193e+07</td>
      <td>5521.000000</td>
      <td>1.303223e+08</td>
      <td>1.398517e+06</td>
    </tr>
  </tbody>
</table>
</div>



The highest salary was \$26 million, and the highest total payments was \$309 mission. The total payments seems too high, even for Enron, so we'll have to look into that.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 146 entries, ALLEN PHILLIP K to YEAP SOON
    Data columns (total 21 columns):
    salary                       95 non-null float64
    to_messages                  86 non-null float64
    deferral_payments            39 non-null float64
    total_payments               125 non-null float64
    loan_advances                4 non-null float64
    bonus                        82 non-null float64
    email_address                111 non-null object
    restricted_stock_deferred    18 non-null float64
    deferred_income              49 non-null float64
    total_stock_value            126 non-null float64
    expenses                     95 non-null float64
    from_poi_to_this_person      86 non-null float64
    exercised_stock_options      102 non-null float64
    from_messages                86 non-null float64
    other                        93 non-null float64
    from_this_person_to_poi      86 non-null float64
    poi                          146 non-null bool
    long_term_incentive          66 non-null float64
    shared_receipt_with_poi      86 non-null float64
    restricted_stock             110 non-null float64
    director_fees                17 non-null float64
    dtypes: bool(1), float64(19), object(1)
    memory usage: 24.1+ KB
    

# Cleaning the data

It looks like we have lots of integers and strings and a single column of booleans. The booleans column "poi" indicates whether the persion is a "Person Of Interest". This is the column we'll be trying to predict using the other data. Every person in the dataframe is marked as either a poi or not. Unfortunately, this is not the case with the other columns. From the first fives rows we can tell that the other columns have lots of missing data. Let's look at how bad it is.


```python
# Print the number of missing values
num_missing_values = df.isnull().sum()
print(num_missing_values)
```

    salary                        51
    to_messages                   60
    deferral_payments            107
    total_payments                21
    loan_advances                142
    bonus                         64
    email_address                 35
    restricted_stock_deferred    128
    deferred_income               97
    total_stock_value             20
    expenses                      51
    from_poi_to_this_person       60
    exercised_stock_options       44
    from_messages                 60
    other                         53
    from_this_person_to_poi       60
    poi                            0
    long_term_incentive           80
    shared_receipt_with_poi       60
    restricted_stock              36
    director_fees                129
    dtype: int64
    

That's a lot, and it isn't easy to analyze a dataset with that many missing values. Let's graph it to see what we've got. Remember there are 146 different people in this dataset


```python
fig, ax = plt.subplots(figsize=(16, 10))
x = np.arange(0, len(num_missing_values))
matplotlib.rcParams.update({'font.size': 18})
plt.xticks(x, (df.columns), rotation='vertical')

# create the bars
bars = plt.bar(x, num_missing_values, align='center', linewidth=0)

ax.set_ylabel('Number of missing values')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# direct label each bar with Y axis values
for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())), 
                 ha='center', color='w', fontsize=16)
    
plt.show()
```


![png](2015-06-01-Cleaning-Data-with-Enron-Dataset_files/2015-06-01-Cleaning-Data-with-Enron-Dataset_27_0.png)


The most common missing values are loan_advances and director_fees. These are likely to be zero for most employees. Based on the columns that have the most missing values, the complete lack of zeros in the dataset, and the way it's presented in the spreadsheet, I think we can say that all NaN values should actually be zero. Let's make the change.


```python
df = df.fillna(0)
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>email_address</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALLEN PHILLIP K</th>
      <td>201955.0</td>
      <td>2902.0</td>
      <td>2869717.0</td>
      <td>4484442.0</td>
      <td>0.0</td>
      <td>4175000.0</td>
      <td>phillip.allen@enron.com</td>
      <td>-126027.0</td>
      <td>-3081055.0</td>
      <td>1729541.0</td>
      <td>...</td>
      <td>47.0</td>
      <td>1729541.0</td>
      <td>2195.0</td>
      <td>152.0</td>
      <td>65.0</td>
      <td>False</td>
      <td>304805.0</td>
      <td>1407.0</td>
      <td>126027.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BADUM JAMES P</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>178980.0</td>
      <td>182466.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>257817.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>257817.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BANNANTINE JAMES M</th>
      <td>477.0</td>
      <td>566.0</td>
      <td>0.0</td>
      <td>916197.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>james.bannantine@enron.com</td>
      <td>-560222.0</td>
      <td>-5104.0</td>
      <td>5243487.0</td>
      <td>...</td>
      <td>39.0</td>
      <td>4046157.0</td>
      <td>29.0</td>
      <td>864523.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>465.0</td>
      <td>1757552.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>267102.0</td>
      <td>0.0</td>
      <td>1295738.0</td>
      <td>5634343.0</td>
      <td>0.0</td>
      <td>1200000.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>-1386055.0</td>
      <td>10623258.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>6680544.0</td>
      <td>0.0</td>
      <td>2660303.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>1586055.0</td>
      <td>0.0</td>
      <td>3942714.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>239671.0</td>
      <td>0.0</td>
      <td>260455.0</td>
      <td>827696.0</td>
      <td>0.0</td>
      <td>400000.0</td>
      <td>frank.bay@enron.com</td>
      <td>-82782.0</td>
      <td>-201641.0</td>
      <td>63014.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>69.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>145796.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Let's look at the checksum values to make sure everything checks out. By "checksums", I'm referring to values that are supposed to be the sum of other values in the table. They can be used to quickly find errors in the data. In this case, the total_payments field is supposed to be the sum of all the payment categories: salary, bonus, long_term_incentive, deferred_income, deferral_payments, loan_advances, other, expenses, and director_fees.

The original spreadsheet divides up the information into payments and stock value. We'll do that now and add a separate category for the email


```python
payment_categories = ['salary', 'bonus', 'long_term_incentive', 'deferred_income',
                      'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments']
stock_value_categories = ['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']
```

Now let's sum together all the payment categories (except total_payments) and compare it to the total payments. It should be the same value. We'll print out any rows that aren't the same


```python
# Let's look at the instances where the total we calculate is not equal to the total listed on the spreadsheet
df[df[payment_categories[:-1]].sum(axis='columns') != df['total_payments']][payment_categories]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BELFER ROBERT</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-102500.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3285.0</td>
      <td>102500.0</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>15456290.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's look at the instances where the total we calculate is not equal to the total listed on the spreadsheet
df[df[stock_value_categories[:-1]].sum(axis='columns') != df['total_stock_value']][stock_value_categories]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BELFER ROBERT</th>
      <td>3285.0</td>
      <td>0.0</td>
      <td>44093.0</td>
      <td>-44093.0</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>2604490.0</td>
      <td>-2604490.0</td>
      <td>15456290.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



On the original spreadsheet, 102,500 is listed in Robert Belfer's deferred income section, not in deferral payments. And 3,285 should be the expenses column, and director fees should be 102,500, and total payments should be 3,285. Looks like everything shifted one column to the right. We'll have to move it one to the left to fix it. The opposite happened to Sanjay Bhatnagar, so we'll have to move his columns to the right to fix them. Let's do that now


```python
df.loc['BELFER ROBERT']
```




    salary                            0
    to_messages                       0
    deferral_payments           -102500
    total_payments               102500
    loan_advances                     0
    bonus                             0
    email_address                     0
    restricted_stock_deferred     44093
    deferred_income                   0
    total_stock_value            -44093
    expenses                          0
    from_poi_to_this_person           0
    exercised_stock_options        3285
    from_messages                     0
    other                             0
    from_this_person_to_poi           0
    poi                           False
    long_term_incentive               0
    shared_receipt_with_poi           0
    restricted_stock                  0
    director_fees                  3285
    Name: BELFER ROBERT, dtype: object



Unfortunately, the order of the columns in the actual spreadsheet is different than the one in this dataset, so I can't use `pop` to push them all over one. I'll have to manually fix every incorrect value.


```python
df.loc[('BELFER ROBERT','deferral_payments')] = 0
df.loc[('BELFER ROBERT','total_payments')] = 3285
df.loc[('BELFER ROBERT','restricted_stock_deferred')] = -44093
df.loc[('BELFER ROBERT','deferred_income')] = -102500
df.loc[('BELFER ROBERT','total_stock_value')] = 0
df.loc[('BELFER ROBERT','expenses')] = 3285
df.loc[('BELFER ROBERT','exercised_stock_options')] = 0
df.loc[('BELFER ROBERT','restricted_stock')] = 44093
df.loc[('BELFER ROBERT','director_fees')] = 102500
```


```python
df.loc['BHATNAGAR SANJAY']
```




    salary                                                0
    to_messages                                         523
    deferral_payments                                     0
    total_payments                              1.54563e+07
    loan_advances                                         0
    bonus                                                 0
    email_address                sanjay.bhatnagar@enron.com
    restricted_stock_deferred                   1.54563e+07
    deferred_income                                       0
    total_stock_value                                     0
    expenses                                              0
    from_poi_to_this_person                               0
    exercised_stock_options                     2.60449e+06
    from_messages                                        29
    other                                            137864
    from_this_person_to_poi                               1
    poi                                               False
    long_term_incentive                                   0
    shared_receipt_with_poi                             463
    restricted_stock                           -2.60449e+06
    director_fees                                    137864
    Name: BHATNAGAR SANJAY, dtype: object




```python
df.loc[('BHATNAGAR SANJAY','total_payments')] = 137864
df.loc[('BHATNAGAR SANJAY','restricted_stock_deferred')] = -2604490
df.loc[('BHATNAGAR SANJAY','total_stock_value')] = 15456290
df.loc[('BHATNAGAR SANJAY','expenses')] = 137864
df.loc[('BHATNAGAR SANJAY','exercised_stock_options')] = 15456290
df.loc[('BHATNAGAR SANJAY','other')] = 0
df.loc[('BHATNAGAR SANJAY','restricted_stock')] = 2604490
df.loc[('BHATNAGAR SANJAY','director_fees')] = 0
```

Let's check to make sure that fixes our problems


```python
print(df[df[payment_categories[:-1]].sum(axis='columns') != df['total_payments']][payment_categories])
print(df[df[stock_value_categories[:-1]].sum(axis='columns') != df['total_stock_value']][stock_value_categories])
```

    Empty DataFrame
    Columns: [salary, bonus, long_term_incentive, deferred_income, deferral_payments, loan_advances, other, expenses, director_fees, total_payments]
    Index: []
    Empty DataFrame
    Columns: [exercised_stock_options, restricted_stock, restricted_stock_deferred, total_stock_value]
    Index: []
    

Looks good!

## Look for anomolies in the data

Now we're going to use a couple of functions provided by the Udacity class. They help to get data out of the dictionaries and into a more useable form. Here they are


```python
def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """


    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = list(dictionary.keys())

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)


def targetFeatureSplit( data ):
    """ 
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as 
        input formats when training/predicting)
    """

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features
```

One of the best ways to detect anomolies is to graph the data. Anomalies often stick out in these graphs. Let's take a look at how salary correlates with bonus. I suspect it will be positive and fairly strong.


```python
### read in data dictionary, convert to numpy array

features = ["salary", "bonus"]
data = featureFormat(enron_data, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
```


![png](2015-06-01-Cleaning-Data-with-Enron-Dataset_files/2015-06-01-Cleaning-Data-with-Enron-Dataset_49_0.png)


OK, someone's bonus and salary are way higher than everyone else's. That looks suspicious so let's take a look at it


```python
df[df['salary'] > 10000000]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>email_address</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TOTAL</th>
      <td>26704229.0</td>
      <td>0.0</td>
      <td>32083396.0</td>
      <td>309886585.0</td>
      <td>83925000.0</td>
      <td>97343619.0</td>
      <td>0</td>
      <td>-7576788.0</td>
      <td>-27992891.0</td>
      <td>434509511.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>311764000.0</td>
      <td>0.0</td>
      <td>42667589.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>48521928.0</td>
      <td>0.0</td>
      <td>130322299.0</td>
      <td>1398517.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>



Ah, there's an "employee" named "TOTAL" in the spreadsheet. Having a row that is the total of our other rows will mess up our statistics, so we'll remove it. We'll also remove The Travel Agency in the Park that we noticed earlier.


```python
entries_to_delete = ['THE TRAVEL AGENCY IN THE PARK', 'TOTAL']
for entry in entries_to_delete:
    if entry in df.index:
        df = df.drop(entry)
```

Now let's look again


```python
### read in data dictionary, convert to numpy array

features = ["salary", "bonus"]
#data = df["salary", "bonus"]

salary = df['salary'].values
bonus = df['bonus'].values
plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
```


![png](2015-06-01-Cleaning-Data-with-Enron-Dataset_files/2015-06-01-Cleaning-Data-with-Enron-Dataset_55_0.png)


That looks better

Now that we've cleaned up the data, let's save it as a CSV so we can pick it up and do some analysis on it another time.


```python
clean_file = 'clean_df.csv'
df.to_csv(path+clean_file, index_label='name')
```
