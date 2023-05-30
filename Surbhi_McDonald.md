```python
import pandas as pd 
import numpy as np 
import seaborn as sns
from matplotlib import pyplot as plt
```


```python
data = pd.read_csv("C://Users/Surbhi Pawar/OneDrive/Desktop/mcdonalds.csv")
data
```




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
      <th>yummy</th>
      <th>convenient</th>
      <th>spicy</th>
      <th>fattening</th>
      <th>greasy</th>
      <th>fast</th>
      <th>cheap</th>
      <th>tasty</th>
      <th>expensive</th>
      <th>healthy</th>
      <th>disgusting</th>
      <th>Like</th>
      <th>Age</th>
      <th>VisitFrequency</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>-3</td>
      <td>61</td>
      <td>Every three months</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>+2</td>
      <td>51</td>
      <td>Every three months</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>+1</td>
      <td>62</td>
      <td>Every three months</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>+4</td>
      <td>69</td>
      <td>Once a week</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>+2</td>
      <td>49</td>
      <td>Once a month</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>I hate it!-5</td>
      <td>47</td>
      <td>Once a year</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>+2</td>
      <td>36</td>
      <td>Once a week</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>1450</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>+3</td>
      <td>52</td>
      <td>Once a month</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>1451</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>+4</td>
      <td>41</td>
      <td>Every three months</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1452</th>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>-3</td>
      <td>30</td>
      <td>Every three months</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
<p>1453 rows × 15 columns</p>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1453 entries, 0 to 1452
    Data columns (total 15 columns):
     #   Column          Non-Null Count  Dtype 
    ---  ------          --------------  ----- 
     0   yummy           1453 non-null   object
     1   convenient      1453 non-null   object
     2   spicy           1453 non-null   object
     3   fattening       1453 non-null   object
     4   greasy          1453 non-null   object
     5   fast            1453 non-null   object
     6   cheap           1453 non-null   object
     7   tasty           1453 non-null   object
     8   expensive       1453 non-null   object
     9   healthy         1453 non-null   object
     10  disgusting      1453 non-null   object
     11  Like            1453 non-null   object
     12  Age             1453 non-null   int64 
     13  VisitFrequency  1453 non-null   object
     14  Gender          1453 non-null   object
    dtypes: int64(1), object(14)
    memory usage: 170.4+ KB
    


```python
data.describe()
```




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
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1453.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>44.604955</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.221178</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>33.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>57.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>71.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.isnull().mean()
```




    yummy             0.0
    convenient        0.0
    spicy             0.0
    fattening         0.0
    greasy            0.0
    fast              0.0
    cheap             0.0
    tasty             0.0
    expensive         0.0
    healthy           0.0
    disgusting        0.0
    Like              0.0
    Age               0.0
    VisitFrequency    0.0
    Gender            0.0
    dtype: float64




```python
data.mean()
```

    C:\Users\Surbhi Pawar\AppData\Local\Temp\ipykernel_14228\531903386.py:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      data.mean()
    




    Age    44.604955
    dtype: float64




```python
columns = data.columns
```


```python
Gender = ["Female", "Male"]
Color = ["yellow", "green"]
Size = data["Gender"].value_counts()
plt.pie(Size, labels=Gender, colors=Color, autopct="%.2f%%")
plt.legend()
plt.show()
```


    
![png](output_7_0.png)
    



```python
f = sns.countplot(x=data["Age"], palette='hsv')
f.bar_label(f.containers[0])
plt.rcParams['figure.figsize'] = (20, 5)
```


    
![png](output_8_0.png)
    



```python
data["Like"]=data["Like"].replace({'I hate it!-5': '-5', 'I love it!+5':'+5'})
data.dtypes
```




    yummy             object
    convenient        object
    spicy             object
    fattening         object
    greasy            object
    fast              object
    cheap             object
    tasty             object
    expensive         object
    healthy           object
    disgusting        object
    Like              object
    Age                int64
    VisitFrequency    object
    Gender            object
    dtype: object




```python
plt.figure(figsize=(20,5))
df.Age.value_counts().plot(kind='bar')
plt.xlabel('Age Distribution')
plt.ylabel('Number of Persons')
```




    Text(0, 0.5, 'Number of Persons')




    
![png](output_10_1.png)
    



```python
data.VisitFrequency.value_counts().plot(kind = 'bar')
plt.ylabel('Count')
plt.xlabel('Frequency')
```




    Text(0.5, 0, 'Frequency')




    
![png](output_11_1.png)
    



```python
data.hist()
```




    array([[<AxesSubplot:title={'center':'Age'}>]], dtype=object)




    
![png](output_12_1.png)
    



```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

cols=["yummy", "convenient", "spicy", "fattening", "greasy", "fast", "cheap", "tasty", "expensive", "healthy", "disgusting"]
    
for i in cols:
    data[i]=le.fit_transform(data[i])
data
```




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
      <th>yummy</th>
      <th>convenient</th>
      <th>spicy</th>
      <th>fattening</th>
      <th>greasy</th>
      <th>fast</th>
      <th>cheap</th>
      <th>tasty</th>
      <th>expensive</th>
      <th>healthy</th>
      <th>disgusting</th>
      <th>Like</th>
      <th>Age</th>
      <th>VisitFrequency</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-3</td>
      <td>61</td>
      <td>Every three months</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>+2</td>
      <td>51</td>
      <td>Every three months</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>+1</td>
      <td>62</td>
      <td>Every three months</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>+4</td>
      <td>69</td>
      <td>Once a week</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>+2</td>
      <td>49</td>
      <td>Once a month</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-5</td>
      <td>47</td>
      <td>Once a year</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>+2</td>
      <td>36</td>
      <td>Once a week</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>1450</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>+3</td>
      <td>52</td>
      <td>Once a month</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>1451</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>+4</td>
      <td>41</td>
      <td>Every three months</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1452</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-3</td>
      <td>30</td>
      <td>Every three months</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
<p>1453 rows × 15 columns</p>
</div>




```python
x = data.loc[:, cols]
x
```




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
      <th>yummy</th>
      <th>convenient</th>
      <th>spicy</th>
      <th>fattening</th>
      <th>greasy</th>
      <th>fast</th>
      <th>cheap</th>
      <th>tasty</th>
      <th>expensive</th>
      <th>healthy</th>
      <th>disgusting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1450</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1451</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1452</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1453 rows × 11 columns</p>
</div>




```python
y = data.Gender
```


```python
from sklearn.decomposition import PCA
from sklearn import preprocessing

mcd_data = preprocessing.scale(x)

mcd = PCA(n_components=11)
mc = mcd.fit_transform(x)
names=["yummy", "convenient", "spicy", "fattening", "greasy", "fast", "cheap", "tasty", "expensive", "healthy", "disgusting"]
md = pd.DataFrame(data=mc, columns=names)
md
```




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
      <th>yummy</th>
      <th>convenient</th>
      <th>spicy</th>
      <th>fattening</th>
      <th>greasy</th>
      <th>fast</th>
      <th>cheap</th>
      <th>tasty</th>
      <th>expensive</th>
      <th>healthy</th>
      <th>disgusting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.425367</td>
      <td>-0.219079</td>
      <td>0.663255</td>
      <td>-0.401300</td>
      <td>0.201705</td>
      <td>-0.389767</td>
      <td>-0.211982</td>
      <td>0.163235</td>
      <td>0.181007</td>
      <td>0.515706</td>
      <td>-0.567074</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.218638</td>
      <td>0.388190</td>
      <td>-0.730827</td>
      <td>-0.094724</td>
      <td>0.044669</td>
      <td>-0.086596</td>
      <td>-0.095877</td>
      <td>-0.034756</td>
      <td>0.111476</td>
      <td>0.493313</td>
      <td>-0.500440</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.375415</td>
      <td>0.730435</td>
      <td>-0.122040</td>
      <td>0.692262</td>
      <td>0.839643</td>
      <td>-0.687406</td>
      <td>0.583112</td>
      <td>0.364379</td>
      <td>-0.322288</td>
      <td>0.061759</td>
      <td>0.242741</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.172926</td>
      <td>-0.352752</td>
      <td>-0.843795</td>
      <td>0.206998</td>
      <td>-0.681415</td>
      <td>-0.036133</td>
      <td>-0.054284</td>
      <td>-0.231477</td>
      <td>-0.028003</td>
      <td>-0.250678</td>
      <td>-0.051034</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.187057</td>
      <td>-0.807610</td>
      <td>0.028537</td>
      <td>0.548332</td>
      <td>0.854074</td>
      <td>-0.097305</td>
      <td>-0.457043</td>
      <td>0.171758</td>
      <td>-0.074409</td>
      <td>0.031897</td>
      <td>0.082245</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>1.550242</td>
      <td>0.275031</td>
      <td>-0.013737</td>
      <td>0.200604</td>
      <td>-0.145063</td>
      <td>0.306575</td>
      <td>-0.075308</td>
      <td>0.345552</td>
      <td>-0.136589</td>
      <td>-0.432798</td>
      <td>-0.456076</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>-0.957339</td>
      <td>0.014308</td>
      <td>0.303843</td>
      <td>0.444350</td>
      <td>-0.133690</td>
      <td>0.381804</td>
      <td>-0.326432</td>
      <td>0.878047</td>
      <td>-0.304441</td>
      <td>-0.247443</td>
      <td>-0.193671</td>
    </tr>
    <tr>
      <th>1450</th>
      <td>-0.185894</td>
      <td>1.062662</td>
      <td>0.220857</td>
      <td>-0.467643</td>
      <td>-0.187757</td>
      <td>-0.192703</td>
      <td>-0.091597</td>
      <td>-0.036576</td>
      <td>0.038255</td>
      <td>0.056518</td>
      <td>-0.012800</td>
    </tr>
    <tr>
      <th>1451</th>
      <td>-1.182064</td>
      <td>-0.038570</td>
      <td>0.561561</td>
      <td>0.701126</td>
      <td>0.047645</td>
      <td>0.193687</td>
      <td>-0.027335</td>
      <td>-0.339374</td>
      <td>0.022267</td>
      <td>-0.002573</td>
      <td>-0.105316</td>
    </tr>
    <tr>
      <th>1452</th>
      <td>1.550242</td>
      <td>0.275031</td>
      <td>-0.013737</td>
      <td>0.200604</td>
      <td>-0.145063</td>
      <td>0.306575</td>
      <td>-0.075308</td>
      <td>0.345552</td>
      <td>-0.136589</td>
      <td>-0.432798</td>
      <td>-0.456076</td>
    </tr>
  </tbody>
</table>
<p>1453 rows × 11 columns</p>
</div>




```python
from sklearn.cluster import KMeans

wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(md)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
```


    
![png](output_17_0.png)
    



```python
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0).fit(x)

df['cluster_num'] = kmeans.labels_
print("Cluster labels:", kmeans.labels_)
print("Within-cluster sum of squares (inertia):", kmeans.inertia_)
print("Number of iterations:", kmeans.n_iter_)
print("Cluster centers:", kmeans.cluster_centers_)
```

    Cluster labels: [3 1 2 ... 2 4 0]
    Within-cluster sum of squares (inertia): 1434.6060971914799
    Number of iterations: 11
    Cluster centers: [[ 2.15517241e-02  6.85344828e-01  8.62068966e-02  9.35344828e-01
       7.32758621e-01  7.54310345e-01  6.89655172e-02  8.62068966e-02
       9.22413793e-01  6.03448276e-02  7.37068966e-01]
     [ 7.92880259e-01  9.80582524e-01  1.22977346e-01  9.70873786e-01
       1.00000000e+00  9.48220065e-01  8.93203883e-01  9.54692557e-01
       1.06796117e-01  1.81229773e-01  1.71521036e-01]
     [ 8.59922179e-01  9.53307393e-01  9.72762646e-02  8.83268482e-01
       5.21400778e-01  8.40466926e-01 -2.22044605e-16  9.41634241e-01
       1.00000000e+00  1.94552529e-01  6.61478599e-02]
     [ 3.78787879e-03  8.71212121e-01  6.43939394e-02  9.01515152e-01
       5.75757576e-01  9.35606061e-01  8.78787879e-01  3.78787879e-03
       1.51515152e-02  8.33333333e-02  4.01515152e-01]
     [ 8.46547315e-01  9.76982097e-01  9.20716113e-02  7.10997442e-01
       0.00000000e+00  9.64194373e-01  8.84910486e-01  9.66751918e-01
       3.06905371e-02  3.75959079e-01  1.53452685e-02]]
    


```python
from collections import Counter
Counter(kmeans.labels_)
```




    Counter({3: 264, 1: 309, 2: 257, 4: 391, 0: 232})




```python
std_dev = np.std(md,axis = 0)
print('Standard Deviation:',std_dev)
     
```

    Standard Deviation: yummy         0.756789
    convenient    0.607246
    spicy         0.504446
    fattening     0.398661
    greasy        0.337289
    fast          0.310168
    cheap         0.289598
    tasty         0.275027
    expensive     0.265160
    healthy       0.248756
    disgusting    0.236821
    dtype: float64
    


```python

```
