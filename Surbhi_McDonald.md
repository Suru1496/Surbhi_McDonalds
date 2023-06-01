```python
import pandas as pd 
import numpy as np 
import seaborn as sns
from matplotlib import pyplot as plt
```

, McDonald’s can take the position that, despite their market power,
there is value in investigating systematic heterogeneity among consumers and
harvest these differences using a differentiated marketing strategy


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



McDonald’s management needs to decide which key features make a market segment. The data set contains responses from 1453 adult Australian consumers relating to
their perceptions of McDonald’s with respect to the following attributes: YUMMY,
CONVENIENT, SPICY, FATTENING, GREASY, FAST, CHEAP, TASTY, EXPENSIVE,
HEALTHY, and DISGUSTING. These attributes emerged from a qualitative study conduct in preparation of survey study


```python
#Exploring data 
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
data.shape
```




    (1453, 15)




```python
data.columns.tolist()
```




    ['yummy',
     'convenient',
     'spicy',
     'fattening',
     'greasy',
     'fast',
     'cheap',
     'tasty',
     'expensive',
     'healthy',
     'disgusting',
     'Like',
     'Age',
     'VisitFrequency',
     'Gender']




```python
data.head(5)
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
  </tbody>
</table>
</div>



As we can see from the output, the first respondent believes that McDonald’s is not
yummy, convenient, not spicy, fattening, not greasy, fast, cheap, not tasty, expensive,
not healthy and not disgusting. This same respondent does not like McDonald’s
(rating of −3), is 61 years old, eats at McDonald’s every three months and is female


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
#Extract the first 11 columns and convert to matrix
MD_x = data.iloc[:, 0:11].values

# Convert "Yes" to 1 and "No" to 0
MD_x = (MD_x == "Yes").astype(int)

# Calculate column means and round to 2 decimal places
col_means = np.round(np.mean(MD_x, axis=0), 2)

col_means
```




    array([0.55, 0.91, 0.09, 0.87, 0.53, 0.9 , 0.6 , 0.64, 0.36, 0.2 , 0.24])



The average values of the transformed binary numeric segmentation variables
indicate that about half of the respondents (55%) perceive McDonald’s as YUMMY,
91% believe that eating at McDonald’s is CONVENIENT, but only 9% think that
McDonald’s food is SPICY.  Here, we
calculate principal components because we use the resulting components to rotate
and project the data for the perceptual map. We use unstandardised data because our
segmentation variables are all binary.


```python
from sklearn.decomposition import PCA
# Perform PCA
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

# Display the summary
print("Summary of MD.pca:")
print("Standard deviations:\n", pca.explained_variance_)
print("Proportions of variance:\n", pca.explained_variance_ratio_)
```

    Summary of MD.pca:
    Standard deviations:
     [0.57312398 0.36900226 0.2546408  0.15904032 0.11384214 0.09627033
     0.08392454 0.07569209 0.07035814 0.06192225 0.05612296]
    Proportions of variance:
     [0.29944723 0.19279721 0.13304535 0.08309578 0.05948052 0.05029956
     0.0438491  0.03954779 0.0367609  0.03235329 0.02932326]
    

Results from principal components analysis indicate that the first two components
capture about 50% of the information contained in the segmentation variables. The
following command returns the factor loadings


```python
# Print the standard deviations
print("Standard deviations (1, ..., p=11):")
print(pca.explained_variance_.round(1))

```

    Standard deviations (1, ..., p=11):
    [0.6 0.4 0.3 0.2 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
    


```python
print("Proportion of Variance:")
print(pca.explained_variance_ratio_.round(2))
```

    Proportion of Variance:
    [0.3  0.19 0.13 0.08 0.06 0.05 0.04 0.04 0.04 0.03 0.03]
    


```python
# Select the factor loadings for the first two components
factor_loadings = pca.components_[:2, :]

# Display the factor loadings
print("Factor Loadings:")
print(factor_loadings)
```

    Factor Loadings:
    [[-0.47693349 -0.15533159 -0.00635636  0.11623168  0.3044427  -0.10849325
      -0.33718593 -0.47151394  0.32904173 -0.21371062  0.37475293]
     [ 0.36378978  0.016414    0.01880869 -0.03409395 -0.06383884 -0.0869722
      -0.61063276  0.3073178   0.60128596  0.07659344 -0.13965633]]
    


```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Plot the projected points along the principal components
plt.scatter(MD_pca[:, 0], MD_pca[:, 1], color='grey')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
```




    Text(0, 0.5, 'Principal Component 2')




    
![png](output_18_1.png)
    


 The attributes CHEAP and
EXPENSIVE play a key role in the evaluation of McDonald’s, and these two attributes
are assessed quite independently of the others. The remaining attributes align
with what can be interpreted as positive versus negative perceptions: FATTENING,
DISGUSTING and GREASY point in the same direction in the perceptual chart,
indicating that respondents who view McDonald’s as FATTENING, DISGUSTING are
also likely to view it as GREASY. In the opposite direction are the positive attributes
FAST, CONVENIENT, HEALTHY, as well as TASTY and YUMMY. The observations
along the EXPENSIVE versus CHEAP axis cluster around three values: a group of
consumers at the top around the arrow pointing to CHEAP, a group of respondentat the bottom around the arrow pointing to EXPENSIVE, and a group of respondents
in the middle.


```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set the random seed
np.random.seed(1234)
# Perform k-means clustering with 2 to 8 clusters
k_values = range(2, 9)
scores = []

for k in k_values:
    model = KMeans(n_clusters=k, n_init=10, random_state=1234)
    model.fit(MD_x)
    score = model.score(MD_x)
    scores.append(score)
```

The k-means clustering is performed using the KMeans class, and the best model is selected based on the highest score (in this case, the sum of squared distances within each cluster). The clusters are then relabeled using the labels_ attribute of the best model.


```python
# Plot the k-means clustering results
plt.plot(k_values, scores, marker='o')
plt.xlabel('Number of Segments')
plt.ylabel('Score')
plt.title('K-means Clustering Results')
plt.show()
```


    
![png](output_22_0.png)
    


the sum of distances within
market segments drops slowly as the number of market segments increases. We
expect the values to decrease because more market segments automatically mean
that the segments are smaller and, as a consequence, that segment members are more
similar to one another. But the much anticipated point where the sum of distances
drops dramatically is not visible. This scree plot does not provide useful guidance
on the number of market segments to extract.


```python
from sklearn.metrics import adjusted_rand_score
# Define the number of clusters
k_values = range(2, 9)

# Define the number of repetitions and bootstrap iterations
n_rep = 10
n_boot = 100

# Initialize an empty list to store the adjusted Rand indices
ari_values = []

# Perform bootstrapping for each number of clusters
for k in k_values:
    ari_boot = []
    
    # Perform repetitions for each number of clusters
    for _ in range(n_rep):
        model = KMeans(n_clusters=k, n_init=10, random_state=1234)
        model.fit(MD_x)
        
        # Perform bootstrap iterations
        for _ in range(n_boot):
            bootstrap_indices = np.random.choice(range(len(MD_x)), size=len(MD_x), replace=True)
            bootstrap_samples = MD_x[bootstrap_indices]
            bootstrap_labels = model.predict(bootstrap_samples)
            ari = adjusted_rand_score(model.labels_, bootstrap_labels)
            ari_boot.append(ari)
    
    ari_values.append(ari_boot)

# Plot the global stability boxplot
plt.boxplot(ari_values)
plt.xlabel('Number of Segments')
plt.ylabel('Adjusted Rand Index')
plt.title('Global Stability Boxplot')
plt.show()




```


    
![png](output_24_0.png)
    



```python
import matplotlib.pyplot as plt

# Extract the desired column or flatten the array
cluster_4_data = MD_pca[:, 0]  # Assuming the desired column is the first column

# Plot the histogram
plt.hist(cluster_4_data, bins=np.linspace(0, 1, num=11))
plt.xlabel('Variable Value')
plt.ylabel('Frequency')
plt.show()

```


    
![png](output_25_0.png)
    



```python
from sklearn.mixture import GaussianMixture

# Fit Gaussian mixture models with different numbers of components
k_values = range(2, 9)
models = []
for k in k_values:
    model = GaussianMixture(n_components=k, random_state=1234)
    model.fit(MD_x)
    models.append(model)

# Print the fitted models
for k, model in zip(k_values, models):
    print(f"Number of components (k): {k}")
    print(model)
    print()
```

    Number of components (k): 2
    GaussianMixture(n_components=2, random_state=1234)
    
    Number of components (k): 3
    GaussianMixture(n_components=3, random_state=1234)
    
    Number of components (k): 4
    GaussianMixture(n_components=4, random_state=1234)
    
    Number of components (k): 5
    GaussianMixture(n_components=5, random_state=1234)
    
    Number of components (k): 6
    GaussianMixture(n_components=6, random_state=1234)
    
    Number of components (k): 7
    GaussianMixture(n_components=7, random_state=1234)
    
    Number of components (k): 8
    GaussianMixture(n_components=8, random_state=1234)
    
    


```python
import matplotlib.pyplot as plt

# Calculate the AIC, BIC, and ICL values for each model
aic_values = [model.aic(MD_x) for model in models]
bic_values = [model.bic(MD_x) for model in models]
icl_values = [model.lower_bound_ for model in models]

# Plot the information criteria values
plt.plot(k_values, aic_values, label='AIC')
plt.plot(k_values, bic_values, label='BIC')
plt.plot(k_values, icl_values, label='ICL')
plt.xlabel('Number of components (k)')
plt.ylabel('Value of information criteria')
plt.legend()
plt.show()

```


    
![png](output_27_0.png)
    



```python
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np

# Compute the pairwise distance matrix
distance_matrix = np.transpose(MD_x)
linkage_matrix = linkage(distance_matrix)

# Plot the dendrogram
dendrogram(linkage_matrix)

# Show the plot
plt.show()

```


    
![png](output_28_0.png)
    



```python
import numpy as np
from sklearn.cluster import KMeans

# Set the random seed
np.random.seed(1234)

# Perform K-means clustering
k_values = range(2, 9)
n_rep = 10
MD_km28 = None

for k in k_values:
    km = KMeans(n_clusters=k, random_state=1234, n_init=n_rep)
    labels = km.fit_predict(MD_x)
    if MD_km28 is None:
        MD_km28 = labels
    else:
        MD_km28 = np.vstack((MD_km28, labels))

# Relabel the clusters
MD_km28 = np.argmax(MD_km28, axis=0)

# Print the relabeled cluster labels
print('Relabeled cluster labels:')
print(MD_km28)


```

    Relabeled cluster labels:
    [6 4 6 ... 6 5 4]
    


```python
# Assuming you have already computed MD_km28

# Extract the cluster labels for cluster 4
MD_k4 = MD_km28[MD_km28 == 3]  # Assuming 3 represents cluster 4

# Print the cluster labels for cluster 4
print('Cluster labels for cluster 4:')
print(MD_k4)

```

    Cluster labels for cluster 4:
    [3]
    


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have already computed MD_k4 and MD_pca

# Create a scatter plot of the cluster solution on the principal components
plt.scatter(MD_pca[:, 0], MD_pca[:, 1],c = MD_km28, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Add projected axes to the plot
plt.arrow(0, 0, 1, 0, color='red', width=0.01, head_width=0.1)
plt.arrow(0, 0, 0, 1, color='blue', width=0.01, head_width=0.1)

# Set the aspect ratio to 'equal'
plt.gca().set_aspect('equal')

# Show the plot
plt.show()

```


    
![png](output_31_0.png)
    



```python
# Assuming mcdonalds is a DataFrame with a column named 'Like'
like_counts = data['Like'].value_counts()
reversed_counts = like_counts[::-1]

# Print the reversed table
print(reversed_counts)
```

    -1               58
    -2               59
    -4               71
    -3               73
    I love it!+5    143
    I hate it!-5    152
    +1              152
    +4              160
    0               169
    +2              187
    +3              229
    Name: Like, dtype: int64
    


```python
import pandas as pd

# Replace non-numeric values with NaN in the 'Like' column
data['Like'] = pd.to_numeric(data['Like'], errors='coerce')

# Create a new column 'Like.n' by subtracting the numeric values in 'Like' from 6
data['Like.n'] = 6 - data['Like']

# Calculate the frequency table for the 'Like.n' column
like_n_table = pd.value_counts(data['Like.n'])

# Print the table
print(like_n_table)

```

    3.0     229
    4.0     187
    6.0     169
    2.0     160
    5.0     152
    9.0      73
    10.0     71
    8.0      59
    7.0      58
    Name: Like.n, dtype: int64
    


```python
pip install patsy
```

    Requirement already satisfied: patsy in c:\users\surbhi pawar\anaconda3\lib\site-packages (0.5.2)
    Requirement already satisfied: six in c:\users\surbhi pawar\anaconda3\lib\site-packages (from patsy) (1.16.0)
    Requirement already satisfied: numpy>=1.4 in c:\users\surbhi pawar\anaconda3\lib\site-packages (from patsy) (1.21.5)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import patsy

# Create the formula string
f = "Like.n ~ " + " + ".join(data.columns[:11])

# Create the formula object
f = patsy.ModelDesc.from_formula(f)

# Print the formula
print(f)

```

    ModelDesc(lhs_termlist=[Term([EvalFactor('Like.n')])],
              rhs_termlist=[Term([]),
                            Term([EvalFactor('yummy')]),
                            Term([EvalFactor('convenient')]),
                            Term([EvalFactor('spicy')]),
                            Term([EvalFactor('fattening')]),
                            Term([EvalFactor('greasy')]),
                            Term([EvalFactor('fast')]),
                            Term([EvalFactor('cheap')]),
                            Term([EvalFactor('tasty')]),
                            Term([EvalFactor('expensive')]),
                            Term([EvalFactor('healthy')]),
                            Term([EvalFactor('disgusting')])])
    


```python
from sklearn.mixture import GaussianMixture
import numpy as np

# Set the random seed
np.random.seed(1234)

# Perform model selection
k_values = range(2, 9)  # Range of k values to consider
bic_values = []  # List to store BIC values

for k in k_values:
    # Fit the Gaussian Mixture model
    model = GaussianMixture(n_components=k, random_state=1234)
    model.fit(data)
    
    # Calculate BIC
    bic = model.bic(data)
    bic_values.append(bic)

# Find the optimal number of components based on BIC
optimal_k = k_values[np.argmin(bic_values)]

# Fit the final model with the optimal number of components
final_model = GaussianMixture(n_components=optimal_k, random_state=1234)
final_model.fit(data)

# Print the final model parameters
print("Weights:")
print(final_model.weights_)
print("Means:")
print(final_model.means_)
print("Covariances:")
print(final_model.covariances_)

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Input In [32], in <cell line: 11>()
         11 for k in k_values:
         12     # Fit the Gaussian Mixture model
         13     model = GaussianMixture(n_components=k, random_state=1234)
    ---> 14     model.fit(data)
         16     # Calculate BIC
         17     bic = model.bic(data)
    

    File ~\anaconda3\lib\site-packages\sklearn\mixture\_base.py:198, in BaseMixture.fit(self, X, y)
        172 def fit(self, X, y=None):
        173     """Estimate model parameters with the EM algorithm.
        174 
        175     The method fits the model ``n_init`` times and sets the parameters with
       (...)
        196         The fitted mixture.
        197     """
    --> 198     self.fit_predict(X, y)
        199     return self
    

    File ~\anaconda3\lib\site-packages\sklearn\mixture\_base.py:228, in BaseMixture.fit_predict(self, X, y)
        201 def fit_predict(self, X, y=None):
        202     """Estimate model parameters using X and predict the labels for X.
        203 
        204     The method fits the model n_init times and sets the parameters with
       (...)
        226         Component labels.
        227     """
    --> 228     X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        229     if X.shape[0] < self.n_components:
        230         raise ValueError(
        231             "Expected n_samples >= n_components "
        232             f"but got n_components = {self.n_components}, "
        233             f"n_samples = {X.shape[0]}"
        234         )
    

    File ~\anaconda3\lib\site-packages\sklearn\base.py:566, in BaseEstimator._validate_data(self, X, y, reset, validate_separately, **check_params)
        564     raise ValueError("Validation should be done on X, y or both.")
        565 elif not no_val_X and no_val_y:
    --> 566     X = check_array(X, **check_params)
        567     out = X
        568 elif no_val_X and not no_val_y:
    

    File ~\anaconda3\lib\site-packages\sklearn\utils\validation.py:746, in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
        744         array = array.astype(dtype, casting="unsafe", copy=False)
        745     else:
    --> 746         array = np.asarray(array, order=order, dtype=dtype)
        747 except ComplexWarning as complex_warning:
        748     raise ValueError(
        749         "Complex data not supported\n{}\n".format(array)
        750     ) from complex_warning
    

    File ~\anaconda3\lib\site-packages\pandas\core\generic.py:2064, in NDFrame.__array__(self, dtype)
       2063 def __array__(self, dtype: npt.DTypeLike | None = None) -> np.ndarray:
    -> 2064     return np.asarray(self._values, dtype=dtype)
    

    ValueError: could not convert string to float: 'No'



```python
data.mean()
```

    C:\Users\Surbhi Pawar\AppData\Local\Temp\ipykernel_23960\531903386.py:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      data.mean()
    




    Like       1.013817
    Age       44.604955
    Like.n     4.986183
    dtype: float64




```python
data.VisitFrequency.value_counts().plot(kind = 'bar')
plt.ylabel('Count')
plt.xlabel('Frequency')
```




    Text(0.5, 0, 'Frequency')




    
![png](output_38_1.png)
    



```python
data.hist()
```




    array([[<AxesSubplot:title={'center':'Like'}>,
            <AxesSubplot:title={'center':'Age'}>],
           [<AxesSubplot:title={'center':'Like.n'}>, <AxesSubplot:>]],
          dtype=object)




    
![png](output_39_1.png)
    



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
      <th>Like.n</th>
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
      <td>-3.0</td>
      <td>61</td>
      <td>Every three months</td>
      <td>Female</td>
      <td>9.0</td>
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
      <td>2.0</td>
      <td>51</td>
      <td>Every three months</td>
      <td>Female</td>
      <td>4.0</td>
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
      <td>1.0</td>
      <td>62</td>
      <td>Every three months</td>
      <td>Female</td>
      <td>5.0</td>
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
      <td>4.0</td>
      <td>69</td>
      <td>Once a week</td>
      <td>Female</td>
      <td>2.0</td>
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
      <td>2.0</td>
      <td>49</td>
      <td>Once a month</td>
      <td>Male</td>
      <td>4.0</td>
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
      <td>NaN</td>
      <td>47</td>
      <td>Once a year</td>
      <td>Male</td>
      <td>NaN</td>
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
      <td>2.0</td>
      <td>36</td>
      <td>Once a week</td>
      <td>Female</td>
      <td>4.0</td>
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
      <td>3.0</td>
      <td>52</td>
      <td>Once a month</td>
      <td>Female</td>
      <td>3.0</td>
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
      <td>4.0</td>
      <td>41</td>
      <td>Every three months</td>
      <td>Male</td>
      <td>2.0</td>
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
      <td>-3.0</td>
      <td>30</td>
      <td>Every three months</td>
      <td>Male</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
<p>1453 rows × 16 columns</p>
</div>




```python
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
import numpy as np

# Preprocess data
label_encoder = LabelEncoder()
encoded_data = data.apply(label_encoder.fit_transform)

# Set the random seed
np.random.seed(1234)

# Perform model selection
k_values = range(2, 9)  # Range of k values to consider
bic_values = []  # List to store BIC values

for k in k_values:
    # Fit the Gaussian Mixture model
    model = GaussianMixture(n_components=k, random_state=1234)
    model.fit(encoded_data)
    
    # Calculate BIC
    bic = model.bic(encoded_data)
    bic_values.append(bic)

# Find the optimal number of components based on BIC
optimal_k = k_values[np.argmin(bic_values)]

# Fit the final model with the optimal number of components
final_model = GaussianMixture(n_components=optimal_k, random_state=1234)
final_model.fit(encoded_data)

# Print the final model parameters
print("Weights:")
print(final_model.weights_)
print("Means:")
print(final_model.means_)
print("Covariances:")
print(final_model.covariances_)

```

    Weights:
    [0.01307547 0.20165175 0.00756674 0.03165889 0.08189508 0.56160491
     0.01376458 0.08878259]
    Means:
    [[4.21082752e-01 3.15741738e-01 3.15741738e-01 7.36823449e-01
      5.26294104e-01 1.00000000e+00 2.63176132e-01 5.78988679e-01
      7.89459177e-01 2.63177253e-01 3.15741299e-01 4.47398308e+00
      1.29465776e+01 3.15787750e+00 5.78978563e-01 3.52601692e+00]
     [5.05119454e-01 7.50853242e-01 8.87372014e-02 8.32764505e-01
      5.80204778e-01 8.60068259e-01 5.73378840e-01 5.35836177e-01
      4.33447099e-01 2.38907850e-01 4.70989761e-01 9.00000000e+00
      2.54778157e+01 2.86348123e+00 4.50511945e-01 9.00000000e+00]
     [9.09548443e-02 2.72864533e-01 2.72864533e-01 8.18090311e-01
      7.27135467e-01 5.45325431e-01 4.54448807e-01 2.72864533e-01
      5.45551193e-01 1.81909689e-01 5.45310596e-01 3.27409027e+00
      1.75450423e+01 2.81852526e+00 5.45551193e-01 6.54500661e+00]
     [8.47819934e-01 9.13042497e-01 2.39128557e-01 4.34784436e-02
      0.00000000e+00 9.56520855e-01 8.91304267e-01 9.34781981e-01
      8.69555478e-02 8.26080450e-01 2.17389598e-02 6.36952782e+00
      3.87390725e+01 2.63045241e+00 4.56518214e-01 1.63047218e+00]
     [2.85729302e-01 8.15069528e-01 2.85729302e-01 1.00000000e+00
      1.00000000e+00 8.82338512e-01 6.05031299e-01 8.15069528e-01
      4.20181687e-01 1.59668200e-01 4.37020689e-01 3.81487535e+00
      2.82433663e+01 2.47897435e+00 4.70588424e-01 4.18512465e+00]
     [6.24990865e-01 1.00000000e+00 0.00000000e+00 1.00000000e+00
      5.07360142e-01 9.21569606e-01 6.05394003e-01 6.48289455e-01
      3.38234047e-01 1.38478976e-01 1.59313778e-01 5.17278866e+00
      2.50845864e+01 2.61029239e+00 4.38725110e-01 2.82721134e+00]
     [4.00000910e-01 3.00001031e-01 4.00001326e-01 4.00000854e-01
      3.50001178e-01 6.50001788e-01 4.50001194e-01 6.00001516e-01
      3.00000888e-01 2.00000591e-01 1.50000330e-01 4.09999890e+00
      3.62500000e+01 2.59999347e+00 3.50000815e-01 3.90000110e+00]
     [4.26353208e-01 1.00000000e+00 3.72100587e-01 3.72100587e-01
      2.86828358e-01 9.06977532e-01 5.89140097e-01 6.51155275e-01
      2.79078091e-01 2.94571176e-01 1.31792345e-01 5.06197505e+00
      3.42324583e+01 2.36434603e+00 5.73640336e-01 2.93802495e+00]]
    Covariances:
    [[[ 2.43773068e-01 -2.76815229e-02 -2.76815229e-02 ...  1.44059763e-01
       -3.32553957e-02 -5.89946280e-01]
      [-2.76815229e-02  2.16049893e-01  2.16048893e-01 ... -1.02510426e-01
        8.03589068e-02  9.67552530e-02]
      [-2.76815229e-02  2.16048893e-01  2.16049893e-01 ... -1.02510426e-01
        8.03589068e-02  9.67552530e-02]
      ...
      [ 1.44059763e-01 -1.02510426e-01 -1.02510426e-01 ...  3.08039473e+00
       -2.49280136e-01  9.16821706e-01]
      [-3.32553957e-02  8.03589068e-02  8.03589068e-02 ... -2.49280136e-01
        2.43763387e-01 -2.51951588e-01]
      [-5.89946280e-01  9.67552530e-02  9.67552530e-02 ...  9.16821706e-01
       -2.51951588e-01  5.82740386e+00]]
    
     [[ 2.49974791e-01  9.85451199e-02  2.95868327e-03 ... -2.71406772e-03
       -3.30230987e-02  0.00000000e+00]
      [ 9.85451199e-02  1.87073651e-01  8.45670887e-03 ...  2.05942993e-02
       -2.08622116e-02  0.00000000e+00]
      [ 2.95868327e-03  8.45670887e-03  8.08639105e-02 ... -4.95055271e-03
        1.46303393e-02  0.00000000e+00]
      ...
      [-2.71406772e-03  2.05942993e-02 -4.95055271e-03 ...  2.12470833e+00
        1.71347366e-02  0.00000000e+00]
      [-3.30230987e-02 -2.08622116e-02  1.46303393e-02 ...  1.71347366e-02
        2.47551932e-01  0.00000000e+00]
      [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
        0.00000000e+00  1.00000000e-06]]
    
     [[ 8.26830606e-02  6.61364932e-02  6.61364932e-02 ...  1.07460851e-01
        4.13343205e-02 -1.40525836e-01]
      [ 6.61364932e-02  1.98410480e-01  1.07454635e-01 ...  3.22382554e-01
       -5.79067271e-02  1.24151558e-01]
      [ 6.61364932e-02  1.07454635e-01  1.98410480e-01 ...  4.95180209e-02
        3.30481172e-02 -3.30622664e-01]
      ...
      [ 1.07460851e-01  3.22382554e-01  4.95180209e-02 ...  3.05803044e+00
       -2.64706083e-01  2.81282623e-01]
      [ 4.13343205e-02 -5.79067271e-02  3.30481172e-02 ... -2.64706083e-01
        2.47926089e-01 -2.06458982e-01]
      [-1.40525836e-01  1.24151558e-01 -3.30622664e-01 ...  2.81282623e-01
       -2.06458982e-01  2.43025403e+00]]
    
     ...
    
     [[ 2.34378284e-01  1.26882942e-31  0.00000000e+00 ... -3.21675673e-02
       -7.04622791e-03 -6.56703374e-01]
      [ 1.26882942e-31  1.00000000e-06  0.00000000e+00 ...  4.96897768e-31
        7.32295835e-32  7.77248649e-31]
      [ 0.00000000e+00  0.00000000e+00  1.00000000e-06 ...  0.00000000e+00
        0.00000000e+00  0.00000000e+00]
      ...
      [-3.21675673e-02  4.96897768e-31  0.00000000e+00 ...  3.19618657e+00
       -2.26554786e-02  1.40999732e-01]
      [-7.04622791e-03  7.32295835e-32  0.00000000e+00 ... -2.26554786e-02
        2.46246388e-01  1.82128956e-02]
      [-6.56703374e-01  7.77248649e-31  0.00000000e+00 ...  1.40999732e-01
        1.82128956e-02  5.23849656e+00]]
    
     [[ 2.40001182e-01  2.99997562e-02 -1.00004526e-02 ...  6.00034860e-02
        5.99996768e-02 -9.10002344e-01]
      [ 2.99997562e-02  2.10001413e-01  1.80000221e-01 ... -3.29999341e-01
       -5.00016723e-03 -7.00001551e-02]
      [-1.00004526e-02  1.80000221e-01  2.40001265e-01 ... -3.89998864e-01
       -4.00003518e-02  1.90000501e-01]
      ...
      [ 6.00034860e-02 -3.29999341e-01 -3.89998864e-01 ...  3.23999274e+00
       -1.09997622e-01 -1.99000912e+00]
      [ 5.99996768e-02 -5.00016723e-03 -4.00003518e-02 ... -1.09997622e-01
        2.27501244e-01 -1.14999948e-01]
      [-9.10002344e-01 -7.00001551e-02  1.90000501e-01 ... -1.99000912e+00
       -1.14999948e-01  8.19002446e+00]]
    
     [[ 2.44577150e-01  1.07779495e-31 -4.23679523e-02 ... -1.78595706e-01
       -4.30245548e-02 -6.32485970e-01]
      [ 1.07779495e-31  1.00000000e-06  7.03242093e-32 ...  4.64751297e-31
        1.33004483e-31  8.31660215e-31]
      [-4.23679523e-02  7.18529965e-32  2.33642740e-01 ... -3.77707870e-03
       -2.74051826e-02  3.33184875e-01]
      ...
      [-1.78595706e-01  4.83096742e-31 -3.77707870e-03 ...  3.42538454e+00
        8.04577497e-03  4.33481598e-01]
      [-4.30245548e-02  1.37590844e-31 -2.74051826e-02 ...  8.04577497e-03
        2.44578101e-01  1.44082164e-01]
      [-6.32485970e-01  8.31660215e-31  3.33184875e-01 ...  4.33481598e-01
        1.44082164e-01  5.42267123e+00]]]
    


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
# Fit the final model with the optimal number of components
final_model.fit(encoded_data)

# Access the attributes of the fitted model
weights = final_model.weights_
means = final_model.means_
covariances = final_model.covariances_

# Compute summary statistics
num_components = final_model.n_components
num_features = encoded_data.shape[1]

print("Number of components:", num_components)
print("Number of features:", num_features)
print()

for component in range(num_components):
    print("Component", component + 1)
    print("Weight:", weights[component])
    print("Mean:", means[component])
    print("Covariance:")
    print(covariances[component])
    print()

```

    Number of components: 8
    Number of features: 16
    
    Component 1
    Weight: 0.013075466664287756
    Mean: [ 0.42108275  0.31574174  0.31574174  0.73682345  0.5262941   1.
      0.26317613  0.57898868  0.78945918  0.26317725  0.3157413   4.47398308
     12.94657758  3.1578775   0.57897856  3.52601692]
    Covariance:
    [[ 2.43773068e-01 -2.76815229e-02 -2.76815229e-02 -4.70868247e-02
       4.15644340e-02  2.07609693e-32  4.70863656e-02  1.77280602e-01
      -6.92501880e-02  4.70872304e-02 -1.32952512e-01  5.89946280e-01
      -1.50392694e+00  1.44059763e-01 -3.32553957e-02 -5.89946280e-01]
     [-2.76815229e-02  2.16049893e-01  2.16048893e-01 -7.48101096e-02
      -8.32608373e-03  7.13658319e-33 -3.04600266e-02 -7.75390187e-02
       1.38408610e-02  7.48105893e-02  5.81425427e-02 -9.67552530e-02
       2.22675772e+00 -1.02510426e-01  8.03589068e-02  9.67552530e-02]
     [-2.76815229e-02  2.16048893e-01  2.16049893e-01 -7.48101096e-02
      -8.32608373e-03  7.13658319e-33 -3.04600266e-02 -7.75390187e-02
       1.38408610e-02  7.48105893e-02  5.81425427e-02 -9.67552530e-02
       2.22675772e+00 -1.02510426e-01  8.03589068e-02  9.67552530e-02]
     [-4.70868247e-02 -7.48101096e-02 -7.48101096e-02  1.93915654e-01
       8.58729568e-02  3.37365751e-32 -3.60088334e-02 -5.52968748e-03
       4.98612123e-02 -1.41279160e-01  8.30957060e-02 -2.43705940e-01
      -1.06676574e+00 -1.16356277e-01 -5.81676600e-02  2.43705940e-01]
     [ 4.15644340e-02 -8.32608373e-03 -8.32608373e-03  8.58729568e-02
       2.49309620e-01  3.11414539e-32 -3.32370778e-02  1.16365407e-01
       5.53542340e-03 -3.32373146e-02 -8.32640161e-03  6.66007586e-02
      -3.02539178e+00  7.47559282e-02  1.11000430e-02 -6.66007586e-02]
     [ 2.33560904e-32  7.13658319e-33  7.13658319e-33  2.91951131e-32
       3.37365751e-32  1.00000000e-06  1.16780452e-32  3.11414539e-32
       6.29316881e-32  1.16780452e-32  3.37365751e-32  2.49131631e-31
       1.16261428e-30  2.02419450e-31  4.67121809e-32  1.45326785e-31]
     [ 4.70863656e-02 -3.04600266e-02 -3.04600266e-02 -3.60088334e-02
      -3.32370778e-02  1.10292649e-32  1.93915455e-01  5.52929050e-03
      -1.55131502e-01 -1.66266609e-02 -8.30954550e-02  1.91070458e-01
      -1.03864378e+00 -1.99457366e-01  5.53303641e-03 -1.91070458e-01]
     [ 1.77280602e-01 -7.75390187e-02 -7.75390187e-02 -5.52968748e-03
       1.16365407e-01  3.11414539e-32  5.52929050e-03  2.43761789e-01
      -3.60045401e-02  5.81652927e-02 -1.30174624e-01  4.62466220e-01
      -2.81137413e+00  3.29671211e-01  3.32263886e-02 -4.62466220e-01]
     [-6.92501880e-02  1.38408610e-02  1.38408610e-02  4.98612123e-02
       5.53542340e-03  6.42292487e-32 -1.55131502e-01 -3.60045401e-02
       1.66214385e-01  2.77424505e-03  6.64763125e-02 -5.81127703e-02
       9.88820384e-01  8.58768145e-02  1.66275587e-02  5.81127703e-02]
     [ 4.70872304e-02  7.48105893e-02  7.48105893e-02 -1.41279160e-01
      -3.32373146e-02  1.03804846e-32 -1.66266609e-02  5.81652927e-02
       2.77424505e-03  1.93915987e-01 -3.04599160e-02  2.43707711e-01
       1.32994934e+00  2.21626786e-01  1.10803265e-01 -2.43707711e-01]
     [-1.32952512e-01  5.81425427e-02  5.81425427e-02  8.30957060e-02
      -8.32640161e-03  3.37365751e-32 -8.30954550e-02 -1.30174624e-01
       6.64763125e-02 -3.04599160e-02  2.16049731e-01 -6.23108773e-01
       1.75304804e+00 -2.07780736e-01 -2.49120085e-02  6.23108773e-01]
     [ 5.89946280e-01 -9.67552530e-02 -9.67552530e-02 -2.43705940e-01
       6.66007586e-02  2.49131631e-31  1.91070458e-01  4.62466220e-01
      -5.81127703e-02  2.43707711e-01 -6.23108773e-01  5.82740386e+00
      -7.65651431e+00 -9.16821706e-01  2.51951588e-01 -5.82740286e+00]
     [-1.50392694e+00  2.22675772e+00  2.22675772e+00 -1.06676574e+00
      -3.02539178e+00  1.18337525e-30 -1.03864378e+00 -2.81137413e+00
       9.88820384e-01  1.32994934e+00  1.75304804e+00 -7.65651431e+00
       1.10470064e+02  4.06109350e+00 -4.42915763e-01  7.65651431e+00]
     [ 1.44059763e-01 -1.02510426e-01 -1.02510426e-01 -1.16356277e-01
       7.47559282e-02  2.02419450e-31 -1.99457366e-01  3.29671211e-01
       8.58768145e-02  2.21626786e-01 -2.07780736e-01 -9.16821706e-01
       4.06109350e+00  3.08039473e+00 -2.49280136e-01  9.16821706e-01]
     [-3.32553957e-02  8.03589068e-02  8.03589068e-02 -5.81676600e-02
       1.11000430e-02  4.67121809e-32  5.53303641e-03  3.32263886e-02
       1.66275587e-02  1.10803265e-01 -2.49120085e-02  2.51951588e-01
      -4.42915763e-01 -2.49280136e-01  2.43763387e-01 -2.51951588e-01]
     [-5.89946280e-01  9.67552530e-02  9.67552530e-02  2.43705940e-01
      -6.66007586e-02  1.45326785e-31 -1.91070458e-01 -4.62466220e-01
       5.81127703e-02 -2.43707711e-01  6.23108773e-01 -5.82740286e+00
       7.65651431e+00  9.16821706e-01 -2.51951588e-01  5.82740386e+00]]
    
    Component 2
    Weight: 0.20165175498967652
    Mean: [ 0.50511945  0.75085324  0.0887372   0.83276451  0.58020478  0.86006826
      0.57337884  0.53583618  0.4334471   0.23890785  0.47098976  9.
     25.4778157   2.86348123  0.45051195  9.        ]
    Covariance:
    [[ 2.49974791e-01  9.85451199e-02  2.95868327e-03 -5.20448695e-02
      -8.82945637e-02  4.67914594e-02  7.21499377e-02  2.27632238e-01
      -5.85330056e-02  1.04578970e-01 -1.96950460e-01  0.00000000e+00
      -3.34715605e+00 -2.71406772e-03 -3.30230987e-02  0.00000000e+00]
     [ 9.85451199e-02  1.87073651e-01  8.45670887e-03 -7.10549919e-04
      -4.31571713e-02  5.72866312e-02  5.75312467e-02  1.02785123e-01
      -5.58305863e-02  4.58712390e-02 -9.42585237e-02  0.00000000e+00
      -1.31440087e+00  2.05942993e-02 -2.08622116e-02  0.00000000e+00]
     [ 2.95868327e-03  8.45670887e-03  8.08639105e-02  1.18813265e-03
      -2.91208983e-04  2.17824319e-03  7.14044427e-03  1.04718750e-02
       5.90571818e-03  1.29296789e-02 -4.25165115e-03  0.00000000e+00
       3.33026593e-01 -4.95055271e-03  1.46303393e-02  0.00000000e+00]
     [-5.20448695e-02 -7.10549919e-04  1.18813265e-03  1.39268784e-01
       8.33789561e-02  1.41411082e-02 -1.33257231e-02 -4.34949737e-02
       2.47061701e-02 -6.92611446e-02  5.14624515e-02  0.00000000e+00
      -5.41252665e-01 -2.96567229e-02 -6.56967466e-03  0.00000000e+00]
     [-8.82945637e-02 -4.31571713e-02 -2.91208983e-04  8.33789561e-02
       2.43568194e-01 -1.77870447e-02 -2.89228762e-02 -8.22257685e-02
       4.88532190e-02 -8.05949982e-02  1.02156111e-01  0.00000000e+00
      -9.29108085e-01 -6.41358665e-02  2.18872672e-02  0.00000000e+00]
     [ 4.67914594e-02  5.72866312e-02  2.17824319e-03  1.41411082e-02
      -1.77870447e-02  1.20351849e-01  5.63431141e-02  4.76767347e-02
      -4.17360715e-02  1.63659449e-02 -4.33086000e-02  0.00000000e+00
      -6.49861967e-01 -5.66459714e-02 -5.21846498e-03  0.00000000e+00]
     [ 7.21499377e-02  5.75312467e-02  7.14044427e-03 -1.33257231e-02
      -2.89228762e-02  5.63431141e-02  2.44616546e-01  6.81894955e-02
      -1.70031101e-01  5.75545434e-02 -5.16255285e-02  0.00000000e+00
      -7.51785111e-01 -3.63428811e-03 -3.98839823e-02  0.00000000e+00]
     [ 2.27632238e-01  1.02785123e-01  1.04718750e-02 -4.34949737e-02
      -8.22257685e-02  4.76767347e-02  6.81894955e-02  2.48716768e-01
      -4.79562954e-02  9.38275344e-02 -1.84113968e-01  0.00000000e+00
      -2.81575790e+00  1.85441881e-02 -2.29705646e-02  0.00000000e+00]
     [-5.85330056e-02 -5.58305863e-02  5.90571818e-03  2.47061701e-02
       4.88532190e-02 -4.17360715e-02 -1.70031101e-01 -4.79562954e-02
       2.45571711e-01 -3.87074980e-02  6.20624585e-02  0.00000000e+00
       6.56373400e-01  3.18699111e-02  6.75255390e-02  0.00000000e+00]
     [ 1.04578970e-01  4.58712390e-02  1.29296789e-02 -6.92611446e-02
      -8.05949982e-02  1.63659449e-02  5.75545434e-02  9.38275344e-02
      -3.87074980e-02  1.81831889e-01 -9.20453354e-02  0.00000000e+00
      -3.97430372e-01  1.55505597e-02 -1.20677003e-02  0.00000000e+00]
     [-1.96950460e-01 -9.42585237e-02 -4.25165115e-03  5.14624515e-02
       1.02156111e-01 -4.33086000e-02 -5.16255285e-02 -1.84113968e-01
       6.20624585e-02 -9.20453354e-02  2.49159406e-01  0.00000000e+00
       1.73058510e+00 -4.15031043e-02  6.08510291e-02  0.00000000e+00]
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
       0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
       0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e-06
       0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
     [-3.34715605e+00 -1.31440087e+00  3.33026593e-01 -5.41252665e-01
      -9.29108085e-01 -6.49861967e-01 -7.51785111e-01 -2.81575790e+00
       6.56373400e-01 -3.97430372e-01  1.73058510e+00  0.00000000e+00
       2.24952581e+02 -2.76066116e-01 -2.83521066e-01  0.00000000e+00]
     [-2.71406772e-03  2.05942993e-02 -4.95055271e-03 -2.96567229e-02
      -6.41358665e-02 -5.66459714e-02 -3.63428811e-03  1.85441881e-02
       3.18699111e-02  1.55505597e-02 -4.15031043e-02  0.00000000e+00
      -2.76066116e-01  2.12470833e+00  1.71347366e-02  0.00000000e+00]
     [-3.30230987e-02 -2.08622116e-02  1.46303393e-02 -6.56967466e-03
       2.18872672e-02 -5.21846498e-03 -3.98839823e-02 -2.29705646e-02
       6.75255390e-02 -1.20677003e-02  6.08510291e-02  0.00000000e+00
      -2.83521066e-01  1.71347366e-02  2.47551932e-01  0.00000000e+00]
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
       0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
       0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
       0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e-06]]
    
    Component 3
    Weight: 0.007566735463525319
    Mean: [ 0.09095484  0.27286453  0.27286453  0.81809031  0.72713547  0.54532543
      0.45444881  0.27286453  0.54555119  0.18190969  0.5453106   3.27409027
     17.54504229  2.81852526  0.54555119  6.54500661]
    Covariance:
    [[ 8.26830606e-02  6.61364932e-02  6.61364932e-02 -7.44092769e-02
       2.48183511e-02  4.13548546e-02  4.96205238e-02  6.61364932e-02
      -4.96205238e-02 -1.65455674e-02 -4.95986403e-02 -2.49298379e-02
      -2.31483925e-01  1.07460851e-01  4.13343205e-02 -1.40525836e-01]
     [ 6.61364932e-02  1.98410480e-01  1.07454635e-01 -4.13181421e-02
       7.44550533e-02  3.31097195e-02  1.48861571e-01  1.98409480e-01
      -5.79067271e-02  1.32272986e-01 -1.48795921e-01  2.89029863e-01
      -5.77678653e-02  3.22382554e-01 -5.79067271e-02  1.24151558e-01]
     [ 6.61364932e-02  1.07454635e-01  1.98410480e-01 -4.13181421e-02
       7.44550533e-02  1.24064564e-01  5.79067271e-02  1.07454635e-01
       3.30481172e-02  4.13181421e-02 -5.78410767e-02 -1.65744358e-01
       5.78916045e-01  4.95180209e-02  3.30481172e-02 -3.30622664e-01]
     [-7.44092769e-02 -4.13181421e-02 -4.13181421e-02  1.48819554e-01
       4.13181421e-02  8.24513512e-03 -8.28620332e-03 -4.13181421e-02
       9.92410476e-02  3.30911348e-02  9.91972807e-02 -4.95869390e-01
       1.28156145e+00 -3.30120139e-02 -8.26686410e-02 -8.27677050e-02]
     [ 2.48183511e-02  7.44550533e-02  7.44550533e-02  4.13181421e-02
       1.98410480e-01  5.78451248e-02  1.24002961e-01  7.44550533e-02
      -3.30481172e-02  4.96367022e-02 -3.31137675e-02 -3.79984708e-01
       1.14922600e+00 -1.40472865e-01 -3.30481172e-02 -3.31967133e-02]
     [ 4.13548546e-02  3.31097195e-02  1.24064564e-01  8.24513512e-03
       5.78451248e-02  2.47946605e-01  2.47232241e-02  3.31097195e-02
       6.62316201e-02 -8.24513512e-03 -2.48260201e-02 -7.85026220e-01
       1.24841321e+00  1.90066819e-01  6.62316201e-02 -2.06973574e-01]
     [ 4.96205238e-02  1.48861571e-01  5.79067271e-02 -8.28620332e-03
       1.24002961e-01  2.47232241e-02  2.47926089e-01  1.48861571e-01
      -1.56970245e-01  9.92410476e-02 -6.62314755e-02 -1.23596949e-01
      -1.56738974e-01  1.73751239e-01 -1.56970245e-01  2.06458982e-01]
     [ 6.61364932e-02  1.98409480e-01  1.07454635e-01 -4.13181421e-02
       7.44550533e-02  3.31097195e-02  1.48861571e-01  1.98410480e-01
      -5.79067271e-02  1.32272986e-01 -1.48795921e-01  2.89029863e-01
      -5.77678653e-02  3.22382554e-01 -5.79067271e-02  1.24151558e-01]
     [-4.96205238e-02 -5.79067271e-02  3.30481172e-02  9.92410476e-02
      -3.30481172e-02  6.62316201e-02 -1.56970245e-01 -5.79067271e-02
       2.47926089e-01 -8.28620332e-03  6.62314755e-02 -6.04041805e-01
       1.33915195e+00  9.91132939e-02  6.60154003e-02 -3.88368670e-01]
     [-1.65455674e-02  1.32272986e-01  4.13181421e-02  3.30911348e-02
       4.96367022e-02 -8.24513512e-03  9.92410476e-02  1.32272986e-01
      -8.28620332e-03  1.48819554e-01 -9.91972807e-02  3.13959701e-01
       1.73716060e-01  2.14921702e-01 -9.92410476e-02  2.64677394e-01]
     [-4.95986403e-02 -1.48795921e-01 -5.78410767e-02  9.91972807e-02
      -3.31137675e-02 -2.48260201e-02 -6.62314755e-02 -1.48795921e-01
       6.62314755e-02 -9.91972807e-02  2.47947950e-01 -6.03182447e-01
       5.21283208e-01 -4.46164385e-01 -2.47233687e-02 -3.88790360e-01]
     [-2.49298379e-02  2.89029863e-01 -1.65744358e-01 -4.95869390e-01
      -3.79984708e-01 -7.85026220e-01 -1.23596949e-01  2.89029863e-01
      -6.04041805e-01  3.13959701e-01 -6.03182447e-01  8.38036777e+00
      -6.78781392e+00 -8.60710926e-01  1.23596949e-01  2.03561779e+00]
     [-2.31483925e-01 -5.77678653e-02  5.78916045e-01  1.28156145e+00
       1.14922600e+00  1.24841321e+00 -1.56738974e-01 -5.77678653e-02
       1.33915195e+00  1.73716060e-01  5.21283208e-01 -6.78781392e+00
       2.26193016e+01 -2.65816244e-01  1.56738974e-01 -3.29915523e+00]
     [ 1.07460851e-01  3.22382554e-01  4.95180209e-02 -3.30120139e-02
      -1.40472865e-01  1.90066819e-01  1.73751239e-01  3.22382554e-01
       9.91132939e-02  2.14921702e-01 -4.46164385e-01 -8.60710926e-01
      -2.65816244e-01  3.05803044e+00 -2.64706083e-01  2.81282623e-01]
     [ 4.13343205e-02 -5.79067271e-02  3.30481172e-02 -8.26686410e-02
      -3.30481172e-02  6.62316201e-02 -1.56970245e-01 -5.79067271e-02
       6.60154003e-02 -9.92410476e-02 -2.47233687e-02  1.23596949e-01
       1.56738974e-01 -2.64706083e-01  2.47926089e-01 -2.06458982e-01]
     [-1.40525836e-01  1.24151558e-01 -3.30622664e-01 -8.27677050e-02
      -3.31967133e-02 -2.06973574e-01  2.06458982e-01  1.24151558e-01
      -3.88368670e-01  2.64677394e-01 -3.88790360e-01  2.03561779e+00
      -3.29915523e+00  2.81282623e-01 -2.06458982e-01  2.43025403e+00]]
    
    Component 4
    Weight: 0.03165888586243596
    Mean: [8.47819934e-01 9.13042497e-01 2.39128557e-01 4.34784436e-02
     0.00000000e+00 9.56520855e-01 8.91304267e-01 9.34781981e-01
     8.69555478e-02 8.26080450e-01 2.17389598e-02 6.36952782e+00
     3.87390725e+01 2.63045241e+00 4.56518214e-01 1.63047218e+00]
    Covariance:
    [[ 1.29022294e-01  5.19848206e-02 -2.88262796e-02 -1.51224073e-02
       0.00000000e+00  3.68624859e-02  9.21544092e-02  5.52931367e-02
      -7.37226468e-02  1.25712977e-01 -1.84307234e-02  3.82352945e-01
      -9.09216359e-01  5.24427762e-02 -1.74828164e-02 -3.82352945e-01]
     [ 5.19848206e-02  7.93968957e-02 -9.44937526e-04 -3.96976667e-02
       0.00000000e+00  3.96983071e-02  3.40272355e-02  5.95468231e-02
      -1.41775224e-02  7.18338932e-02 -1.98485941e-02  4.23437213e-01
      -9.35712091e-01 -9.73547312e-02 -3.78035201e-03 -4.23437213e-01]
     [-2.88262796e-02 -9.44937526e-04  1.81947090e-01  1.13420223e-02
       0.00000000e+00  1.03971052e-02 -1.74856657e-02  1.55954908e-02
       2.26843648e-02 -4.53667079e-02 -5.19840608e-03  2.03301455e-02
       1.60586137e+00 -4.20643770e-02  2.12672166e-02 -2.03301455e-02]
     [-1.51224073e-02 -3.96976667e-02  1.13420223e-02  4.15890685e-02
       0.00000000e+00 -1.98485542e-02 -1.70130385e-02 -1.89033818e-02
       1.79582679e-02 -3.59166922e-02  2.07937836e-02 -1.89978871e-01
       3.80896791e-01  1.03023659e-01  1.89037482e-03  1.89978871e-01]
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
       1.00000000e-06  0.00000000e+00  0.00000000e+00  0.00000000e+00
       0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
       0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
     [ 3.68624859e-02  3.96983071e-02  1.03971052e-02 -1.98485542e-02
       0.00000000e+00  4.15897090e-02  3.87531475e-02  4.06435213e-02
      -1.79582069e-02  3.59172717e-02 -2.07937684e-02  2.33458802e-01
      -3.80904770e-01 -3.78090239e-02 -2.36288979e-02 -2.33458802e-01]
     [ 9.21544092e-02  3.40272355e-02 -1.74856657e-02 -1.70130385e-02
       0.00000000e+00  3.87531475e-02  9.68819707e-02  3.63902246e-02
      -7.75038508e-02  8.97914201e-02 -1.93760276e-02  1.70602306e-01
      -3.54447439e-01  3.30722250e-03 -3.73339658e-02 -1.70602306e-01]
     [ 5.52931367e-02  5.95468231e-02  1.55954908e-02 -1.89033818e-02
       0.00000000e+00  4.06435213e-02  3.63902246e-02  6.09656292e-02
      -1.60678912e-02  5.38753306e-02 -2.03211879e-02  3.50186291e-01
      -6.69182391e-01 -4.58425721e-02 -1.37047059e-02 -3.50186291e-01]
     [-7.37226468e-02 -1.41775224e-02  2.26843648e-02  1.79582679e-02
       0.00000000e+00 -1.79582069e-02 -7.75038508e-02 -1.60678912e-02
       7.93952805e-02 -7.18322780e-02  1.98486366e-02 -5.38714534e-02
       8.79097603e-02  1.03952532e-02  2.55197966e-02  5.38714534e-02]
     [ 1.25712977e-01  7.18338932e-02 -4.53667079e-02 -3.59166922e-02
       0.00000000e+00  3.59172717e-02  8.97914201e-02  5.38753306e-02
      -7.18322780e-02  1.43672540e-01 -1.79581297e-02  4.55603867e-01
      -1.17574606e+00  9.30617118e-04 -7.55846253e-03 -4.55603867e-01]
     [-1.84307234e-02 -1.98485941e-02 -5.19840608e-03  2.07937836e-02
       0.00000000e+00 -2.07937684e-02 -1.93760276e-02 -2.03211879e-02
       1.98486366e-02 -1.79581297e-02  2.12673774e-02 -1.16727949e-01
       1.14367092e-01  5.15114997e-02  1.18147287e-02  1.16727949e-01]
     [ 3.82352945e-01  4.23437213e-01  2.03301455e-02 -1.89978871e-01
       0.00000000e+00  2.33458802e-01  1.70602306e-01  3.50186291e-01
      -5.38714534e-02  4.55603867e-01 -1.16727949e-01  3.88531958e+00
      -4.33804483e+00 -6.46112483e-01  4.86930607e-02 -3.88531858e+00]
     [-9.09216359e-01 -9.35712091e-01  1.60586137e+00  3.80896791e-01
       0.00000000e+00 -3.80904770e-01 -3.54447439e-01 -6.69182391e-01
       8.79097603e-02 -1.17574606e+00  1.14367092e-01 -4.33804483e+00
       1.32105408e+02  9.68656886e-01 -3.37395893e-01  4.33804483e+00]
     [ 5.24427762e-02 -9.73547312e-02 -4.20643770e-02  1.03023659e-01
       0.00000000e+00 -3.78090239e-02  3.30722250e-03 -4.58425721e-02
       1.03952532e-02  9.30617118e-04  5.15114997e-02 -6.46112483e-01
       9.68656886e-01  2.92866363e+00  2.99137900e-01  6.46112483e-01]
     [-1.74828164e-02 -3.78035201e-03  2.12672166e-02  1.89037482e-03
       0.00000000e+00 -2.36288979e-02 -3.73339658e-02 -1.37047059e-02
       2.55197966e-02 -7.55846253e-03  1.18147287e-02  4.86930607e-02
      -3.37395893e-01  2.99137900e-01  2.48110334e-01 -4.86930607e-02]
     [-3.82352945e-01 -4.23437213e-01 -2.03301455e-02  1.89978871e-01
       0.00000000e+00 -2.33458802e-01 -1.70602306e-01 -3.50186291e-01
       5.38714534e-02 -4.55603867e-01  1.16727949e-01 -3.88531858e+00
       4.33804483e+00  6.46112483e-01 -4.86930607e-02  3.88531958e+00]]
    
    Component 5
    Weight: 0.08189507934760967
    Mean: [ 0.2857293   0.81506953  0.2857293   1.          1.          0.88233851
      0.6050313   0.81506953  0.42018169  0.1596682   0.43702069  3.81487535
     28.2433663   2.47897435  0.47058842  4.18512465]
    Covariance:
    [[ 2.04089068e-01  4.44362376e-02  2.04088068e-01  6.62944255e-32
       6.62944255e-32  1.68117074e-02  1.20085599e-02  4.44362376e-02
       3.96140595e-02  4.68199474e-02 -1.56201211e-02  5.06700820e-01
      -1.06118693e+00  8.16420260e-02  1.68074150e-02 -5.06700820e-01]
     [ 4.44362376e-02  1.50732192e-01  4.44362376e-02  2.16492733e-31
       2.16492733e-31  2.86729139e-02  4.46281221e-02  1.50731192e-01
      -4.83692894e-02  1.27198818e-02 -7.88927423e-02  4.61749758e-01
      -1.13948427e+00 -2.06789809e-02 -2.22399142e-02 -4.61749758e-01]
     [ 2.04088068e-01  4.44362376e-02  2.04089068e-01  6.62944255e-32
       6.62944255e-32  1.68117074e-02  1.20085599e-02  4.44362376e-02
       3.96140595e-02  4.68199474e-02 -1.56201211e-02  5.06700820e-01
      -1.06118693e+00  8.16420260e-02  1.68074150e-02 -5.06700820e-01]
     [ 5.96649829e-32  2.16285563e-31  5.96649829e-32  1.00000000e-06
       1.97215226e-31  2.33687850e-31  1.27616769e-31  2.16285563e-31
       7.29238680e-32  4.64060978e-32  1.25959408e-31  8.48568646e-31
       7.37194011e-30  5.36984846e-31  9.36408760e-32  3.18213242e-31]
     [ 5.96649829e-32  2.16285563e-31  5.96649829e-32  1.97215226e-31
       1.00000000e-06  2.33687850e-31  1.27616769e-31  2.16285563e-31
       7.29238680e-32  4.64060978e-32  1.25959408e-31  8.48568646e-31
       7.37194011e-30  5.36984846e-31  9.36408760e-32  3.18213242e-31]
     [ 1.68117074e-02  2.86729139e-02  1.68117074e-02  2.32859169e-31
       2.32859169e-31  1.03818262e-01  3.75730976e-02  2.86729139e-02
      -2.62026804e-02 -6.42464611e-03 -7.41537740e-03  6.22755831e-02
      -7.19210870e-01  8.99903431e-02 -1.18679379e-02 -6.22755831e-02]
     [ 1.20085599e-02  4.46281221e-02  1.20085599e-02  1.29274130e-31
       1.29274130e-31  3.75730976e-02  2.38969426e-01  4.46281221e-02
      -1.44973720e-01  2.10457314e-02 -2.06810009e-02  1.62255781e-01
      -1.55903918e-01  6.30941608e-02 -3.26340602e-02 -1.62255781e-01]
     [ 4.44362376e-02  1.50731192e-01  4.44362376e-02  2.16492733e-31
       2.16492733e-31  2.86729139e-02  4.46281221e-02  1.50732192e-01
      -4.83692894e-02  1.27198818e-02 -7.88927423e-02  4.61749758e-01
      -1.13948427e+00 -2.06789809e-02 -2.22399142e-02 -4.61749758e-01]
     [ 3.96140595e-02 -4.83692894e-02  3.96140595e-02  6.79517861e-32
       6.79517861e-32 -2.62026804e-02 -1.44973720e-01 -4.83692894e-02
       2.43630037e-01  8.54418365e-03  7.68928761e-02 -9.87606998e-02
      -2.70427200e-01 -6.67669332e-02  6.27887591e-02  9.87606998e-02]
     [ 4.68199474e-02  1.27198818e-02  4.68199474e-02  4.64060978e-32
       4.64060978e-32 -6.42464611e-03  2.10457314e-02  1.27198818e-02
       8.54418365e-03  1.34175266e-01  5.85554880e-03  2.14442193e-01
      -5.51470143e-01 -3.44508405e-02  8.89773566e-03 -2.14442193e-01]
     [-1.56201211e-02 -7.88927423e-02 -1.56201211e-02  1.30931490e-31
       1.30931490e-31 -7.41537740e-03 -2.06810009e-02 -7.88927423e-02
       7.68928761e-02  5.85554880e-03  2.46034606e-01 -3.89843272e-01
      -3.50348143e-01 -7.65304727e-03  1.28444649e-02  3.89843272e-01]
     [ 5.06700820e-01  4.61749758e-01  5.06700820e-01  8.22050876e-31
       8.22050876e-31  6.22755831e-02  1.62255781e-01  4.61749758e-01
      -9.87606998e-02  2.14442193e-01 -3.89843272e-01  5.02502763e+00
      -8.19769174e+00 -4.58125390e-02  7.86540349e-02 -5.02502663e+00]
     [-1.06118693e+00 -1.13948427e+00 -1.06118693e+00  7.42497565e-30
       7.42497565e-30 -7.19210870e-01 -1.55903918e-01 -1.13948427e+00
      -2.70427200e-01 -5.51470143e-01 -3.50348143e-01 -8.19769174e+00
       1.92413582e+02 -7.21901192e-01 -4.42181120e-01  8.19769174e+00]
     [ 8.16420260e-02 -2.06789809e-02  8.16420260e-02  5.10467076e-31
       5.10467076e-31  8.99903431e-02  6.30941608e-02 -2.06789809e-02
      -6.67669332e-02 -3.44508405e-02 -7.65304727e-03 -4.58125390e-02
      -7.21901192e-01  4.03096360e+00 -7.41294123e-02  4.58125390e-02]
     [ 1.68074150e-02 -2.22399142e-02  1.68074150e-02  9.28121957e-32
       9.28121957e-32 -1.18679379e-02 -3.26340602e-02 -2.22399142e-02
       6.27887591e-02  8.89773566e-03  1.28444649e-02  7.86540349e-02
      -4.42181120e-01 -7.41294123e-02  2.49135959e-01 -7.86540349e-02]
     [-5.06700820e-01 -4.61749758e-01 -5.06700820e-01  3.18213242e-31
       3.18213242e-31 -6.22755831e-02 -1.62255781e-01 -4.61749758e-01
       9.87606998e-02 -2.14442193e-01  3.89843272e-01 -5.02502663e+00
       8.19769174e+00  4.58125390e-02 -7.86540349e-02  5.02502763e+00]]
    
    Component 6
    Weight: 0.5616049050880546
    Mean: [ 0.62499086  1.          0.          1.          0.50736014  0.92156961
      0.605394    0.64828945  0.33823405  0.13847898  0.15931378  5.17278866
     25.08458637  2.61029239  0.43872511  2.82721134]
    Covariance:
    [[ 2.34378284e-01  1.26882942e-31  0.00000000e+00  1.26882942e-31
       3.97828456e-03 -6.11795553e-07  2.75615892e-03  1.73247931e-01
       3.06446029e-03  2.61953555e-02 -5.42271813e-02  6.56703374e-01
      -1.91190716e+00 -3.21675673e-02 -7.04622791e-03 -6.56703374e-01]
     [ 1.26882942e-31  1.00000000e-06  0.00000000e+00  1.97215226e-31
       1.45975803e-31  1.71050290e-31  9.71560811e-32  2.01079252e-31
       1.17940715e-31 -4.35027229e-33  4.61612226e-32  1.14847188e-30
       5.50647799e-30  4.96897768e-31  7.32295835e-32  7.77248649e-31]
     [ 0.00000000e+00  0.00000000e+00  1.00000000e-06  0.00000000e+00
       0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
       0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
       0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
     [ 1.26882942e-31  1.97215226e-31  0.00000000e+00  1.00000000e-06
       1.45975803e-31  1.71050290e-31  9.71560811e-32  2.01079252e-31
       1.17940715e-31 -4.35027229e-33  4.61612226e-32  1.14847188e-30
       5.50647799e-30  4.96897768e-31  7.32295835e-32  7.77248649e-31]
     [ 3.97828456e-03  1.45975803e-31  0.00000000e+00  1.45975803e-31
       2.49946828e-01 -5.55018634e-03 -1.30287444e-02 -3.84646820e-02
       2.20218433e-02 -1.87882693e-02  4.78475098e-02 -1.53844516e-01
      -1.92888882e+00  6.54100796e-03 -4.45153523e-03  1.53844516e-01]
     [-6.11795553e-07  1.71050290e-31  0.00000000e+00  1.71050290e-31
      -5.55018634e-03  7.22800670e-02  2.66482478e-02  6.72842741e-03
      -2.24912165e-02 -1.68290913e-04 -5.88704824e-03  1.11009560e-02
       1.29180688e-01  1.47775691e-02 -8.48230147e-03 -1.11009560e-02]
     [ 2.75615892e-03  9.71560811e-32  0.00000000e+00  9.71560811e-32
      -1.30287444e-02  2.66482478e-02  2.38893104e-01  1.31714727e-02
      -1.77804439e-01  1.05275219e-02 -1.55648964e-02  5.59288296e-02
       1.86554610e-01  1.90178072e-02 -1.80521168e-02 -5.59288296e-02]
     [ 1.73247931e-01  2.01079252e-31  0.00000000e+00  2.01079252e-31
      -3.84646820e-02  6.72842741e-03  1.31714727e-02  2.28011238e-01
      -8.48863644e-03  2.54205426e-02 -6.03875313e-02  6.66154901e-01
      -1.32072054e+00 -3.90264936e-02 -2.55620464e-03 -6.66154901e-01]
     [ 3.06446029e-03  1.17940715e-31  0.00000000e+00  1.17940715e-31
       2.20218433e-02 -2.24912165e-02 -1.77804439e-01 -8.48863644e-03
       2.23832776e-01 -5.17219079e-03  2.08693171e-02 -4.74137636e-02
      -4.83252557e-01 -1.28002691e-02  1.82743815e-02  4.74137636e-02]
     [ 2.61953555e-02 -4.35027229e-33  0.00000000e+00 -4.35027229e-33
      -1.87882693e-02 -1.68290913e-04  1.05275219e-02  2.54205426e-02
      -5.17219079e-03  1.19303549e-01 -1.59341915e-02  1.10873723e-01
       3.73043431e-02 -8.53443811e-03 -1.17349767e-02 -1.10873723e-01]
     [-5.42271813e-02  4.61612226e-32  0.00000000e+00  4.61612226e-32
       4.78475098e-02 -5.88704824e-03 -1.55648964e-02 -6.03875313e-02
       2.08693171e-02 -1.59341915e-02  1.33933898e-01 -3.52280695e-01
      -1.33584592e-01  6.93797660e-03  3.63431274e-03  3.52280695e-01]
     [ 6.56703374e-01  1.14847188e-30  0.00000000e+00  1.14847188e-30
      -1.53844516e-01  1.11009560e-02  5.59288296e-02  6.66154901e-01
      -4.74137636e-02  1.10873723e-01 -3.52280695e-01  5.23849656e+00
      -6.85884615e+00 -1.40999732e-01 -1.82128956e-02 -5.23849556e+00]
     [-1.91190716e+00  5.50647799e-30  0.00000000e+00  5.50647799e-30
      -1.92888882e+00  1.29180688e-01  1.86554610e-01 -1.32072054e+00
      -4.83252557e-01  3.73043431e-02 -1.33584592e-01 -6.85884615e+00
       1.88048369e+02 -7.43399121e-03  2.29255079e-02  6.85884615e+00]
     [-3.21675673e-02  4.96897768e-31  0.00000000e+00  4.96897768e-31
       6.54100796e-03  1.47775691e-02  1.90178072e-02 -3.90264936e-02
      -1.28002691e-02 -8.53443811e-03  6.93797660e-03 -1.40999732e-01
      -7.43399121e-03  3.19618657e+00 -2.26554786e-02  1.40999732e-01]
     [-7.04622791e-03  7.32295835e-32  0.00000000e+00  7.32295835e-32
      -4.45153523e-03 -8.48230147e-03 -1.80521168e-02 -2.55620464e-03
       1.82743815e-02 -1.17349767e-02  3.63431274e-03 -1.82128956e-02
       2.29255079e-02 -2.26554786e-02  2.46246388e-01  1.82128956e-02]
     [-6.56703374e-01  7.77248649e-31  0.00000000e+00  7.77248649e-31
       1.53844516e-01 -1.11009560e-02 -5.59288296e-02 -6.66154901e-01
       4.74137636e-02 -1.10873723e-01  3.52280695e-01 -5.23849556e+00
       6.85884615e+00  1.40999732e-01  1.82128956e-02  5.23849656e+00]]
    
    Component 7
    Weight: 0.013764584365596081
    Mean: [ 0.40000091  0.30000103  0.40000133  0.40000085  0.35000118  0.65000179
      0.45000119  0.60000152  0.30000089  0.20000059  0.15000033  4.0999989
     36.25000003  2.59999347  0.35000081  3.9000011 ]
    Covariance:
    [[ 2.40001182e-01  2.99997562e-02 -1.00004526e-02 -6.00006789e-02
      -9.00006427e-02 -1.00008383e-02 -3.00007134e-02  1.09999611e-01
       2.99998135e-02  7.00000233e-02 -6.00002684e-02  9.10002344e-01
      -2.25000818e+00  6.00034860e-02  5.99996768e-02 -9.10002344e-01]
     [ 2.99997562e-02  2.10001413e-01  1.80000221e-01 -1.20000521e-01
       4.49998748e-02  4.99953008e-03 -8.50005318e-02 -3.00006153e-02
       1.10000018e-01  3.99999130e-02 -4.50002536e-02  7.00001551e-02
      -6.25000440e-01 -3.29999341e-01 -5.00016723e-03 -7.00001551e-02]
     [-1.00004526e-02  1.80000221e-01  2.40001265e-01 -6.00004297e-02
       1.09999948e-01 -1.00006929e-02 -8.00006365e-02 -9.00009437e-02
       7.99998404e-02  6.99999422e-02  3.99999639e-02 -1.90000501e-01
      -1.49999044e-01 -3.89998864e-01 -4.00003518e-02  1.90000501e-01]
     [-6.00006789e-02 -1.20000521e-01 -6.00004297e-02  2.40001171e-01
       5.99999664e-02 -1.00008582e-02  1.19999693e-01 -4.00007810e-02
      -7.00004597e-02 -3.00002579e-02  9.00000697e-02 -3.40001331e-01
       9.99993123e-02  1.60003238e-01  5.99997482e-02  3.40001331e-01]
     [-9.00006427e-02  4.49998748e-02  1.09999948e-01  5.99999664e-02
       2.27501354e-01 -2.75006551e-02 -7.50036328e-03 -6.00007796e-02
       4.49997818e-02 -2.00002933e-02  4.75000025e-02 -6.85002014e-01
      -3.74987128e-02 -1.59998513e-01  2.74998878e-02  6.85002014e-01]
     [-1.00008383e-02  4.99953008e-03 -1.00006929e-02 -1.00008582e-02
      -2.75006551e-02  2.27500464e-01  1.07499466e-01 -4.00012785e-02
      -9.50008147e-02 -3.00004452e-02  2.49969990e-03 -1.15001006e-01
       2.83751048e+00 -1.39995934e-01 -2.75007825e-02  1.15001006e-01]
     [-3.00007134e-02 -8.50005318e-02 -8.00006365e-02  1.19999693e-01
      -7.50036328e-03  1.07499466e-01  2.47501119e-01 -2.00009181e-02
      -1.35000758e-01 -4.00003556e-02  3.24998548e-02 -2.95001752e-01
       1.68750708e+00 -1.99968780e-02 -7.50056337e-03  2.95001752e-01]
     [ 1.09999611e-01 -3.00006153e-02 -9.00009437e-02 -4.00007810e-02
      -6.00007796e-02 -4.00012785e-02 -2.00009181e-02  2.40000697e-01
       1.99996058e-02 -2.00003633e-02 -9.00004251e-02  8.40002373e-01
      -1.60000623e+00  2.40005315e-01  3.99994649e-02 -8.40002373e-01]
     [ 2.99998135e-02  1.10000018e-01  7.99998404e-02 -7.00004597e-02
       4.49997818e-02 -9.50008147e-02 -1.35000758e-01  1.99996058e-02
       2.10001355e-01 -1.00002077e-02 -4.50002321e-02 -2.30000351e-01
      -2.42500713e+00 -2.29998726e-01  4.49998909e-02  2.30000351e-01]
     [ 7.00000233e-02  3.99999130e-02  6.99999422e-02 -3.00002579e-02
      -2.00002933e-02 -3.00004452e-02 -4.00003556e-02 -2.00003633e-02
      -1.00002077e-02  1.60001355e-01  1.99999927e-02  3.30001192e-01
      -1.00000295e-01 -1.19999048e-01 -2.00002206e-02 -3.30001192e-01]
     [-6.00002684e-02 -4.50002536e-02  3.99999639e-02  9.00000697e-02
       4.75000025e-02  2.49969990e-03  3.24998548e-02 -9.00004251e-02
      -4.50002321e-02  1.99999927e-02  1.27501231e-01 -2.65000604e-01
       6.12501491e-01  6.00008869e-02 -5.25002375e-02  2.65000604e-01]
     [ 9.10002344e-01  7.00001551e-02 -1.90000501e-01 -3.40001331e-01
      -6.85002014e-01 -1.15001006e-01 -2.95001752e-01  8.40002373e-01
      -2.30000351e-01  3.30001192e-01 -2.65000604e-01  8.19002446e+00
      -8.42504070e+00  1.99000912e+00  1.14999948e-01 -8.19002346e+00]
     [-2.25000818e+00 -6.25000440e-01 -1.49999044e-01  9.99993123e-02
      -3.74987128e-02  2.83751048e+00  1.68750708e+00 -1.60000623e+00
      -2.42500713e+00 -1.00000295e-01  6.12501491e-01 -8.42504070e+00
       8.13877902e+01 -4.80001794e+00 -1.33750462e+00  8.42504070e+00]
     [ 6.00034860e-02 -3.29999341e-01 -3.89998864e-01  1.60003238e-01
      -1.59998513e-01 -1.39995934e-01 -1.99968780e-02  2.40005315e-01
      -2.29998726e-01 -1.19999048e-01  6.00008869e-02  1.99000912e+00
      -4.80001794e+00  3.23999274e+00 -1.09997622e-01 -1.99000912e+00]
     [ 5.99996768e-02 -5.00016723e-03 -4.00003518e-02  5.99997482e-02
       2.74998878e-02 -2.75007825e-02 -7.50056337e-03  3.99994649e-02
       4.49998909e-02 -2.00002206e-02 -5.25002375e-02  1.14999948e-01
      -1.33750462e+00 -1.09997622e-01  2.27501244e-01 -1.14999948e-01]
     [-9.10002344e-01 -7.00001551e-02  1.90000501e-01  3.40001331e-01
       6.85002014e-01  1.15001006e-01  2.95001752e-01 -8.40002373e-01
       2.30000351e-01 -3.30001192e-01  2.65000604e-01 -8.19002346e+00
       8.42504070e+00 -1.99000912e+00 -1.14999948e-01  8.19002446e+00]]
    
    Component 8
    Weight: 0.08878258821881387
    Mean: [ 0.42635321  1.          0.37210059  0.37210059  0.28682836  0.90697753
      0.5891401   0.65115528  0.27907809  0.29457118  0.13179235  5.06197505
     34.23245832  2.36434603  0.57364034  2.93802495]
    Covariance:
    [[ 2.44577150e-01  1.07779495e-31 -4.23679523e-02 -4.23679523e-02
      -8.35305819e-02  9.01062622e-04  4.63017751e-03  1.02219837e-01
       1.27960706e-02  1.39423739e-02 -3.29344631e-02  6.32485970e-01
      -5.64221003e-01 -1.78595706e-01 -4.30245548e-02 -6.32485970e-01]
     [ 1.07779495e-31  1.00000000e-06  7.03242093e-32  7.03242093e-32
       1.03957527e-31  1.76574917e-31  1.28418121e-31  9.78423782e-32
       6.72666350e-32  6.57378479e-32  2.56071849e-32  1.16187824e-30
       8.16983858e-30  4.64751297e-31  1.33004483e-31  8.31660215e-31]
     [-4.23679523e-02  7.18529965e-32  2.33642740e-01  2.33641740e-01
       7.15731521e-02  1.13580937e-02  2.88406701e-02  5.76488568e-03
      -3.06018690e-03 -3.20913502e-02  2.07373613e-02 -3.33184875e-01
       2.28547163e+00 -3.77707870e-03 -2.74051826e-02  3.33184875e-01]
     [-4.23679523e-02  7.18529965e-32  2.33641740e-01  2.33642740e-01
       7.15731521e-02  1.13580937e-02  2.88406701e-02  5.76488568e-03
      -3.06018690e-03 -3.20913502e-02  2.07373613e-02 -3.33184875e-01
       2.28547163e+00 -3.77707870e-03 -2.74051826e-02  3.33184875e-01]
     [-8.35305819e-02  1.03957527e-31  7.15731521e-02  7.15731521e-02
       2.04558851e-01  3.42586052e-03 -6.19263048e-03 -2.39802430e-02
       2.84876567e-02 -2.24763532e-02  3.97258206e-02 -3.74406436e-01
      -2.45057192e-01  1.74580366e-01  5.25162551e-02  3.74406436e-01]
     [ 9.01062622e-04  1.77339311e-31  1.13580937e-02  1.13580937e-02
       3.42586052e-03  8.43702883e-02  3.92995219e-02  2.18127057e-02
      -3.60544477e-02  4.14612217e-03 -3.24409457e-03  9.10356513e-02
       6.03831768e-02  5.71479767e-02 -9.01663321e-04 -9.10356513e-02]
     [ 4.63017751e-03  1.28418121e-31  2.88406701e-02  2.88406701e-02
      -6.19263048e-03  3.92995219e-02  2.42055043e-01  1.94735117e-02
      -1.25656707e-01 -1.85062148e-02 -1.25426927e-04  6.42715521e-02
       2.66153656e-01  2.03943726e-01 -5.11352477e-02 -6.42715521e-02]
     [ 1.02219837e-01  9.78423782e-32  5.76488568e-03  5.76488568e-03
      -2.39802430e-02  2.18127057e-02  1.94735117e-02  2.27153083e-01
      -3.42992699e-03  1.74890109e-02 -2.38022956e-02  5.33292372e-01
      -3.68412733e-01 -4.69609579e-03 -3.24463797e-02 -5.33292372e-01]
     [ 1.27960706e-02  6.87954222e-32 -3.06018690e-03 -3.06018690e-03
       2.84876567e-02 -3.60544477e-02 -1.25656707e-01 -3.42992699e-03
       2.01194510e-01  4.18216059e-02  1.98951785e-03 -3.28487532e-02
      -7.54895269e-01 -1.01666423e-01  3.37081420e-02  3.28487532e-02]
     [ 1.39423739e-02  6.26802735e-32 -3.20913502e-02 -3.20913502e-02
      -2.24763532e-02  4.14612217e-03 -1.85062148e-02  1.74890109e-02
       4.18216059e-02  2.07799998e-01 -6.28658154e-05  1.52285209e-01
      -9.83196536e-01 -6.08145371e-02 -2.16923300e-02 -1.52285209e-01]
     [-3.29344631e-02  2.75181689e-32  2.07373613e-02  2.07373613e-02
       3.97258206e-02 -3.24409457e-03 -1.25426927e-04 -2.38022956e-02
       1.98951785e-03 -6.28658154e-05  1.14424123e-01 -3.49299454e-01
       9.32900832e-02  6.82745175e-02  1.91896344e-03  3.49299454e-01]
     [ 6.32485970e-01  1.15576309e-30 -3.33184875e-01 -3.33184875e-01
      -3.74406436e-01  9.10356513e-02  6.42715521e-02  5.33292372e-01
      -3.28487532e-02  1.52285209e-01 -3.49299454e-01  5.42267123e+00
      -4.09918215e+00 -4.33481598e-01 -1.44082164e-01 -5.42267023e+00]
     [-5.64221003e-01  8.16983858e-30  2.28547163e+00  2.28547163e+00
      -2.45057192e-01  6.03831768e-02  2.66153656e-01 -3.68412733e-01
      -7.54895269e-01 -9.83196536e-01  9.32900832e-02 -4.09918215e+00
       1.40953542e+02 -4.78245734e+00 -9.24054218e-01  4.09918215e+00]
     [-1.78595706e-01  4.83096742e-31 -3.77707870e-03 -3.77707870e-03
       1.74580366e-01  5.71479767e-02  2.03943726e-01 -4.69609579e-03
      -1.01666423e-01 -6.08145371e-02  6.82745175e-02 -4.33481598e-01
      -4.78245734e+00  3.42538454e+00  8.04577497e-03  4.33481598e-01]
     [-4.30245548e-02  1.37590844e-31 -2.74051826e-02 -2.74051826e-02
       5.25162551e-02 -9.01663321e-04 -5.11352477e-02 -3.24463797e-02
       3.37081420e-02 -2.16923300e-02  1.91896344e-03 -1.44082164e-01
      -9.24054218e-01  8.04577497e-03  2.44578101e-01  1.44082164e-01]
     [-6.32485970e-01  8.31660215e-31  3.33184875e-01  3.33184875e-01
       3.74406436e-01 -9.10356513e-02 -6.42715521e-02 -5.33292372e-01
       3.28487532e-02 -1.52285209e-01  3.49299454e-01 -5.42267023e+00
       4.09918215e+00  4.33481598e-01  1.44082164e-01  5.42267123e+00]]
    
    


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Plot the means
plt.scatter(means[:, 0], means[:, 1], c='red', marker='x', label='Means')

# Plot the covariances as ellipses
for i in range(num_components):
    cov = covariances[i]
    v, w = np.linalg.eigh(cov)
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = Ellipse(means[i], v[0], v[1], 180. + angle, color='blue', alpha=0.5)
    plt.gca().add_patch(ell)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Gaussian Mixture Model Components')
plt.legend()
plt.show()

```


    
![png](output_44_0.png)
    


For members of segment 1, liking McDonald’s is not associated with
their perception of whether eating at McDonald’s is convenient, and whether food
served at McDonald’s is healthy. In contrast, perceiving McDonald’s as convenient
and healthy is important to segment 2 (component 2). Using the perception of
healthy as an example: if segment 2 is targeted, it is important for McDonald’s
to convince segment members that McDonald’s serves (at least some) healthy food
items. The health argument is unnecessary for members of segment 1. Instead, this
segment wants to hear about how good the food tastes, and how fast and cheap it is


```python
y = data.Gender
```


```python
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate pairwise distances
distances = pdist(MD_x.T)

# Perform hierarchical clustering
linkage_matrix = linkage(distances)

# Plot the dendrogram
dendrogram(linkage_matrix)

# Add labels to the x-axis
plt.xlabel('Samples')

# Add labels to the y-axis
plt.ylabel('Distance')

# Set the title of the plot
plt.title('Hierarchical Clustering Dendrogram')

# Show the plot
plt.show()

```


    
![png](output_47_0.png)
    



```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=2)
MD_pca = pca.fit_transform(MD_x)

# Plot the clusters with PCA projection
plt.scatter(MD_pca[:, 0], MD_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Get the projection axes
projection_axes = pca.components_
print(projection_axes)

```


    
![png](output_48_0.png)
    


    [[-0.47693349 -0.15533159 -0.00635636  0.11623168  0.3044427  -0.10849325
      -0.33718593 -0.47151394  0.32904173 -0.21371062  0.37475293]
     [ 0.36378978  0.016414    0.01880869 -0.03409395 -0.06383884 -0.0869722
      -0.61063276  0.3073178   0.60128596  0.07659344 -0.13965633]]
    


```python
from sklearn.cluster import KMeans

wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(MD_x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
```


    
![png](output_49_0.png)
    


After the market segmentation analysis is completed, and all strategic and tactical
marketing activities have been undertaken, the success of the market segmentation
strategy has to be evaluated, and the market must be carefully monitored on a
continuous basis. It is possible, for example, that members of segment 3 start earning
more money and the MCSUPERBUDGET line is no longer suitable for them. Changes
can occur within existing market segments.
