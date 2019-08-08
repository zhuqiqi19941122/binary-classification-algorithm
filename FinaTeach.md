
import pandas as pd
import numpy as np


```python
train_data = pd.read_csv('train_set.csv')
print(train_data.shape)
train_data.head()
```

    (25317, 18)
    
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>43</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>291</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>9</td>
      <td>may</td>
      <td>150</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>42</td>
      <td>technician</td>
      <td>divorced</td>
      <td>primary</td>
      <td>no</td>
      <td>5076</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>7</td>
      <td>apr</td>
      <td>99</td>
      <td>1</td>
      <td>251</td>
      <td>2</td>
      <td>other</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>47</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>104</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>14</td>
      <td>jul</td>
      <td>77</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>28</td>
      <td>management</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>-994</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>18</td>
      <td>jul</td>
      <td>174</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>42</td>
      <td>technician</td>
      <td>divorced</td>
      <td>secondary</td>
      <td>no</td>
      <td>2974</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>21</td>
      <td>may</td>
      <td>187</td>
      <td>5</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_1 = train_data[train_data["y"] == 1]
y_0 = train_data[train_data["y"] == 0]
print(len(y_1), len(y_0))
```

    2961 22356
    


```python
pdays_neg = train_data[train_data['pdays'] == -1]
print(len(pdays_neg))
```

    20674
    


```python
train_job = train_data["job"].value_counts()
train_job
```




    blue-collar      5456
    management       5296
    technician       4241
    admin.           2909
    services         2342
    retired          1273
    self-employed     884
    entrepreneur      856
    unemployed        701
    housemaid         663
    student           533
    unknown           163
    Name: job, dtype: int64




```python
import seaborn as sns
sns.barplot(y=train_job.index, x=train_job.values)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x24c362e82b0>




![png](https://github.com/zhuqiqi19941122/binary-classification-algorithm/blob/master/fig/output_5_1.png)



```python
train_job_y = train_data.groupby(["job","y"])
train_job_y_counts = train_job_y.size().unstack()
#type(train_job_y_counts)
# train_job_y_counts.sum(1).nlargest(10)
train_job_y_counts = train_job_y_counts.stack()
train_job_y_counts.name = "total"
train_job_y_counts = train_job_y_counts.reset_index()
train_job_y_counts
```


</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>job</th>
      <th>y</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>admin.</td>
      <td>0</td>
      <td>2568</td>
    </tr>
    <tr>
      <th>1</th>
      <td>admin.</td>
      <td>1</td>
      <td>341</td>
    </tr>
    <tr>
      <th>2</th>
      <td>blue-collar</td>
      <td>0</td>
      <td>5069</td>
    </tr>
    <tr>
      <th>3</th>
      <td>blue-collar</td>
      <td>1</td>
      <td>387</td>
    </tr>
    <tr>
      <th>4</th>
      <td>entrepreneur</td>
      <td>0</td>
      <td>789</td>
    </tr>
    <tr>
      <th>5</th>
      <td>entrepreneur</td>
      <td>1</td>
      <td>67</td>
    </tr>
    <tr>
      <th>6</th>
      <td>housemaid</td>
      <td>0</td>
      <td>605</td>
    </tr>
    <tr>
      <th>7</th>
      <td>housemaid</td>
      <td>1</td>
      <td>58</td>
    </tr>
    <tr>
      <th>8</th>
      <td>management</td>
      <td>0</td>
      <td>4560</td>
    </tr>
    <tr>
      <th>9</th>
      <td>management</td>
      <td>1</td>
      <td>736</td>
    </tr>
    <tr>
      <th>10</th>
      <td>retired</td>
      <td>0</td>
      <td>975</td>
    </tr>
    <tr>
      <th>11</th>
      <td>retired</td>
      <td>1</td>
      <td>298</td>
    </tr>
    <tr>
      <th>12</th>
      <td>self-employed</td>
      <td>0</td>
      <td>779</td>
    </tr>
    <tr>
      <th>13</th>
      <td>self-employed</td>
      <td>1</td>
      <td>105</td>
    </tr>
    <tr>
      <th>14</th>
      <td>services</td>
      <td>0</td>
      <td>2131</td>
    </tr>
    <tr>
      <th>15</th>
      <td>services</td>
      <td>1</td>
      <td>211</td>
    </tr>
    <tr>
      <th>16</th>
      <td>student</td>
      <td>0</td>
      <td>390</td>
    </tr>
    <tr>
      <th>17</th>
      <td>student</td>
      <td>1</td>
      <td>143</td>
    </tr>
    <tr>
      <th>18</th>
      <td>technician</td>
      <td>0</td>
      <td>3760</td>
    </tr>
    <tr>
      <th>19</th>
      <td>technician</td>
      <td>1</td>
      <td>481</td>
    </tr>
    <tr>
      <th>20</th>
      <td>unemployed</td>
      <td>0</td>
      <td>587</td>
    </tr>
    <tr>
      <th>21</th>
      <td>unemployed</td>
      <td>1</td>
      <td>114</td>
    </tr>
    <tr>
      <th>22</th>
      <td>unknown</td>
      <td>0</td>
      <td>143</td>
    </tr>
    <tr>
      <th>23</th>
      <td>unknown</td>
      <td>1</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>




```python
def normal_total(group):
    group['norm'] = group.total/group.total.sum()
    return group
```


```python
results = train_job_y_counts.groupby('job').apply(normal_total)
```


```python
sns.barplot(x='total',y='job',hue='y',data=train_job_y_counts)
```




![png](https://github.com/zhuqiqi19941122/binary-classification-algorithm/blob/master/fig/output_9_1.png)



```python
train_marital_y = train_data.groupby(["marital","y"]).size()
train_marital_y.name = 'total'
train_marital_y = train_marital_y.reset_index()
train_marital_y = train_marital_y.groupby('marital').apply(normal_total)
sns.barplot(x='marital',y='norm',hue='y',data=train_marital_y)
```



![png](https://github.com/zhuqiqi19941122/binary-classification-algorithm/blob/master/fig/output_10_1.png)



```python
train_education_y = train_data.groupby(["education","y"]).size().unstack()
train_education_y.plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x24c36563240>




![png](https://github.com/zhuqiqi19941122/binary-classification-algorithm/blob/master/fig/output_11_1.png)



```python
train_poutcome_y = train_data.groupby(["poutcome","y"]).size().unstack()
train_poutcome_y.plot.bar()
```




![png](https://github.com/zhuqiqi19941122/binary-classification-algorithm/blob/master/fig/output_12_1.png)


train_contact_y = train_data.groupby(["contact","y"]).size().unstack()
train_contact_y.plot.bar()
