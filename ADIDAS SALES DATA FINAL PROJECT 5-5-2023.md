```python
import pandas as pd
```


```python
#Load the sales data from th excel file
df = pd.read_excel(r"/Users/mariamasoumahoro/Desktop/Adidas US Sales Datasets2.xlsx")
```


```python
#Print first few rows of the data
print(df.head())
```

          Retailer  Retailer ID                 Date  Year     Region     State  \
    0  Foot Locker      1185732  2020-01-01 00:00:00  2020  Northeast  New York   
    1  Foot Locker      1185732  2020-02-01 00:00:00  2020  Northeast  New York   
    2  Foot Locker      1185732  2020-03-01 00:00:00  2020  Northeast  New York   
    3  Foot Locker      1185732  2020-04-01 00:00:00  2020  Northeast  New York   
    4  Foot Locker      1185732  2020-05-01 00:00:00  2020  Northeast  New York   
    
           City                    Product  Price per Unit  Units Sold  \
    0  New York      Men's Street Footwear              50        1200   
    1  New York    Men's Athletic Footwear              50        1000   
    2  New York    Women's Street Footwear              40        1000   
    3  New York  Women's Athletic Footwear              45         850   
    4  New York              Men's Apparel              60         900   
    
       Total Sales  Operating Profit  Operating Margin Sales Method  
    0       600000            300000              0.50     In-store  
    1       500000            150000              0.30     In-store  
    2       400000            140000              0.35     In-store  
    3       382500            133875              0.35     In-store  
    4       540000            162000              0.30     In-store  



```python
#DATA PREPARATION
#Remove duplicate

df.drop_duplicates(inplace=True)

#datatypes of attributes

df.info()

#Checking unique values in dataset

df.apply(lambda x: len(x.unique()))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 9648 entries, 0 to 9647
    Data columns (total 14 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   Retailer          9648 non-null   object 
     1   Retailer ID       9648 non-null   int64  
     2   Date              9648 non-null   object 
     3   Year              9648 non-null   int64  
     4   Region            9648 non-null   object 
     5   State             9648 non-null   object 
     6   City              9648 non-null   object 
     7   Product           9648 non-null   object 
     8   Price per Unit    9648 non-null   int64  
     9   Units Sold        9648 non-null   int64  
     10  Total Sales       9648 non-null   int64  
     11  Operating Profit  9648 non-null   int64  
     12  Operating Margin  9648 non-null   float64
     13  Sales Method      9648 non-null   object 
    dtypes: float64(1), int64(6), object(7)
    memory usage: 1.1+ MB





    Retailer               6
    Retailer ID            4
    Date                 724
    Year                   2
    Region                 5
    State                 50
    City                  52
    Product                6
    Price per Unit        94
    Units Sold           361
    Total Sales         3138
    Operating Profit    4187
    Operating Margin      66
    Sales Method           3
    dtype: int64




```python
#Check for null values

df.isnull().sum()
```




    Retailer            0
    Retailer ID         0
    Date                0
    Year                0
    Region              0
    State               0
    City                0
    Product             0
    Price per Unit      0
    Units Sold          0
    Total Sales         0
    Operating Profit    0
    Operating Margin    0
    Sales Method        0
    dtype: int64




```python
#Change date format

from datetime import datetime

df['Date']= pd.to_datetime(df['Date'])
```


```python
#Check for categorical attributes

cat_col = []
for x in df.dtypes.index:
    if df.dtypes [x] =='object':
        cat_col.append(x)
    
cat_col
```




    ['Retailer', 'Region', 'State', 'City', 'Product', 'Sales Method']




```python
#print the categorical columns

for col in cat_col:
    print(col)
    print(df[col].value_counts())
    print()
```

    Retailer
    Foot Locker      2637
    West Gear        2374
    Sports Direct    2032
    Kohl's           1030
    Amazon            949
    Walmart           626
    Name: Retailer, dtype: int64
    
    Region
    West         2448
    Northeast    2376
    Midwest      1872
    South        1728
    Southeast    1224
    Name: Region, dtype: int64
    
    State
    California        432
    Texas             432
    New York          360
    Florida           360
    Mississippi       216
    Oregon            216
    Louisiana         216
    Idaho             216
    New Mexico        216
    Georgia           216
    Arkansas          216
    Virginia          216
    Oklahoma          216
    Connecticut       216
    Rhode Island      216
    Massachusetts     216
    Vermont           216
    Utah              216
    Arizona           216
    New Hampshire     216
    Pennsylvania      216
    Nevada            216
    Alabama           216
    Tennessee         216
    South Dakota      144
    Illinois          144
    Colorado          144
    New Jersey        144
    Delaware          144
    Maryland          144
    West Virginia     144
    Indiana           144
    Wisconsin         144
    Iowa              144
    North Dakota      144
    Michigan          144
    Kansas            144
    Missouri          144
    Minnesota         144
    Montana           144
    Kentucky          144
    Ohio              144
    North Carolina    144
    South Carolina    144
    Nebraska          144
    Maine             144
    Alaska            144
    Hawaii            144
    Wyoming           144
    Washington        144
    Name: State, dtype: int64
    
    City
    Portland          360
    Charleston        288
    Orlando           216
    Salt Lake City    216
    Houston           216
    Boise             216
    Phoenix           216
    Albuquerque       216
    Atlanta           216
    New York          216
    Jackson           216
    Little Rock       216
    Oklahoma City     216
    Hartford          216
    Providence        216
    Boston            216
    Burlington        216
    Richmond          216
    New Orleans       216
    Manchester        216
    Dallas            216
    Philadelphia      216
    Knoxville         216
    Birmingham        216
    Las Vegas         216
    Los Angeles       216
    San Francisco     216
    Chicago           144
    Newark            144
    Baltimore         144
    Indianapolis      144
    Milwaukee         144
    Des Moines        144
    Fargo             144
    Sioux Falls       144
    Wichita           144
    Wilmington        144
    Honolulu          144
    Albany            144
    Louisville        144
    Columbus          144
    Charlotte         144
    Seattle           144
    Miami             144
    Minneapolis       144
    Billings          144
    Omaha             144
    St. Louis         144
    Detroit           144
    Anchorage         144
    Cheyenne          144
    Denver            144
    Name: City, dtype: int64
    
    Product
    Men's Street Footwear        1610
    Men's Athletic Footwear      1610
    Women's Street Footwear      1608
    Women's Apparel              1608
    Women's Athletic Footwear    1606
    Men's Apparel                1606
    Name: Product, dtype: int64
    
    Sales Method
    Online      4889
    Outlet      3019
    In-store    1740
    Name: Sales Method, dtype: int64
    



```python
#Import librariesimport matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
sns.distplot(df['Price per Unit'])
```




    <AxesSubplot:xlabel='Price per Unit'>




![png](output_9_1.png)



```python
sns.distplot(df['Total Sales'])
```




    <AxesSubplot:xlabel='Total Sales'>




![png](output_10_1.png)



```python
sns.countplot(df["Product"])
```




    <AxesSubplot:xlabel='Product', ylabel='count'>




![png](output_11_1.png)



```python
#Encode categorical variables 
df = pd.get_dummies(df, columns=['Retailer', 'Region', 'State', 'City', 'Product', 'Sales Method'])
```


```python
#Normalize the numerical variables

numerical_cols = ['Price per Unit', 'Units Sold','Total Sales','Operating Profit','Operating Margin']
df[numerical_cols] = (df[numerical_cols]-df[numerical_cols].mean())/df[numerical_cols].std()
```


```python
#Create new features based on existings ones

df['Revenue']  = df['Price per Unit'] * df['Units Sold']
df['Month'] = pd.to_datetime(df['Date']).dt.month
```


```python
print(df.head())
```

       Retailer ID       Date  Year  Price per Unit  Units Sold  Total Sales  \
    0      1185732 2020-01-01  2020        0.325280    4.401685     3.570609   
    1      1185732 2020-02-01  2020        0.325280    3.468205     2.865967   
    2      1185732 2020-03-01  2020       -0.354742    3.468205     2.161324   
    3      1185732 2020-04-01  2020       -0.014731    2.768095     2.038012   
    4      1185732 2020-05-01  2020        1.005303    3.001465     3.147823   
    
       Operating Profit  Operating Margin  Retailer_Amazon  Retailer_Foot Locker  \
    0          4.900524          0.792292                0                     1   
    1          2.132645         -1.265376                0                     1   
    2          1.948120         -0.750959                0                     1   
    3          1.835098         -0.750959                0                     1   
    4          2.354076         -1.265376                0                     1   
    
       ...  Product_Men's Athletic Footwear  Product_Men's Street Footwear  \
    0  ...                                0                              1   
    1  ...                                1                              0   
    2  ...                                0                              0   
    3  ...                                0                              0   
    4  ...                                0                              0   
    
       Product_Women's Apparel  Product_Women's Athletic Footwear  \
    0                        0                                  0   
    1                        0                                  0   
    2                        0                                  0   
    3                        0                                  1   
    4                        0                                  0   
    
       Product_Women's Street Footwear  Sales Method_In-store  \
    0                                0                      1   
    1                                0                      1   
    2                                1                      1   
    3                                0                      1   
    4                                0                      1   
    
       Sales Method_Online  Sales Method_Outlet   Revenue  Month  
    0                    0                    0  1.431781      1  
    1                    0                    0  1.128138      2  
    2                    0                    0 -1.230319      3  
    3                    0                    0 -0.040777      4  
    4                    0                    0  3.017381      5  
    
    [5 rows x 132 columns]



```python
#Model selection\

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```


```python
#SPLIT the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Operating Margin','Operating Profit','Total Sales','Date'], axis=1), df['Total Sales'], test_size=0.2, random_state=42)


```


```python
# Train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_mse = mean_squared_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)
```


```python
# Train and evaluate decision tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
dt_mae = mean_absolute_error(y_test, dt_preds)
dt_mse = mean_squared_error(y_test, dt_preds)
dt_r2 = r2_score(y_test, dt_preds)

```


```python
# Train and evaluate neural network
nn = MLPRegressor(hidden_layer_sizes=(50, 50))
nn.fit(X_train, y_train)
nn_preds = nn.predict(X_test)
nn_mae = mean_absolute_error(y_test, nn_preds)
nn_mse = mean_squared_error(y_test, nn_preds)
nn_r2 = r2_score(y_test, nn_preds)
```


```python
# Print the evaluation metrics for each model
print('Linear Regression - MAE:', lr_mae, 'MSE:', lr_mse, 'R-squared:', lr_r2)
print('Decision Tree - MAE:', dt_mae, 'MSE:', dt_mse, 'R-squared:', dt_r2)
print('Neural Network - MAE:', nn_mae, 'MSE:', nn_mse, 'R-squared:', nn_r2)
```

    Linear Regression - MAE: 0.09029739759911155 MSE: 0.015159928809056372 R-squared: 0.9850307589458736
    Decision Tree - MAE: 0.011971704978220601 MSE: 0.00402578264563253 R-squared: 0.9960248552870518
    Neural Network - MAE: 5.39849352064958 MSE: 54.40090020781986 R-squared: -52.7166235428399



```python
#Model Training
# Train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Print the coefficients and intercept of the linear regression model
print('Coefficients:', lr.coef_)
print('Intercept:', lr.intercept_)
```

    Coefficients: [-2.00601004e-06 -6.16698603e-02  1.36376381e-01  8.96415109e-01
      3.58644880e-02  3.04965419e-02 -9.45377182e-03  2.27240830e-02
     -4.37889771e-02 -3.58423640e-02  1.01822565e-01  1.21421989e-01
     -4.14087714e-02 -1.30471248e-01 -5.13645345e-02 -1.55926394e-02
      1.45469093e-01  2.87971962e-03  5.58609813e-02 -1.46692277e-01
     -7.85642655e-03  7.60271668e-03  9.73046887e-03 -8.57502021e-02
      7.97495930e-03  4.97569713e-02 -1.60352398e-02  7.58322011e-02
      3.56034208e-02  1.62616901e-02  2.13354162e-02  1.40489226e-01
     -3.14413788e-02  1.05706630e-01  5.39576107e-02  2.18352245e-02
     -1.10085424e-01  3.29033399e-02 -6.25041751e-04 -5.92160517e-02
     -1.18261974e-02  5.83262923e-02 -1.03729065e-02  1.91440685e-02
      2.51504706e-02 -4.74005149e-03 -3.19521245e-01 -5.32593842e-02
      3.35118915e-02 -8.90031689e-02  8.02989110e-02 -4.95995006e-02
      8.06604237e-02  3.02871525e-02 -1.44783434e-01  6.18441157e-02
     -2.18318806e-02 -1.08077723e-01  2.46833901e-02  1.01303206e-02
      4.85758727e-03 -1.61932735e-02  7.67381484e-02  3.63350389e-02
     -2.26640330e-02  7.83746737e-02 -4.74005149e-03  1.45469093e-01
      7.97495930e-03  5.39576107e-02 -1.18261974e-02 -1.55926394e-02
     -1.60352398e-02  2.18352245e-02  1.01303206e-02 -6.80452861e-02
     -5.32593842e-02 -2.26640330e-02  7.58322011e-02 -8.90031689e-02
      3.94254323e-02 -7.85642655e-03  1.62616901e-02 -1.10085424e-01
      3.35118915e-02  7.60271668e-03  4.97569713e-02 -1.47503155e-01
      3.56034208e-02 -6.25041751e-04 -2.18318806e-02 -1.03729065e-02
      5.58609813e-02  3.87520020e-02  1.40489226e-01  1.91440685e-02
     -2.09289274e-01  3.63350389e-02  3.29033399e-02 -3.14413788e-02
     -3.97895919e-01  2.51504706e-02  8.02989110e-02  5.83262923e-02
      1.23539072e-01  8.06604237e-02  2.87971962e-03  5.61071290e-02
      3.02871525e-02  4.85758727e-03  2.46833901e-02 -1.85444279e-01
     -1.61932735e-02  6.18441157e-02 -5.92160517e-02  2.13354162e-02
      9.73046887e-03  1.56759373e-01 -3.38348005e-02 -1.96974723e-01
     -1.85120346e-04  8.00852229e-02 -5.84995177e-03  5.81939932e-02
     -4.85225786e-03 -5.33417353e-02  2.65984998e-01 -1.77484531e-03]
    Intercept: 126.93197704664149



```python
#Model Evaluation
lr_preds = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_mse = mean_squared_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

# Print the evaluation metrics for the model
print('Linear Regression - MAE:', lr_mae, 'MSE:', lr_mse, 'R-squared:', lr_r2)
```

    Linear Regression - MAE: 0.09029739759911155 MSE: 0.015159928809056372 R-squared: 0.9850307589458736



```python

```
