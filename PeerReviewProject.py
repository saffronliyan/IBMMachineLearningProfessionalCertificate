import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

file = '\\data\\Ames_Housing_Sales.csv'

data = pd.read_csv(file, header = 0)

cat_cols = data.columns[data.dtypes == object]

data2 = pd.get_dummies(data,columns = cat_cols,drop_first = True)

X = data2.drop('SalePrice', axis = 1)
y = data2['SalePrice']

ss = StandardScaler()
pf = PolynomialFeatures()

X_train,X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3)


###Linear
#coef_df = pd.DataFrame()
r2_df = pd.DataFrame()

linear = LinearRegression()

X_train_pf = pf.fit_transform(X_train)
X_train_ss = ss.fit_transform(X_train)
#X_train_ss = X_train_pf

X_test_pf = pf.fit_transform(X_test)
X_test_ss = ss.fit_transform(X_test)
#X_test_ss=X_test_pf

fit1 = linear.fit(X_train_ss,y_train)

linear_coef = fit1.coef_
coef_df = pd.DataFrame(zip(X_train.columns.to_list(),linear_coef))

pred1 = fit1.predict(X_test_ss)

r2_1 = r2_score(y_test, pred1)
r2_df['Linear']= pd.Series(data= r2_1)

###Ridge

ridge = Ridge()

fit2 = ridge.fit(X_train_ss,y_train)

ridge_coef =fit2.coef_
#coef_df['Ridge'] = ridge_coef



pred2 = fit2.predict(X_test_ss)
r2_2 = r2_score(y_test, pred2)
r2_df['Ridge']= pd.Series(data= r2_2)

###Lasso

lasso = Lasso()

fit3 = lasso.fit(X_train_ss,y_train)

lasso_coef =fit3.coef_
#coef_df['Lasso'] = lasso_coef



pred3 = fit3.predict(X_test_ss)
r2_3 = r2_score(y_test, pred3)
r2_df['Lasso']= pd.Series(data= r2_3)

print(coef_df)
print(r2_df)

print('Linear non-zero features',len(abs(linear_coef)>0))
print('Ridge non-zero features',len(abs(ridge_coef)>0))
print('Lasso non-zero features',len(abs(lasso_coef)>0))

print('Linear sum of coefs',sum(abs(linear_coef)))
print('Ridge sum of coefs',sum(abs(ridge_coef)))
print('Lasso sum pf coefs',sum(abs(lasso_coef)))
