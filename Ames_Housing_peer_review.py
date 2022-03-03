import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import statistics 
import math
from scipy import stats

path='data\\Ames_Housing_Data.tsv'

df = pd.read_csv(path,sep = '\t')

###breif description of data
###housing data size is 2930 x 82
###has misssing values
###has categorical fields and numeric fields
df.shape
df.info()
df.describe()

###cleaning and explore data
###according to the author get rid of living area above 4000
df1 = df.loc[df['Gr Liv Area'] <= 4000,:]

###subseting data

smaller_df= df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond', 
                      'Year Built', 'Year Remod/Add', 'Gr Liv Area', 
                      'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 
                      'Garage Cars','SalePrice']]

#fill the missing with 0; better be mean
smaller_df2 = smaller_df.fillna(0)

#pair plot
#sns.pairplot(smaller_df2,plot_kws=dict(alpha = 0.1,edgecolor = 'none'))

###hypothesis testing : Is sale price normally distributed
###null: yes with a fixed mean and variance

sale = df1['SalePrice']
mu = sale.mean()
var = statistics.variance(sale)
sigma = math.sqrt(var)
plt.plot(sale,stats.norm.pdf(sale,mu,sigma))

##confirmed by skew
sale.skew()
