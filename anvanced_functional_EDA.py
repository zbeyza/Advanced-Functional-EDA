#############################################################
# ADVANCED FUNCTIONAL EDA
###################################################################
"""
Aim of the advanced functional EDA :
It aims to be able to process data in a scalable, that is, functional way,
and to gain insights about the data quickly.
the process of quickly analyzing data with general functions

1. chech the dataset
    - Getting an outline of the internal and external features of the data set,
     how many observations, how many variables, missing values, etc.
2. analysis of categorical variables
3. analysis of numerical variables
4. analysis of target variables
5. analysis of correlation

"""

###################
# 1. check the dataset
###################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape # satır, sütun
df.info()
df.columns
df.index
df.describe().T # sayısal değişkenlerin beitmsel istatistikeleri
df.isnull().values.any()
df.isnull().sum()

def check_df(dataframe, head=5):
    print("############### SHAPE ################")
    print(dataframe.shape)
    print("############## TYPES #################")
    print(dataframe.dtypes)
    print("############### HEAD #################")
    print(dataframe.head(head))
    print("############### TAIL #################")
    print(dataframe.tail(head))
    print("############### NA ###################")
    print(dataframe.isnull().sum())
    print("############# QUANTITIES #############")
    print(dataframe.describe([0, 0.05, 0.5, 0.95]).T)

check_df(df)

df = sns.load_dataset("flights")
check_df(df)

#####################################
# 2. analysis of categorical varibales
#####################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["embarked"].value_counts()
df["sex"].unique()
df["sex"].nunique()


#########
# type of the variablec and selection
#########

#######
# Category :
#######
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
cat_cols

"""
df["sex"].dtypes
Out[36]: dtype('O')

str(df["sex"].dtypes)
Out[37]: 'object'
"""
#######
# Numeric as category
#######

# find the varibales which types are integer of float, then check the number of the unique values,
# if their number of unique values are less than 10 then they are numerical but as categorical varibales.
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

########
# categoric but as cardinal variables
########
# variables with high cardinality : too many classes to have explainability

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

# categorical and as categorical variables:
cat_cols = cat_cols + num_but_cat

# remowing cat_but car variables from categorical variables.
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols] # not categorical varibales, numeric variables


def cat_summary(dataframe, col_name):
    """
    what is expected from the function:

    - find the given variable's value_counts
        df[" "].value_counts()
    - print the percentage of classes:
        100 * df[" "].value_counts() / len(df)
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################3")

cat_summary(df, "sex")

#cat_summary function for all category variables
for col in cat_cols:
    cat_summary(df, col)

# improving cat_summary function
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True)

for col in cat_cols:
    # countplot function don't work with bools so they were passed
    if df[col].dtypes == "bool":
        print("ksdnlsnsnnanlnflkn")
    else:
        cat_summary(df, col, plot=True)

# assume we want to change the bool type variable
df["adult_male"].astype(int) # true=1, false=0


for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df,col,plot=True)
    else:
        cat_summary(df, col, plot=True)


##################################################
# 3. analysis of numerical varibales
##################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.info()
df[["age","fare"]].describe().T # descriptive statistics

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object","category", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64","float64"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

# we know that age and fare are the numeric varibles. but how can we select the numerical varibles from dataset?

# select numeric varibales:
num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
num_summary(df,"age")

for col in num_cols:
    num_summary(df,col)

# adding visualization option to num_summary function
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
num_summary(df, "age", plot =True)

# for loop to call the function for all variables
for col in num_cols:
    num_summary(df, col, plot=True)

##########################################################
# capturing variables and generalizing operations
###########################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    give the name of teh categorical, numeric, categoric but as cardinal variables

    Parameters
    ----------
    dataframe : dataframe
        dataframe that the name of the variables are wanted
    cat_th = int, float
        threshold value for numeric but as categorical variables
    car_th = int, float
        threshold value for categorical but as cardinal variables

    Returns
    -------
    cat_cols: list
        list of categorical variables
    num_cols: list
        list of numeric varibales
    cat_but_car: list
        list of categorical but as cardinal varibles

    Notes
    -------
    cat_cols + num_cols + cat_but_car = sum of the number of varibales
    num_but_cat is in the cat_cols

    """
    # selection category type variables
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # selecting numeric variables
    num_cols = [col for col in df.columns if df[col].dtypes in ["float64", "int64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    # report:
    print(f"Observations: {dataframe.shape[0]}") # number of the observations
    print(f"Vriables: {dataframe.shape[1]}") # number of the varibales
    print(f"cat_cols: {len(cat_cols)}") # dimensions of the categprical varibales
    print(f"num_cols: {len(num_cols)}") # dimensions of the numeric variables
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


############
# BONUS:
############
df =sns.load_dataset("titanic")
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype("int64")

cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.info()

type(df.isnull())


###################################################
# Analysis of target varibale
###################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype("int64")

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    give the name of teh categorical, numeric, categoric but as cardinal variables

    Parameters
    ----------
    dataframe : dataframe
        dataframe that the name of the variables are wanted
    cat_th = int, float
        threshold value for numeric but as categorical variables
    car_th = int, float
        threshold value for categorical but as cardinal variables

    Returns
    -------
    cat_cols: list
        list of categorical variables
    num_cols: list
        list of numeric varibales
    cat_but_car: list
        list of categorical but as cardinal varibles

    Notes
    -------
    cat_cols + num_cols + cat_but_car = sum of the number of varibales
    num_but_cat is in the cat_cols

    """
    # selecting categorical variables
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # selecting numerical varibales
    num_cols = [col for col in df.columns if df[col].dtypes in ["float64", "int64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    #report:
    print(f"Observations: {dataframe.shape[0]}") # number of observations
    print(f"Vriables: {dataframe.shape[1]}") # number of variables
    print(f"cat_cols: {len(cat_cols)}") # dimensions of category type variables
    print(f"num_cols: {len(num_cols)}") # dimension of numeric variables
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# in this case, survived is the target variable.

df.head()
df["survived"].value_counts()
cat_summary(df, "survived")

# to understand that why the survived ones survived, is should be crossed with dependent variables.


################
# Analysis of target variable with categorical variables
################

df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

###############
# Analysis of target variable with numeric variables
###############

df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age":"mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")

target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)


############################################################
# analysis of correlation
############################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

corr = df[num_cols].corr()

sns.set(rc={"figure.figsize": (12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show(block=True)

###############
# dropping the high correlated variables
###############

# we don't care if the correlation is negative or positive and
# we take the absolute value to make the negatives positive
cor_matrix = df.corr().abs()

#           0         1         2         3
# 0  1.000000  0.117570  0.871754  0.817941
# 1  0.117570  1.000000  0.428440  0.366126
# 2  0.871754  0.428440  1.000000  0.962865
# 3  0.817941  0.366126  0.962865  1.000000


#     0        1         2         3
# 0 NaN  0.11757  0.871754  0.817941
# 1 NaN      NaN  0.428440  0.366126
# 2 NaN      NaN       NaN  0.962865
# 3 NaN      NaN       NaN       NaN

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
## A numpy array of 1s in the size of cor_matrix we created is created and converted to bool,
# and then numpy's triu function is used to convert it to the structure seen above (NaN below the diagonal).

# deletion of values with a correlation higher than 90%
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
cor_matrix[drop_list]
df.drop(drop_list, axis=1)


# İçlmelerin fonksiyonlaştırılması
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show(block=True)
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)


# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv

df = pd.read_csv("datasets/fraud_train_transaction.csv")
len(df.columns)
df.head()

drop_list = high_correlated_cols(df, plot=True)

len(df.drop(drop_list, axis=1).columns)



