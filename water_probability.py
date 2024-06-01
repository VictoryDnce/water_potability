import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",500)
warnings.filterwarnings("ignore")
matplotlib.use("Qt5Agg")

"""
pH: pH of the water (measured in pH units).
Hardness: Hardness of the water (measured in mg/L).
Solids: Total dissolved solids in the water (measured in ppm).
Chloramines: Amount of chloramines in the water (measured in ppm).
Sulfate: Amount of sulfate in the water (measured in mg/L).
Conductivity: Conductivity of the water (measured in μS/cm).
Organic_carbon: Amount of organic carbon in the water (measured in ppm).
Trihalomethanes: Amount of trihalomethanes in the water (measured in μg/L).
Turbidity: Turbidity of the water (measured in NTU).
Potability: Potability of the water (1 indicates potable, 0 indicates non-potable).

"""
dataset = pd.read_csv("free_work/water_potability/water_potability.csv")
df = dataset.copy()


#-------------------------------------- Data Preprocessing --------------------------------------

df.head()
df.shape
df.info()
df.isnull().sum()
df.describe().T

# ------------------------------- EDA(Exploratory Data Analysis) --------------------------------

# Checking class distributions for the target variable

fig, ax = plt.subplots(1,2, figsize=(10,5))

# Count
sns.countplot(x="Potability", data=df,ax=ax[0], palette=['#005b96',"#c6e2ff"])

# Pie
ax[1] = plt.pie(df["Potability"].value_counts(),
            labels=['Non-potable', 'Potable'],
            autopct='%1.2f%%',
            shadow=True,
            explode=(0.05, 0),
            startangle=60,
            colors=['#005b96',"#c6e2ff"]
                )

fig.suptitle('Distribution of the Potability', fontsize=24)


#################################
# Checking for outliers
#################################

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 12))
axes = axes.flatten()

for i, col in enumerate(df):
    sns.boxplot(y=col, data=df, ax=axes[i],color="orange")
    axes[i].set_title(f'Box Plot of {col}', fontsize=14)
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()

# There are some outliers in the dataset, so I will trim the outliers from the dataset

# -------------------------------------- Handling Outlier Values --------------------------------------

def handling_outlier(data,variable):
    quartile1 = data[variable].quantile(0.2) # Range (%20-%80)
    quartile3 = data[variable].quantile(0.8)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    data.loc[data[variable] < low_limit, variable] = low_limit
    data.loc[data[variable] > up_limit, variable] = up_limit

for col in df.columns[:-1]:
    handling_outlier(df, col)

#################################
# Rechecking outliers
#################################

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 12))
axes = axes.flatten()

for i, col in enumerate(df):
    sns.boxplot(y=col, data=df, ax=axes[i],color="orange")
    axes[i].set_title(f'Box Plot of {col}', fontsize=14)
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()

#################################
# Checking for missing values
#################################
df.isnull().sum()

plt.pie(df[['ph', 'Sulfate', 'Trihalomethanes']].isnull().sum(),
            labels=df.columns[df.isnull().any()],
            autopct='%1.2f%%',
            shadow=True,
            explode=(0.03, 0.03, 0.03),
            startangle=60)

# ------------------------------------- Handling Missing Values --------------------------------------

# We will try several methods to impute missing values
# 1) Mean Imputation
# 2) K-Nearest Neighbors Imputation
# 3) Multiple Imputation (MiceForest)
# 4) Predictive Modeling Imputation (RandomForestRegressor)

# we will copy the original data frame to compare with them each other
df_mean = df.copy()
df_knn = df.copy()
df_mice = df.copy()
df_rf = df.copy()


#################################
# 1) Mean Imputation
#################################

for col in df.columns[df.isnull().any()]:
    df_mean[col].fillna(df_mean[col].mean(), inplace=True)


#################################
# 2) K-Nearest Neighbors Imputation
#################################
from sklearn.impute import KNNImputer

knn_imp = KNNImputer(n_neighbors=4)
df_knn = pd.DataFrame(knn_imp.fit_transform(df_knn), columns=df_knn.columns)


#################################
# 3) Multiple Imputation (MiceForest)
#################################

from miceforest import ImputationKernel

mice_kernel = ImputationKernel(data=df_mice, save_all_iterations=True, random_state=41)
mice_kernel.mice(2)
df_mice = mice_kernel.complete_data()


#################################
# 4) Predictive Modeling Imputation (RandomForestRegressor)
#################################

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

def fill_missing_numerical(df, col):

    known = df[df[col].notnull()]
    unknown = df[df[col].isnull()]
    X = known.drop(df.columns[df.isnull().any()], axis=1) # features
    y = known[col] # target
    rf.fit(X, y)
    unknown[col] = rf.predict(unknown.drop(df.columns[df.isnull().any()], axis=1))
    df[col].fillna(unknown[col], inplace=True)


for col in df_rf.columns[df_rf.isnull().any()]:
    fill_missing_numerical(df_rf, col)



#################################
# Rechecking
#################################
# Distribution of filled values according to imputation methods

for col in df.columns[df.isnull().any()]:
    plt.figure(figsize=(10,7))
    df_mean.rename(columns={col: 'Mean Imputation of ' + col})['Mean Imputation of ' + col].plot(kind='kde',color='red')
    df_knn.rename(columns={col: 'KNN Imputation of ' + col})['KNN Imputation of ' + col].plot(kind='kde',color='blue')
    df_mice.rename(columns={col: 'Multiple Imputation of ' + col})['Multiple Imputation of ' + col].plot(kind='kde',color='green')
    df_rf.rename(columns={col: 'Predictive Imputation of ' + col})['Predictive Imputation of ' + col].plot(kind='kde',color='yellow')


    plt.legend()
    plt.title("Distribution of Filled " + col.upper() + " Values According to Imputation Methods")

# ---------------------------------- Data Analysis & Visualization -----------------------------------
#################################
# Pair plots of each data frame
#################################
# Pair plots of filled values according to imputation methods

# Mean Imputation
sns.pairplot(df_mean, hue='Potability')
plt.show()


# KNN Imputation
sns.pairplot(df_knn, hue='Potability')
plt.show()


# Multiple Imputation
sns.pairplot(df_mice, hue='Potability')
plt.show()


# Predictive Imputation
sns.pairplot(df_rf, hue='Potability')
plt.show()


#################################
# Correlation matrices of each data frame
#################################
# Correlation matrices of filled values according to imputation methods

fig, ax = plt.subplots(2,2, figsize=(16,16))

sns.heatmap(df_mean.corr(),annot=True, cmap='coolwarm', fmt=".2f", ax=ax[0,0]).set_title('Data Frame Mean')
sns.heatmap(df_knn.corr(),annot=True, cmap='viridis', fmt=".2f", ax=ax[0,1]).set_title('Data Frame KNN')
sns.heatmap(df_mice.corr(),annot=True, cmap='rocket', fmt=".2f", ax=ax[1,0]).set_title('Data Frame Multiple')
sns.heatmap(df_rf.corr(),annot=True, cmap='YlOrBr_r', fmt=".2f", ax=ax[1,1]).set_title('Data Frame Predictive')

fig.suptitle('Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.show()

# We can observe that there are differences between them, albeit small.

# ------------------------------------ Machine Learning Model -------------------------------------
#################################
# Prepraring the data for modelling
#################################
# We prepare training and test sets for each method

from sklearn.model_selection import train_test_split
# Mean Imputation
X_mean = df_mean.drop("Potability",axis=1)
y_mean = df_mean[["Potability"]]

# KNN Imputation
X_knn = df_knn.drop("Potability",axis=1)
y_knn = df_knn[["Potability"]]

# Multiple Imputation
X_mice = df_mice.drop("Potability",axis=1)
y_mice = df_mice[["Potability"]]

# Predictive Imputation
X_rf = df_rf.drop("Potability",axis=1)
y_rf = df_rf[["Potability"]]

X_mean_train, X_mean_test, y_mean_train, y_mean_test = train_test_split(X_mean, y_mean, test_size = 0.2, random_state = 0)
X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(X_knn, y_knn, test_size = 0.2, random_state = 0)
X_mice_train, X_mice_test, y_mice_train, y_mice_test = train_test_split(X_mice, y_mice, test_size = 0.2, random_state = 0)
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size = 0.2, random_state = 0)

#################################
# Calculation function
#################################
def calculate(y):
    acc = accuracy_score(y,y_test)
    pre = precision_score(y,y_test)
    rec = recall_score(y,y_test)
    f1 = f1_score(y,y_test)
    return acc,pre,rec,f1


data = [("Mean", X_mean_train,X_mean_test,y_mean_train,y_mean_test),
        ("KNN", X_knn_train,X_knn_test,y_knn_train,y_knn_test),
        ("Multiple", X_mice_train,X_mice_test,y_mice_train,y_mice_test),
        ("Predictive", X_rf_train,X_rf_test,y_rf_train,y_rf_test)]

#################################
# Modelling
#################################
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,roc_curve

# I will use the models that Gradient Boosting Classifier, Random Forest Classifier, XGBoost Classifier and LightGBM Classifier

gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier()
xgbc = XGBClassifier()
lgbc = lgb.LGBMClassifier()

models = [("Gradient Boosting",gbc), ("Random Forest", rfc), ("XBoost", xgbc), ("LightGBM", lgbc)]


#################################
# Visualization of ROC-AUC curve
#################################
def plot_roc(j, set, name, y_pred, y_test):

    plt.subplot(2, 2, j)
    ns_probs = [0 for i in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

    classifier_fpr, classifier_tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_pred, y_test)

    plt.plot(ns_fpr, ns_tpr, linestyle='solid')
    plt.plot(classifier_fpr, classifier_tpr,linestyle="dashed", marker='.', label = name + " (AUC = %0.4f)" % auc)

    plt.title('ROC Plot for ' + set )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.tight_layout()


#################################
# Model function
#################################
cls_rep = []
def model_call(j, set, X_train, X_test, y_train, y_test):


    for name, model in models:
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        acc, pre, rec, f1 = calculate(y_pred)
        cls_rep.append({'Data': set,'Model': name,'Accuracy': acc,'Precision': pre,'Recall': rec,'F1 Score': f1})

        plot_roc(j, set, name, y_pred, y_test)

    return cls_rep


#################################
# Visualization and evaluation of results
#################################

j = 0
plt.figure(figsize=(10,12))
for set, X_train, X_test, y_train, y_test in data:
    j += 1
    cls_rep = model_call(j, set, X_train, X_test, y_train, y_test)




# we can clearly see the roc-auc curve of each model in each data set

#################################
# Visualization of Confusion Matrix Metrics
#################################

def plot_metrics(k,data_name, df):

    ax = fig.add_subplot(2,2,k)
    df.plot(kind="bar",ax=ax)
    ax.set_title("Evaluation Metrics of Data Frame " + data_name.upper(), fontweight='bold')
    ax.set_xlabel("Metrics", fontweight='bold')
    ax.set_ylabel("Score", fontweight='bold')
    ax.set_xticklabels(rotation=0,labels=df.index)
    ax.grid(color='gray', linestyle=':', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False,fontsize="9")
    fig.tight_layout()


fig = plt.figure(figsize=(16,16))
k = 0
for i in range(0,16,4):
    df = pd.DataFrame(data = cls_rep[i:i+4])
    data_name = cls_rep[i]["Data"]
    df.drop("Data",axis=1,inplace=True)
    df = df.set_index("Model").unstack(level=1).unstack()
    k += 1
    plot_metrics(k,data_name, df)


# Conclusion: Looking at the graph we can say that predictive imputation is better than others for this data set.


"""
data_mean = pd.DataFrame(data = cls_rep[0:4])
data_mean.drop("Data", axis=1, inplace=True)
data_mean = data_mean.set_index("Model").unstack(level=1).unstack()

data_knn = pd.DataFrame(data = cls_rep[4:8])
data_knn.drop("Data", axis=1, inplace=True)
data_knn = data_knn.set_index("Model").unstack(level=1).unstack()

data_mice = pd.DataFrame(data = cls_rep[8:12])
data_mice.drop("Data", axis=1, inplace=True)
data_mice = data_mice.set_index("Model").unstack(level=1).unstack()

data_rf = pd.DataFrame(data = cls_rep[12:16])
data_rf.drop("Data", axis=1, inplace=True)
data_rf = data_rf.set_index("Model").unstack(level=1).unstack()
"""


"""
fig = plt.figure(figsize=(16,16))

ax1 = fig.add_subplot(2,2,1)
data_mean.plot(kind="bar",ax=ax1)
ax1.set_title("Evaluation Metrics of Data Frame MEAN", fontweight='bold')
ax1.set_xlabel("Metrics", fontweight='bold')
ax1.set_ylabel("Score", fontweight='bold')
ax1.set_xticklabels(rotation=0,labels=data_mean.index)
ax1.grid(color='gray', linestyle=':', linewidth=0.6, zorder=0)
ax1.set_axisbelow(True)
ax1.legend(frameon=False,fontsize="9")
fig.tight_layout()


ax2 = fig.add_subplot(2,2,2)
data_knn.plot(kind="bar",ax=ax2)
ax2.set_title("Evaluation Metrics of Data Frame KNN", fontweight='bold')
ax2.set_xlabel("Metrics", fontweight='bold')
ax2.set_ylabel("Score", fontweight='bold')
ax2.set_xticklabels(rotation=0,labels=data_knn.index)
ax2.grid(color='gray', linestyle=':', linewidth=0.6, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(frameon=False,fontsize="9")
fig.tight_layout()


ax3 = fig.add_subplot(2,2,3)
data_mice.plot(kind="bar",ax=ax3)
ax3.set_title("Evaluation Metrics of Data Frame MULTIPLE", fontweight='bold')
ax3.set_xlabel("Metrics", fontweight='bold')
ax3.set_ylabel("Score", fontweight='bold')
ax3.set_xticklabels(rotation=0,labels=data_mice.index)
ax3.grid(color='gray', linestyle=':', linewidth=0.6, zorder=0)
ax3.set_axisbelow(True)
ax3.legend(frameon=False,fontsize="9")
fig.tight_layout()


ax4 = fig.add_subplot(2,2,4)
data_rf.plot(kind="bar",ax=ax4)
ax4.set_title("Evaluation Metrics of Data Frame PREDICTIVE", fontweight='bold')
ax4.set_xlabel("Metrics", fontweight='bold')
ax4.set_ylabel("Score", fontweight='bold')
ax4.set_xticklabels(rotation=0,labels=data_rf.index)
ax4.grid(color='gray', linestyle=':', linewidth=0.6, zorder=0)
ax4.set_axisbelow(True)
ax4.legend(frameon=False,fontsize="9")
fig.tight_layout()
"""
