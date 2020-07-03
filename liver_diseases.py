#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Do necessary import

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# In[2]:


ld = pd.read_csv(r"F:/python/imarticus/dataset/Liver_Diseases_new.csv")


# In[3]:


ld.head()


# In[4]:


#remove unwanted column (Number)

ld.drop(["No"], axis= 1, inplace=True)


# In[5]:


#check nulls, 

ld.isnull().sum()


# In[6]:


#cleaning Gender column

ld.Gender.value_counts()


# In[7]:


# taking fact that male is more than female, we can replace null with male

ld.Gender = ld.Gender.fillna("Male")


# In[8]:


#replacing null of column "total_protiens" by its median value

ld.Total_Protiens = ld.Total_Protiens.fillna(ld.Total_Protiens.median())


# In[9]:


# removing null in "Albumin_and_Globulin_Ratio" with its median

ld.Albumin_and_Globulin_Ratio = ld.Albumin_and_Globulin_Ratio.fillna(ld.Albumin_and_Globulin_Ratio.median())


# In[10]:


#confirm removal of nulls

ld.isnull().sum()


# In[11]:


#now, nulls have been removed, check zeros in columns

def findzero(x):
    c= list(x.columns)
    return(((x[c])== 0).sum())


# In[12]:


findzero(ld) # there are no zeros present in the dataset


# In[13]:


#check the type of the columns

ld.info() #everything is just good


# In[14]:


#let's check valuue_counts() of the column "class"

ld.Class.value_counts()


# In[15]:


le = LabelEncoder()


# In[16]:


ld[ld.select_dtypes(include = ["object"]).columns] = ld[ld.select_dtypes(include = ["object"]).columns].apply(le.fit_transform)


# In[17]:


ld.shape


# In[18]:


#split data into train and test

ld_x = ld.iloc[:,0:10]
ld_y = ld.iloc[:,-1]


# In[19]:


ld_x_train, ld_x_test, ld_y_train, ld_y_test = train_test_split(ld_x, ld_y, test_size = 0.2)


# # Logistic Regression

# In[20]:


glm = LogisticRegression(class_weight= "balanced")


# In[21]:


glm.fit(ld_x_train, ld_y_train)


# In[22]:


pred_glm = glm.predict(ld_x_test)
pred_glm


# In[23]:


from sklearn.metrics import confusion_matrix

cm_glm = confusion_matrix(pred_glm, ld_y_test)
cm_glm


# In[24]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print(classification_report(ld_y_test, pred_glm))


# logistic regreesion gives bad prediction for TN (True Negative), That's why we need to go for feature selection
# Let's check feature imortance by different three techniques 
# # Chi-square Method

# In[43]:


test = SelectKBest(score_func = chi2, k = "all")


# In[44]:


fited = test.fit(ld_x,ld_y)
fited


# In[46]:


fited.scores_


# In[47]:


feature_importance = pd.DataFrame({"Features":list(ld_x_test.columns),"Importance":list(fited.scores_)})
# sorted
feature_importance.sort_values("Importance",ascending = False)

According to chi-square method important features are:
> Aspartate_Aminotransferase
> Alamine_Aminotransferase
> Alkaline_Phosphotase
> Total_Bilirubin
> Direct_Bilirubin
> Age

# # boruta

# In[48]:


from boruta import BorutaPy
rf = RandomForestClassifier()


# In[50]:


ld_xn = ld.iloc[:,0:10]


# In[51]:


import numpy as np
ld_dup = np.array(ld_xn)
ld_dup


# In[52]:


boruta_feature_selector = BorutaPy(rf, random_state = 111, max_iter =25, perc = 100, verbose = 2)
boruta_feature_selector


# In[53]:


#do feature selection on your entire data

boruta_feature_selector.fit(ld_dup,ld_y)


# In[54]:


boruta_feature_selector.support_


# In[55]:


boruta_feature = pd.DataFrame({"Feature":list(ld_x.columns),"importance":list(boruta_feature_selector.support_)})
boruta_feature.sort_values("importance", ascending = False)

According to boruta method important features are:
> Age
> Total_Bilirubin	
> Alkaline_Phosphotase
> Alamine_Aminotransferase
> Aspartate_Aminotransferase
# # RFE

# In[56]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


# In[59]:


rf = RandomForestClassifier()


# In[94]:


rfe_rfc = RFE(rf, 9)


# In[95]:


rfe_rfc.fit(ld_x, ld_y)


# In[96]:


rfe_rfc.support_

feature_importance = pd.DataFrame({"Features":list(ld_x.columns),"Importance":list(rfe_rfc.support_)})
# sorted
feature_importance.sort_values("Importance",ascending = False)imortant features for RFE technique are rank wise:

1. Alkaline_Phosphotase
2. Aspartate_Aminotransferase
3. Age
4. Alamine_Aminotransferase	
5. Total_Bilirubin	
6. Albumin
7. Total_Protiens
8. Direct_Bilirubin	
9. Albumin_and_Globulin_Ratio
10. Gender
# >> On the basis of above analysis, we can seperate important features, which are mentioned below:
# 
# 1. Alkaline_Phosphotase
# 2. Aspartate_Aminotransferase
# 3. Age
# 4. Alamine_Aminotransferase	
# 5. Total_Bilirubin
******* BUILT DIFFERENT MODELS TO CHECK ACCURACY, TP, TN
# # Decision Tree

# In[45]:


dtree = DecisionTreeClassifier(class_weight= "balanced", criterion='entropy')


# In[46]:


dtree.fit(ld_x_train, ld_y_train)


# In[47]:


pred_dtree = dtree.predict(ld_x_test)
pred_dtree


# In[48]:


cm_dtree = confusion_matrix(pred_dtree, ld_y_test)
cm_dtree


# In[49]:


print(classification_report(pred_dtree, ld_y_test))


# # Random Forest

# In[55]:


rf = RandomForestClassifier(class_weight= "balanced") 


# In[56]:


rf.fit(ld_x_train, ld_y_train)


# In[57]:


pred_rf = rf.predict(ld_x_test)


# In[58]:


cm_rf = confusion_matrix(pred_rf, ld_y_test)
cm_rf


# In[59]:


print(classification_report(pred_rf, ld_y_test))


# In[60]:


rf2 = RandomForestClassifier(class_weight= "balanced")


# In[61]:


rf2.fit(ld2_x_train, ld2_y_train)


# In[62]:


pred_rf2 = rf2.predict(ld2_x_test)


# In[63]:


cm_rf2 = confusion_matrix(pred_rf2, ld2_y_test)
cm_rf2


# In[64]:


print(classification_report(pred_rf2, ld2_y_test))


# # Gradient Boosting

# In[65]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()


# In[66]:


gbc.fit(ld_x_train, ld_y_train)


# In[67]:


pred_gbc = gbc.predict(ld_x_test)


# In[68]:


cm_gbc = confusion_matrix(pred_gbc, ld_y_test)
cm_gbc


# In[69]:


print(classification_report(pred_gbc, ld_y_test))


# > Based on above models, it has been observed that accuracy for catagory 1 is very less
# or we can say: model is giving positive result for those who actually don't have liver diseases.

# # Remove outliers in order to impove model

# In[71]:


ld3 = ld.iloc[:,0:]


# In[72]:



ld3.Age = ld3.Age.astype("object")
ld3.Gender = ld3.Gender.astype("object")
ld3.Class = ld3.Class.astype("object")


# In[73]:


numcol = ld3.select_dtypes(include= np.number).columns
numcol


# In[74]:


for i in numcol:
    plt.figure()
    ld3.boxplot([i])


# In[75]:


#Identify the outlier using Boxplot
def remove_outliers_boxplot_reverse(df,col):
    q3 = np.quantile(df[col],0.75)
    q1 = np.quantile(df[col],0.25)
    iqr = q3-q1
    #print("iqr is...",iqr)
    global outlier_free_list
    global outlier_free_df
    outlier_free_list = [ x for x in df[col]  if ( (x < (q3 + 1.5*iqr))   &  (x > (q1 - 1.5*iqr)) )]
    outlier_free_df = df.loc[df[col].isin(outlier_free_list)];print(outlier_free_df.shape); return outlier_free_df


# In[76]:


ld3 = remove_outliers_boxplot_reverse(ld3, "Total_Bilirubin")
ld3 = remove_outliers_boxplot_reverse(ld3, "Direct_Bilirubin")
ld3 = remove_outliers_boxplot_reverse(ld3, "Alkaline_Phosphotase")
ld3 = remove_outliers_boxplot_reverse(ld3, "Alamine_Aminotransferase")
ld3 = remove_outliers_boxplot_reverse(ld3, "Aspartate_Aminotransferase")
ld3 = remove_outliers_boxplot_reverse(ld3, "Total_Protiens")
ld3 = remove_outliers_boxplot_reverse(ld3, "Albumin_and_Globulin_Ratio")


# In[77]:


ld3[ld3.select_dtypes(include = ["object"]).columns] = ld3[ld3.select_dtypes(include = ["object"]).columns].apply(le.fit_transform)


# In[78]:


ld3_x = ld3.iloc[:,0:10]
ld3_y = ld3.iloc[:,-1]


# In[79]:


ld3_x_train, ld3_x_test, ld3_y_train, ld3_y_test = train_test_split(ld3_x, ld3_y, test_size = 0.3)


# In[80]:


dtree3 =DecisionTreeClassifier(class_weight="balanced")


# In[81]:


dtree3.fit(ld3_x_train, ld3_y_train)


# In[82]:


pred_dtree3 = dtree3.predict(ld3_x_test)


# In[83]:


cm_dtree3 = confusion_matrix(pred_dtree3, ld3_y_test)
cm_dtree3


# In[84]:


ld.shape


# > removing outliers reducing data about 45%, therefore this method is also not giving better accuracy for true negative(TN) (for catagory 1), which is happening because of class imbalance

# # Model Building with Class Imbalanced Technique

# In[86]:


ld.Class.value_counts()


# > in class column presence of catagory "1" is about 29% as compare to catagory "0", therefore class imbalanced process is being tested

# In[89]:


ld4 = ld.iloc[:,0:]


# In[91]:


ld4.shape


# In[92]:


ld4_x = ld4.iloc[:,0:10]
ld4_y = ld4.iloc[:,-1]


# In[93]:


ld4_train, ld4_test = train_test_split(ld4, test_size = .2)


# In[97]:


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
abc = ld4_train[ld4_train.Class == 1]


# In[115]:


ld4_train2 = pd.concat([ld4_train,abc,abc])
ld4_train2.Class.value_counts()


# In[116]:


ld4_test_x = ld4_test.iloc[:,0:10]
ld4_test_y = ld4_test.iloc[:,-1]


# In[117]:


ld4_train_x = ld4_train2.iloc[:,0:10]
ld4_train_y = ld4_train2.iloc[:,-1]


# In[ ]:


########              RandomForestClassifier            ###########

###################################################################


# In[118]:


rf4 = RandomForestClassifier()


# In[119]:


rf4.fit(ld4_train_x, ld4_train_y)


# In[120]:


pred_rf4 = rf4.predict(ld4_test_x)


# In[121]:


cm_rf4 = confusion_matrix(pred_rf4, ld4_test_y)
cm_rf4


# In[122]:


print(classification_report(pred_rf4, ld4_test_y))

# It shows that accuracy has improved as compare to before by using class imbalanced technique.
# In[ ]:


########              AdaBoostClassifier            ###########

###################################################################


# In[123]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# In[124]:


abc = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=25)


# In[125]:


abc.fit(ld4_train_x, ld4_train_y)


# In[126]:


pred_ada = abc.predict(ld4_test_x)


# In[127]:


cm_ada = confusion_matrix(pred_ada, ld4_test_y)
cm_ada


# In[128]:


print(classification_report(pred_ada, ld4_test_y))


# In[ ]:


########              DecisionTreeClassifier            ###########

###################################################################


# In[129]:


dtree4 = DecisionTreeClassifier()


# In[130]:


dtree4.fit(ld4_train_x, ld4_train_y)


# In[138]:


pred_dtree4 = dtree4.predict(ld4_test_x)


# In[139]:


cm_dtree4 = confusion_matrix(pred_dtree4, ld4_test_y)
cm_dtree4


# > By fixing class imbalace, models are giving comparatively quite good result than before
