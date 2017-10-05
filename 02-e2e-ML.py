
# coding: utf-8

# In[13]:

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

housing_path = "data/housing"


# In[7]:

def load_housing_data (housing_path=housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[8]:

housing = load_housing_data()


# In[9]:

housing.head()


# ## Initial View

# ``.info()`` gives a quick description of the data. Total number of rows, number of non-null values, data types. 

# In[10]:

housing.info()


# ocean_proximity is some sort of categorical attribue. ``.value_counts()`` gives information on all the difference categories that exista & the number of attributes that below to each category.

# In[11]:

housing['ocean_proximity'].value_counts()


# ``.describe()`` shows summary of numerical attributes, count, mean, max etc

# In[12]:

housing.describe()


# Histograms are also useful to get a quick view of data

# In[15]:

housing.hist(bins = 50, figsize = (20,15))
plt.show()


# Page 46 gives some things to notice about the data. Mainly :
# 
# * The median_income is not in US Dollars
# * The median_house_value and housing_median_age are capped (notice the last bar on those histograms are largest for those plots). These may need to be removed. 
# * Attributes are are very different scales
# * Many of the histograms are tail heavy, the extend much further to the right of the median than the left. These may need to be transformed to more bell shaped curves.

# ## Create Test Set

# Great discussion in the book about dangers of simply doing a random sampling. If you just do a naive random sampling it will generate a different test set each time you run the program, and eventually the ML algorithm will get to see the full data set. 
# 
# One way to prevent this would be to save the test & train sest on first run and load them in on subsequent runs. Or a random seed call be set so it always generates the same shuffled indices. But what if the original dataset is updated? Both these solutions will break. 
# 
# Can compute a hash of each instance's identifer, keep only the last byte of the hash and put the instance in the test set. This ensures that the test set will remain consistent across multiple runs, even if you refresh the dataset. The new test set will contain 20% of the new instances, but it will not contain any instance that was previously in the training set.
# 
# Also need to be careful not to introduce a sampling bias. e.g. if the data was on population need to ensure the % of Male to Female is the same in the test as the training sets - stratified sampling, the population is divided into homogeneous subgroups called strata,
# and the right number of instances is sampled from each stratum to guarantee that the
# test set is representative of the overall population. 
# 
# In this housing dataset the median_income is important and be used for stratified sampling. Look at the histogram again.

# In[16]:

housing["median_income"].hist()


# Most clustered around 2-5, but some go beyond 6. Need to create just the right number of strata for this dataset

# In[17]:

# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[18]:

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[19]:

housing["income_cat"].value_counts() / len(housing)


# In[20]:

strat_test_set["income_cat"].value_counts()/len(strat_test_set)


# The income category proportions in the overall set are very similar to the stratified test set. Can remove the income_cat so the data is back to it's original state

# In[21]:

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis = 1, inplace = True)


# ## Discover and Visualise

# In[23]:

#Create copy of training set so you can play with it without harming it
housing = strat_train_set.copy()


# In[27]:

housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)
plt.show()
#alpha makes it easier to see high density areas


# In[30]:

housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.4
            ,s = housing["population"]/100, label = "population"
            ,c = "median_house_value", cmap=plt.get_cmap("jet"), colorbar = True, figsize = (15,10))
plt.show()
# s is sizing, so now the radius of each circle represents the population
# c is colour, so now the colour of each refers to the median_house_value


# ### Look for Correlations

# In[32]:

#since the dataset isn't too large we can look at the correlation between each pair of attributes
corr_matrix = housing.corr()


# In[33]:

# focus on the correlation with median_house_value
corr_matrix["median_house_value"].sort_values(ascending = False)


# Could also use ``scatter_matrix`` which plots every numerical attribute against every other one. This would be a quite large plot so focus on a few key ones. i.e. the ones with high correlations.

# In[36]:

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
pd.scatter_matrix(housing[attributes], figsize = (12,8))
plt.show()


# In[39]:

#median_income looks the best. 
housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1)


# The price cap is visible as a horizontal line at \$500,000, but also seen are less obvious straight lines, around  \$450,000 and  \$350,000, and possibly around \$280,000. It might be prudent to remove the corresponding districts to prevent the alogrithm from learning to reproduce these data quirks.

# One last thing you may want to do before actually preparing the data for Machine
# Learning algorithms is to try out various attribute combinations. For example, the
# total number of rooms in a district is not very useful if you don’t know how many
# households there are. What you really want is the number of rooms per household.
# Similarly, the total number of bedrooms by itself is not very useful: you probably
# want to compare it to the number of rooms. And the population per household also seems like an interesting attribute combination to look at. Let’s create these new
# attributes:

# In[40]:

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[42]:

#relook at the correlation matrix
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending = False)


# The new bedrooms_per_room attribute is much more correlated with
# the median house value than the total number of rooms or bedrooms.

# ## Data Cleaning

# Separate the predictors and the labels, since we don't necessarily want to apply the same transformations to the predictors and the target values.

# In[44]:

housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()


# Most Machine Learning algorithms cannot work with missing features, so let’s create
# a few functions to take care of them. You noticed earlier that the total_bedrooms
# attribute has some missing values, so let’s fix this. You have three options:
# * Get rid of the corresponding districts.
# * Get rid of the whole attribute.
# * Set the values to some value (zero, the mean, the median, etc.).
# 
# You can accomplish these easily using DataFrame’s dropna(), drop(), and fillna()
# methods:
# 
# If you choose option 3, you should compute the median value on the training set, and
# use it to fill the missing values in the training set, but also don’t forget to save the
# median value that you have computed. You will need it later to replace missing values
# in the test set when you want to evaluate your system, and also once the system goes
# live to replace missing values in new data.
# 
# Of course Scikit-Learn provides a handy class for this

# In[45]:

#Create an Imputer instance
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = "median")
#Can only impute on numerical, need to create a copy without the text attribute
housing_num = housing.drop("ocean_proximity", axis = 1)
#Now can fit the imputer instance to the training data using the fit() method
imputer.fit(housing_num)


# The imputer has simply computed the median of each attribute and stored the result
# in its statistics_ instance variable. Only the total_bedrooms attribute had missing
# values, but we cannot be sure that there won’t be any missing values in new data after
# the system goes live, so it is safer to apply the imputer to all the numerical attributes:

# In[46]:

imputer.statistics_


# In[47]:

housing_num.median().values


# Now you can use this “trained” imputer to transform the training set by replacing
# missing values by the learned medians:

# In[48]:

X = imputer.transform(housing_num)
#results in a numpy array. You can put it into a df
housing_tr = pd.DataFrame(X, columns = housing_num.columns)


# ### Handling Text
# Scikit-Learn provides a transformer to convert text labels to numbers

# In[50]:

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded


# One issue with this representation is that ML algorithms will assume that two nearby
# values are more similar than two distant values. To fix this a common solution is to create one binary attribute per category.This is
# called one-hot encoding, because only one attribute will be equal to 1 (hot), while the
# others will be 0 (cold).
# 
# Scikit-Learn provides a OneHotEncoder encoder to convert integer categorical values
# into one-hot vectors. Let’s encode the categories as one-hot vectors. Note that
# fit_transform() expects a 2D array, but housing_cat_encoded is a 1D array, so we
# need to reshape it:

# In[51]:

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot


# We can apply both transformations (from text categories to integer categories, then
# from integer categories to one-hot vectors) in one shot using the LabelBinarizer
# class:

# In[52]:

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot


# ### Creating Custom Transformers

# All you need is to create a class and implement three methods: fit()
# (returning self), transform(), and fit_transform(). You can get the last one for
# free by simply adding TransformerMixin as a base class. Also, if you add BaseEstima
# tor as a base class (and avoid \*args and \**kargs in your constructor) you will get
# two extra methods (get_params() and set_params()) that will be useful for automatic
# hyperparameter tuning. For example, here is a small transformer class that adds
# the combined attributes we discussed earlier:

# In[55]:

from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# Not that Scikit-Learn provides a transformer called StandardScaler for standardization

# ### Transformation Pipelines

# As you can see, there are many data transformation steps that need to be executed in
# the right order. Fortunately, Scikit-Learn provides the Pipeline class to help with
# such sequences of transformations. Here is a small pipeline for the numerical
# attributes:

# In[57]:

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
('imputer', Imputer(strategy="median")),
('attribs_adder', CombinedAttributesAdder()),
('std_scaler', StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# You now have a pipeline for numerical values, and you also need to apply the LabelBi
# narizer on the categorical values: how can you join these transformations into a single
# pipeline? Scikit-Learn provides a FeatureUnion class for this. A full pipeline handling
# both numerical and categorical attributes may look like this:

# There is nothing in Scikit-Learn
# to handle selection from Pandas DataFrames so we need to write a simple custom transformer for
# this task:

# In[62]:

from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[63]:

from sklearn.pipeline import FeatureUnion
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)


# ## Select and Train a Model

# ### Train & Evaluate on the Training Set
# #### Linear Regression

# In[66]:

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[67]:

#Try a few instances out.
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_labels))


# In[70]:

# can check the RMSE on the whole set by using scikit learns mean_squared_error
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# This is better than nothing but clearly not a great score: most districts’
# median_housing_values range between \$120,000 and \$265,000, so a typical prediction
# error of \$68,628 is not very satisfying. This is an example of model underfitting the training data. When this happens it can mean that the features do not provide
# enough information to make good predictions, or that the model is not powerful
# enough.You could try to add more features (e.g., the log of the population),
# but first let’s try a more complex model to see how it does.

# #### Decision Tree

# In[72]:

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[73]:

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# Zero error? Much more likely that the model has badly overfit the data. How can you be sure?
# As we saw earlier, you don’t want to touch the test set until you are ready to launch a
# model you are confident about, so you need to use part of the training set for training,
# and part for model validation.

# ### Better Evaluation Using Cross-Validation

# One way to evaluate the Decision Tree model would be to use the train_test_split
# function to split the training set into a smaller training set and a validation set, then
# train your models against the smaller training set and evaluate them against the validation
# set. It’s a bit of work, but nothing too difficult and it would work fairly well.
# 
# A great alternative is to use Scikit-Learn’s cross-validation feature. The following code
# performs K-fold cross-validation: it randomly splits the training set into 10 distinct
# subsets called folds, then it trains and evaluates the Decision Tree model 10 times,
# picking a different fold for evaluation every time and training on the other 9 folds.
# The result is an array containing the 10 evaluation scores:
# 
# **NOTE:**
# Scikit-Learn cross-validation features expect a utility function
# (greater is better) rather than a cost function (lower is better), so
# the scoring function is actually the opposite of the MSE (i.e., a negative
# value), which is why the preceding code computes -scores
# before calculating the square root.

# In[78]:

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels
                             ,scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[79]:

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[80]:

display_scores(tree_rmse_scores)


# This actually looks worse than the Linear Regression (\$70545 vs \$68,628). Let's check by doing the same cross validation for Lienar Regression.

# In[82]:

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels
                             ,scoring = "neg_mean_squared_error", cv = 10)
lin_rmse_scores = np.sqrt(-lin_scores)


# In[83]:

display_scores(lin_rmse_scores)


# #### Random Forest
# Random Forests work by training many Decision Trees on random subsets of
# the features, then averaging out their predictions. Building a model on top of many
# other models is called Ensemble Learning, and it is often a great way to push ML algorithms
# even further.

# In[85]:

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)


# In[87]:

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[89]:

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels
                               ,scoring = "neg_mean_squared_error", cv = 10)
forest_rmse_scores = np.sqrt(-forest_scores)


# In[91]:

display_scores(forest_rmse_scores)


# Much better, however, note that
# the score on the training set is still much lower than on the validation sets, meaning
# that the model is still overfitting the training set. Possible solutions for overfitting are
# to simplify the model, constrain it (i.e., regularize it), or get a lot more training data.

# #### Support Vector Machine regressor

# In[93]:

from sklearn.svm import SVR

svm_reg_linear = SVR(kernel = "linear")
svm_reg_linear.fit(housing_prepared, housing_labels)


# In[94]:

housing_predictions = svm_reg_linear.predict(housing_prepared)
svm_reg_linear_mse = mean_squared_error(housing_labels, housing_predictions)
svm_reg_linear_rmse = np.sqrt(svm_reg_linear_mse)
svm_reg_linear_rmse


# ### Grid Search

# Want to fiddle with teh hyperparameters to see which is best. Should get Scikit-Learn’s GridSearchCV to search for you. All you need to
# do is tell it which hyperparameters you want it to experiment with, and what values to
# try out, and it will evaluate all the possible combinations of hyperparameter values,
# using cross-validation. For example, the following code searches for the best combination
# of hyperparameter values for the RandomForestRegressor:

# In[96]:

from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)


# In[97]:

grid_search.best_params_


# In[98]:

grid_search.best_estimator_


# In[99]:

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# You will often gain good insights on the problem by inspecting the best models.

# In[100]:

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[103]:

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse = True)


# With this information, you may want to try dropping some of the less useful features
# (e.g., apparently only one ocean_proximity category is really useful, so you could try
# dropping the others).

# ## Evaluate on the Test Set

# In[108]:

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis = 1)
y_test = strat_test_set["median_house_value"].copy()

#note: call transform() method NOT fit_transform()
X_test_prepared = full_pipeline.transform(X_test)


# In[110]:

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[111]:

from sklearn.externals import joblib

joblib.dump(final_model, "02-e2e-model.pkl")


# In[112]:

final_model_loaded = joblib.load("02-e2e-model.pkl")


# In[113]:

final_predictions_loaded = final_model_loaded.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions_loaded)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[ ]:



