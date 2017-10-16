
# coding: utf-8

# Suppose you ask a complex question to thousands of random people, then aggregate
# their answers. In many cases you will find that this aggregated answer is better than
# an expert’s answer. This is called the wisdom of the crowd. Similarly, if you aggregate
# the predictions of a group of predictors (such as classifiers or regressors), you will
# often get better predictions than with the best individual predictor. A group of predictors
# is called an ensemble; thus, this technique is called Ensemble Learning, and an
# Ensemble Learning algorithm is called an Ensemble method.
# 
# For example, you can train a group of Decision Tree classifiers, each on a different
# random subset of the training set. To make predictions, you just obtain the predictions
# of all individual trees, then predict the class that gets the most votes (see the last
# exercise in Chapter 6). Such an ensemble of Decision Trees is called a Random Forest,
# and despite its simplicity, this is one of the most powerful Machine Learning algorithms
# available today.
# 
# Moreover, as we discussed in Chapter 2, you will often use Ensemble methods near
# the end of a project, once you have already built a few good predictors, to combine
# them into an even better predictor. In fact, the winning solutions in Machine Learning
# competitions often involve several Ensemble methods (most famously in the Netflix
# Prize competition).
# 
# In this chapter we will discuss the most popular Ensemble methods, including bagging,
# boosting, stacking, and a few others. We will also explore Random Forests.

# ## Voting Classifiers

# Suppose you have trained a few classifiers, each one achieving about 80% accuracy.
# You may have a Logistic Regression classifier, an SVM classifier, a Random Forest
# classifier, a K-Nearest Neighbors classifier, and perhaps a few more.
# 
# A very simple way to create an even better classifier is to aggregate the predictions of
# each classifier and predict the class that gets the most votes. This majority-vote classifier
# is called a hard voting classifier.
# 
# Somewhat surprisingly, this voting classifier often achieves a higher accuracy than the
# best classifier in the ensemble. In fact, even if each classifier is a weak learner (meaning
# it does only slightly better than random guessing), the ensemble can still be a
# strong learner (achieving high accuracy), provided there are a sufficient number of
# weak learners and they are sufficiently diverse.
# 
# The following analogy can help shed some light on this mystery.
# Suppose you have a slightly biased coin that has a 51% chance of coming up heads,
# and 49% chance of coming up tails. If you toss it 1,000 times, you will generally get
# more or less 510 heads and 490 tails, and hence a majority of heads. If you do the
# math, you will find that the probability of obtaining a majority of heads after 1,000
# tosses is close to 75%. The more you toss the coin, the higher the probability (e.g.,
# with 10,000 tosses, the probability climbs over 97%). This is due to the law of large
# numbers: as you keep tossing the coin, the ratio of heads gets closer and closer to the
# probability of heads (51%).
# 
# Similarly, suppose you build an ensemble containing 1,000 classifiers that are individually
# correct only 51% of the time (barely better than random guessing). If you predict
# the majority voted class, you can hope for up to 75% accuracy! However, this is
# only true if all classifiers are perfectly independent, making uncorrelated errors,
# which is clearly not the case since they are trained on the same data. They are likely to
# make the same types of errors, so there will be many majority votes for the wrong
# class, reducing the ensemble’s accuracy.
# 
# Ensemble methods work best when the predictors are as independent
# from one another as possible. One way to get diverse classifiers
# is to train them using very different algorithms. This increases the
# chance that they will make very different types of errors, improving
# the ensemble’s accuracy.
# 
# The following code creates and trains a voting classifier in Scikit-Learn, composed of
# three diverse classifiers

# In[6]:

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[7]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[8]:

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()


# In[9]:

voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting ='hard'
)


# In[11]:

voting_clf.fit(X_train, y_train)


# In[12]:

from sklearn.metrics import accuracy_score


# In[13]:

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# There you have it! The voting classifier slightly outperforms all the individual classifiers.

# If all classifiers are able to estimate class probabilities (i.e., they have a pre
# dict_proba() method), then you can tell Scikit-Learn to predict the class with the
# highest class probability, averaged over all the individual classifiers. This is called soft
# voting. It often achieves higher performance than hard voting because it gives more
# weight to highly confident votes. All you need to do is replace voting="hard" with
# voting="soft" and ensure that all classifiers can estimate class probabilities.
# 
# This is not the case of the SVC class by default, so you need to set its probability hyperparameter
# to True (this will make the SVC class use cross-validation to estimate class probabilities,
# slowing down training, and it will add a predict_proba() method). If you
# modify the preceding code to use soft voting, you will find that the voting classifier
# achieves over 91% accuracy!

# In[14]:

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)


# In[15]:

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# ## Bagging and Pasting 

# One way to get a diverse set of classifiers is to use very different training algorithms,
# as just discussed. Another approach is to use the same training algorithm for every
# predictor, but to train them on different random subsets of the training set. When
# sampling is performed with replacement, this method is called bagging (short for
# bootstrap aggregating). When sampling is performed without replacement, it is called
# pasting.
# 
# In other words, both bagging and pasting allow training instances to be sampled several
# times across multiple predictors, but only bagging allows training instances to be
# sampled several times for the same predictor.
# 
# Once all predictors are trained, the ensemble can make a prediction for a new
# instance by simply aggregating the predictions of all predictors. The aggregation
# function is typically the statistical mode (i.e., the most frequent prediction, just like a
# hard voting classifier) for classification, or the average for regression. Each individual
# predictor has a higher bias than if it were trained on the original training set, but
# aggregation reduces both bias and variance. Generally, the net result is that the ensemble has a similar bias but a lower variance than a single predictor trained on the
# original training set.
# 
# Predictors can all be trained in parallel, via different
# CPU cores or even different servers. Similarly, predictions can be made in parallel.
# This is one of the reasons why bagging and pasting are such popular methods: they
# scale very well.

# ### Bagging and Pasting in Scikit-Learn

# Scikit-Learn offers a simple API for both bagging and pasting with the ``BaggingClassifier`` class (or ``BaggingRegressor`` for regression). The following code trains an ensemble of 500 Decision Tree classifiers each trained on 100 training instances randomly sampled from the training set with replacement (this is an example of bagging, but if you want to use pasting instead, just set ``bootstrap=False``). The ``n_jobs`` parameter tells Scikit-Learn the number of CPU cores to use for training and predictions(–1 tells Scikit-Learn to use all available cores):

# In[20]:

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100,bootstrap=True,n_jobs=-1
)


# In[21]:

bag_clf.fit(X_train, y_train)


# In[24]:

y_pred = bag_clf.predict(X_test)


# In[25]:

print(accuracy_score(y_test, y_pred))


# In[26]:

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))


# The BaggingClassifier automatically performs soft voting
# instead of hard voting if the base classifier can estimate class probabilities
# (i.e., if it has a predict_proba() method), which is the case
# with Decision Trees classifiers.
# 
# Bootstrapping introduces a bit more diversity in the subsets that each predictor is
# trained on, so bagging ends up with a slightly higher bias than pasting, but this also
# means that predictors end up being less correlated so the ensemble’s variance is
# reduced. Overall, bagging often results in better models, which explains why it is generally
# preferred. However, if you have spare time and CPU power you can use crossvalidation
# to evaluate both bagging and pasting and select the one that works best.

# ### Out-of-Bag Evaluation

# With bagging, some instances may be sampled several times for any given predictor,
# while others may not be sampled at all. By default a BaggingClassifier samples m
# training instances with replacement (bootstrap=True), where m is the size of the
# training set. This means that only about 63% of the training instances are sampled on
# average for each predictor.6 The remaining 37% of the training instances that are not
# sampled are called out-of-bag (oob) instances. Note that they are not the same 37%
# for all predictors.
# 
# Since a predictor never sees the oob instances during training, it can be evaluated on
# these instances, without the need for a separate validation set or cross-validation. You
# can evaluate the ensemble itself by averaging out the oob evaluations of each predictor.
# 
# In Scikit-Learn, you can set oob_score=True when creating a BaggingClassifier to
# request an automatic oob evaluation after training. The following code demonstrates
# this. The resulting evaluation score is available through the ``oob_score_`` variable:

# In[27]:

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True

)


# In[28]:

bag_clf.fit(X_train, y_train)


# In[30]:

bag_clf.oob_score_


# We are likely to get this accuracy on the test set.

# In[31]:

y_pred = bag_clf.predict(X_test)


# In[32]:

accuracy_score(y_test, y_pred)


# The oob decision function for each training instance is also available through the
# oob_decision_function_ variable.

# ## Random Patches and Random Subspaces

# The BaggingClassifier class supports sampling the features as well. This is controlled
# by two hyperparameters: max_features and bootstrap_features. They work
# the same way as max_samples and bootstrap, but for feature sampling instead of
# instance sampling. Thus, each predictor will be trained on a random subset of the
# input features.
# 
# This is particularly useful when you are dealing with high-dimensional inputs (such
# as images). Sampling both training instances and features is called the Random
# Patches method. Keeping all training instances (i.e., bootstrap=False and max_sam
# ples=1.0) but sampling features (i.e., bootstrap_features=True and/or max_fea
# tures smaller than 1.0) is called the Random Subspaces method.
# 
# Sampling features results in even more predictor diversity, trading a bit more bias for
# a lower variance.

# ## Random Forests

# Instead of building a BaggingClassifier and passing
# it a DecisionTreeClassifier, you can instead use the RandomForestClassifier
# class, which is more convenient and optimized for Decision Trees10 (similarly, there is
# a RandomForestRegressor class for regression tasks). The following code trains a
# Random Forest classifier with 500 trees (each limited to maximum 16 nodes), using
# all available CPU cores:

# In[34]:

from sklearn.ensemble import RandomForestClassifier


# In[35]:

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)


# In[36]:

rnd_clf.fit(X_train, y_train)


# In[37]:

y_pred_rf = rnd_clf.predict(X_test)


# With a few exceptions, a RandomForestClassifier has all the hyperparameters of a
# DecisionTreeClassifier (to control how trees are grown), plus all the hyperparameters
# of a BaggingClassifier to control the ensemble itself.
# 
# The Random Forest algorithm introduces extra randomness when growing trees;
# instead of searching for the very best feature when splitting a node (see Chapter 6), it
# searches for the best feature among a random subset of features. This results in a
# greater tree diversity, which (once again) trades a higher bias for a lower variance,
# generally yielding an overall better model. The following BaggingClassifier is
# roughly equivalent to the previous RandomForestClassifier.

# In[41]:

bag_clf = BaggingClassifier(
        DecisionTreeClassifier(splitter='random', max_leaf_nodes=16), 
        n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1
)


# ### Extra Trees

# When you are growing a tree in a Random Forest, at each node only a random subset
# of the features is considered for splitting (as discussed earlier). It is possible to make
# trees even more random by also using random thresholds for each feature rather than
# searching for the best possible thresholds (like regular Decision Trees do).
# 
# A forest of such extremely random trees is simply called an Extremely Randomized
# Trees ensemble12 (or Extra-Trees for short). Once again, this trades more bias for a
# lower variance. It also makes Extra-Trees much faster to train than regular Random
# Forests since finding the best possible threshold for each feature at every node is one
# of the most time-consuming tasks of growing a tree.
# 
# You can create an Extra-Trees classifier using Scikit-Learn’s ``ExtraTreesClassifier``
# class. Its API is identical to the RandomForestClassifier class. Similarly, the Extra
# TreesRegressor class has the same API as the RandomForestRegressor class.
# 
# It is hard to tell in advance whether a RandomForestClassifier
# will perform better or worse than an ExtraTreesClassifier. Generally,
# the only way to know is to try both and compare them using
# cross-validation (and tuning the hyperparameters using grid
# search).

# ### Feature Importance

# Lastly, if you look at a single Decision Tree, important features are likely to appear
# closer to the root of the tree, while unimportant features will often appear closer to
# the leaves (or not at all). It is therefore possible to get an estimate of a feature’s importance
# by computing the average depth at which it appears across all trees in the forest.
# Scikit-Learn computes this automatically for every feature after training. 
# 
# You can access the result using the ``feature_importances_`` variable. For example, the following
# code trains a RandomForestClassifier on the iris dataset (introduced in Chapter
# 4) and outputs each feature’s importance. It seems that the most important
# features are the petal length and width , while sepal length and width are
# rather unimportant in comparison:

# In[42]:

from sklearn.datasets import load_iris


# In[43]:

iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs = -1)
rnd_clf.fit(iris['data'], iris['target'])
for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print(name, score)


# Random Forests are very handy to get a quick understanding of what features
# actually matter, in particular if you need to perform feature selection.

# ## Boosting
# Boosting (originally called hypothesis boosting) refers to any Ensemble method that
# can combine several weak learners into a strong learner. The general idea of most
# boosting methods is to train predictors sequentially, each trying to correct its predecessor.
# There are many boosting methods available, but by far the most popular are
# AdaBoost13 (short for Adaptive Boosting) and Gradient Boosting. Let’s start with Ada‐Boost

# ### AdaBoost
# 
# One way for a new predictor to correct its predecessor is to pay a bit more attention
# to the training instances that the predecessor underfitted. This results in new predictors
# focusing more and more on the hard cases. This is the technique used by Ada‐
# Boost.
# 
# For example, to build an AdaBoost classifier, a first base classifier (such as a Decision
# Tree) is trained and used to make predictions on the training set. The relative weight
# of misclassified training instances is then increased. A second classifier is trained
# using the updated weights and again it makes predictions on the training set, weights
# are updated, and so on.
# 
# This sequential learning technique has some similarities with Gradient Descent, except that instead of tweaking a single predictor’s parameters to minimize a cost function, AdaBoost adds predictors to the ensemble,
# gradually making it better.
# 
# Once all predictors are trained, the ensemble makes predictions very much like bagging
# or pasting, except that predictors have different weights depending on their
# overall accuracy on the weighted training set.
# 
# There is one important drawback to this sequential learning technique:
# it cannot be parallelized (or only partially), since each predictor
# can only be trained after the previous predictor has been
# trained and evaluated. As a result, it does not scale as well as bagging or pasting.
# 
# Scikit-Learn actually uses a multiclass version of AdaBoost called SAMME16 (which
# stands for Stagewise Additive Modeling using a Multiclass Exponential loss function).
# When there are just two classes, SAMME is equivalent to AdaBoost. Moreover, if the
# predictors can estimate class probabilities (i.e., if they have a predict_proba()
# method), Scikit-Learn can use a variant of SAMME called SAMME.R (the R stands
# for “Real”), which relies on class probabilities rather than predictions and generally
# performs better.
# 
# The following code trains an AdaBoost classifier based on 200 Decision Stumps using
# Scikit-Learn’s AdaBoostClassifier class (as you might expect, there is also an Ada
# BoostRegressor class). A Decision Stump is a Decision Tree with max_depth=1—in
# other words, a tree composed of a single decision node plus two leaf nodes. This is
# the default base estimator for the AdaBoostClassifier class:

# In[45]:

from sklearn.ensemble import AdaBoostClassifier


# In[46]:

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200, 
    algorithm="SAMME.R", learning_rate=0.5
    )


# In[47]:

ada_clf.fit(X_train, y_train)


# If your AdaBoost ensemble is overfitting the training set, you can
# try reducing the number of estimators or more strongly regularizing
# the base estimator.

# ### Gradient Boosting

# Another very popular Boosting algorithm is Gradient Boosting. Just like AdaBoost,
# Gradient Boosting works by sequentially adding predictors to an ensemble, each one
# correcting its predecessor. However, instead of tweaking the instance weights at every
# iteration like AdaBoost does, this method tries to fit the new predictor to the residual
# errors made by the previous predictor.
# 
# Let’s go through a simple regression example using Decision Trees as the base predictors
# (of course Gradient Boosting also works great with regression tasks). This is
# called Gradient Tree Boosting, or Gradient Boosted Regression Trees (GBRT). First, let’s
# fit a DecisionTreeRegressor to the training set (for example, a noisy quadratic training
# set):

# In[49]:

import numpy as np
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)


# In[50]:

from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)


# Now train a second DecisionTreeRegressor on the residual errors made by the first
# predictor:

# In[51]:

y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth = 2, random_state=42)
tree_reg2.fit(X,y2)


# Then we train a third regressor on the residual errors made by the second predictor:

# In[52]:

y3 = y3 = tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth = 2, random_state=42)
tree_reg3.fit(X,y3)


# Now we have an ensemble containing three trees. It can make predictions on a new
# instance simply by adding up the predictions of all the trees:

# In[53]:

X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))


# In[57]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)

plt.figure(figsize=(11,11))

plt.subplot(321)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Residuals and tree predictions", fontsize=16)

plt.subplot(322)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Ensemble predictions", fontsize=16)

plt.subplot(323)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.subplot(325)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
plt.xlabel("$x_1$", fontsize=16)

plt.subplot(326)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)

#save_fig("gradient_boosting_plot")
plt.show()


# The figure above represents the predictions of these three trees in the left column, and the
# ensemble’s predictions in the right column. In the first row, the ensemble has just one
# tree, so its predictions are exactly the same as the first tree’s predictions. In the second
# row, a new tree is trained on the residual errors of the first tree. On the right you can
# see that the ensemble’s predictions are equal to the sum of the predictions of the first
# two trees. Similarly, in the third row another tree is trained on the residual errors of
# the second tree. You can see that the ensemble’s predictions gradually get better as
# trees are added to the ensemble.
# 
# A simpler way to train GBRT ensembles is to use Scikit-Learn’s GradientBoostingRe
# gressor class. Much like the RandomForestRegressor class, it has hyperparameters to
# control the growth of Decision Trees (e.g., max_depth, min_samples_leaf, and so on),
# as well as hyperparameters to control the ensemble training, such as the number of
# trees (n_estimators). The following code creates the same ensemble as the previous
# one:

# In[59]:

from sklearn.ensemble import GradientBoostingRegressor


# In[60]:

gbrt = GradientBoostingRegressor(max_depth=2,n_estimators=3, learning_rate=1.0)


# In[61]:

gbrt.fit(X,y)


# The learning_rate hyperparameter scales the contribution of each tree. If you set it
# to a low value, such as 0.1, you will need more trees in the ensemble to fit the training
# set, but the predictions will usually generalize better. This is a regularization technique
# called shrinkage.

# Figure below shows two GBRT ensembles trained with a low
# learning rate: the one on the left does not have enough trees to fit the training set,
# while the one on the right has too many trees and overfits the training set.

# In[63]:

gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
gbrt_slow.fit(X, y)


# In[64]:

plt.figure(figsize=(11,4))

plt.subplot(121)
plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_slow], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)

#save_fig("gbrt_learning_rate_plot")
plt.show()


# In order to find the optimal number of trees, you can use early stopping (see Chapter
# 4). A simple way to implement this is to use the staged_predict() method: it
# returns an iterator over the predictions made by the ensemble at each stage of training
# (with one tree, two trees, etc.). The following code trains a GBRT ensemble with
# 120 trees, then measures the validation error at each stage of training to find the optimal
# number of trees, and finally trains another GBRT ensemble using the optimal
# number of trees:

# In[65]:

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[66]:

X_train, X_val, y_train, y_val = train_test_split(X,y)


# In[67]:

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)


# In[69]:

errors = [mean_squared_error(y_val, y_pred)
         for y_pred in gbrt.staged_predict(X_val)]


# In[70]:

bst_n_estimators = np.argmin(errors)


# In[71]:

gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)


# In[72]:

gbrt_best.fit(X_train, y_train)


# In[74]:

min_error = np.min(errors)
plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot(errors, "b.-")
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
plt.plot([0, 120], [min_error, min_error], "k--")
plt.plot(bst_n_estimators, min_error, "ko")
plt.text(bst_n_estimators, min_error*1.2, "Minimum", ha="center", fontsize=14)
plt.axis([0, 120, 0, 0.01])
plt.xlabel("Number of trees")
plt.title("Validation error", fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)

#save_fig("early_stopping_gbrt_plot")
plt.show()


# The validation errors are represented on the left of the figure above, and the best model’s
# predictions are represented on the right.
# 
# It is also possible to implement early stopping by actually stopping training early
# (instead of training a large number of trees first and then looking back to find the
# optimal number). You can do so by setting warm_start=True, which makes Scikit-
# Learn keep existing trees when the fit() method is called, allowing incremental
# training. The following code stops training when the validation error does not
# improve for five iterations in a row:

# In[75]:

gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)


# In[77]:

min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val ,y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break


# The GradientBoostingRegressor class also supports a subsample hyperparameter,
# which specifies the fraction of training instances to be used for training each tree. For
# example, if subsample=0.25, then each tree is trained on 25% of the training instances,
# selected randomly. As you can probably guess by now, this trades a higher bias
# for a lower variance. It also speeds up training considerably. This technique is called
# Stochastic Gradient Boosting.

# ## Stacking

# Stacking is based on a simple idea: instead of using trivial functions
# (such as hard voting) to aggregate the predictions of all predictors in an ensemble,
# why don’t we train a model to perform this aggregation?
# 
# Each of the bottom three predictors predicts a different value, and then the final predictor
# (called a blender, or a meta learner) takes these predictions as inputs and makes the
# final prediction.
# 
# To train the blender, a common approach is to use a hold-out set. First, the training set is split in two subsets. The first subset is used to train the predictors in the first layer. Next, the first layer predictors are used to make predictions on the second (held-out) set. This ensures that the predictions are “clean,” since the predictors
# never saw these instances during training. Now for each instance in the hold-out set
# there are three predicted values. We can create a new training set using these predicted
# values as input features (which makes this new training set three-dimensional),
# and keeping the target values. The blender is trained on this new training set, so it
# learns to predict the target value given the first layer’s predictions.
# 
# It is actually possible to train several different blenders this way (e.g., one using Linear
# Regression, another using Random Forest Regression, and so on): we get a whole
# layer of blenders. The trick is to split the training set into three subsets: the first one is
# used to train the first layer, the second one is used to create the training set used to
# train the second layer (using predictions made by the predictors of the first layer),
# and the third one is used to create the training set to train the third layer (using predictions
# made by the predictors of the second layer). Once this is done, we can make
# a prediction for a new instance by going through each layer sequentially. 
# 
# Unfortunately, Scikit-Learn does not support stacking directly, but it is not too hard
# to roll out your own implementation.

# In[ ]:



