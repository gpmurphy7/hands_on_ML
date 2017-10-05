
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist


# In[3]:

X,y = mnist["data"], mnist["target"]


# In[4]:

X.shape


# In[5]:

y.shape


# In[6]:

some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)


# In[7]:

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
plt.axis("off")
plt.show()


# In[8]:

y[36000]


# This data set is already split into training and test. First 60,000 are training, remaining 10,000 are test.

# In[9]:

X_train, X_test, y_train, y_test  = X[:60000], X[60000:], y[:60000], y[60000:]


# Shuffle the training set to guarantee all cross-validation folds will be similar, i.e. no missing numbers from any folds. Also some learning algorithms are sensitive to the order of the training instances, and perform poorly if they get too many similar instances in a row. 
# 
# However shuffling may be a bad idea in some contexts, e.g. working with time series data. 

# In[10]:

shuffle_index = np.random.permutation(60000)
X_train, y_train =  X_train[shuffle_index], y_train[shuffle_index]


# ## Binary Classifier
# Simplifying the problem to only identify one digit, e.g. the number five.

# In[11]:

y_train_5 = (y_train == 5) #true for all 5s, false otherwise
y_test_5 = (y_test ==5)


# ### Stochastic Gradient Descent (SGD) Classifier

# In[12]:

from sklearn.linear_model import SGDClassifier


# In[13]:

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


# In[14]:

sgd_clf.predict([some_digit])


# Correctly predicts the digit we looked at earlier.

# ## Performance Measures

# ### Measuring Accuracy Using Cross- Validation

# In[15]:

from sklearn.model_selection import cross_val_score


# In[16]:

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# Above 95% looks good. But what if we were to simply classify every image as not the number 5? 

# In[17]:

from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y = None):
        pass
    def predict(self, X):
        return np.zeros((len(X),1), dtype=bool)


# In[18]:

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf,X_train,y_train_5, cv=3, scoring="accuracy")


# This hase above 90% accuracy. Only about 10% of the images are 5s, so if you always say that an image is not a 5 you will be right about 90% of the time.
# 
# Demonstrates why accuracy is generally not the preferred performance measure for classifiers, especially with skewed datasets where some classes are much more frequent than others.

# ### Confusion Matrix

# General idea is to count the number of times class A was classified as class B, i.e. confused as class B.
# 
# To do this first need a set of predictions so they can be compared to the actual targets.

# In[19]:

from sklearn.model_selection import cross_val_predict


# In[20]:

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


# In[21]:

from sklearn.metrics import confusion_matrix


# In[22]:

confusion_matrix(y_train_5, y_train_pred)


# * Top left shows the number that were correctly classified as non-5s, the true negatives.
# * Top right shows the number that were wrongly classified as 5s, false positives.
# * Bottom left shows the number wrongly classified as non-5s, false negatives.
# * Bottom right shows the number perfectly classified, true positives.
# 
# 

# ### Precision and Recall

# $$precision = \frac{TP}{TP+FP}$$ 

# $$recall = \frac{TP}{TP+FN}$$

# In[23]:

from sklearn.metrics import precision_score, recall_score


# In[24]:

precision_score(y_train_5,y_train_pred)


# In[25]:

recall_score(y_train_5, y_train_pred)


# Can combine the precision and recall into the F1 score. 
# 
# $$ F_1 = \frac{2}{1/P + 1/R} = 2\cdot\frac{PxR}{P+R} = \frac{TP}{TP+\frac{FN+FP}{2}}$$

# In[26]:

from sklearn.metrics import f1_score


# In[27]:

f1_score(y_train_5, y_train_pred)


# Unfortunately, you can’t have it both ways: increasing precision reduces recall, and
# vice versa. This is called the precision/recall tradeoff.
# 
# 
# Instead of calling the classifier's ``predict()`` method you can call its ``decsion_function()`` which returns a score for each instance and then makes predictions based on those scores

# In[28]:

y_scores = sgd_clf.decision_function([some_digit])
y_scores


# In[29]:

threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# In[30]:

#raising threshold
threshold = 200000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# Changing the threshold reduces the recall, this image was actually a 5, but is now falsely classified as non-5. To decide on which threshold to use first get the scores of all the instances in the training set.

# In[31]:

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv =3, method="decision_function")


# In[32]:

from sklearn.metrics import precision_recall_curve


# In[33]:

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# In[34]:

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


# In[35]:

plt.figure(figsize=(8,4))
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()


# Can simply slect the threshold value that gives the best precision/recall tradeoff. Aim for 90% recall, need approx 300,000 threshold. 

# In[36]:

y_train_pred_90 = (y_scores > 300000)


# In[37]:

precision_score(y_train_5, y_train_pred_90)


# In[38]:

recall_score(y_train_5, y_train_pred_90)


# High precision, but low recall. This type of classifier probably won't be much use

# In[39]:

plt.figure()
plt.plot(recalls[:-1], precisions[:-1], "b")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# Precision really starts to fall sharply around 80% recall so you can pick a precision/recall tradeoff just before that drop, e.g. 60% recall

# ### The ROC Curve

# The receiver operating characteristic (ROC) curve is another common tool used with
# binary classifiers. Instead of plotting
# precision versus recall, the ROC curve plots the true positive rate (another name
# for recall) against the false positive rate. The FPR ratio is the ratio of negative instances that are incorrectly classified as positive. It is equal to one minus the true negative rate,
# which is the ratio of negative instances that are correctly classified as negative. The
# TNR is also called specificity. Hence the ROC curve plots sensitivity (recall) versus
# 1 – specificity.

# In[40]:

from sklearn.metrics import roc_curve


# In[41]:

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# In[42]:

def plot_roc_curve(fpr, tpr, label=None):
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# In[43]:

plot_roc_curve(fpr,tpr,thresholds)
plt.show()


# Once again there is a tradeoff: the higher the recall (TPR), the more false positives
# (FPR) the classifier produces. The dotted line represents the ROC curve of a purely
# random classifier; a good classifier stays as far away from that line as possible (toward
# the top-left corner).
# 
# One way to compare classifiers is to measure the area under the curve (AUC). A perfect
# classifier will have a ROC AUC equal to 1, whereas a purely random classifier will
# have a ROC AUC equal to 0.5. Scikit-Learn provides a function to compute the ROC
# AUC:

# In[44]:

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)


# Since the ROC curve is so similar to the precision/recall (or PR)
# curve, you may wonder how to decide which one to use. As a rule
# of thumb, you should prefer the PR curve whenever the positive
# class is rare or when you care more about the false positives than
# the false negatives, and the ROC curve otherwise. For example,
# looking at the previous ROC curve (and the ROC AUC score), you
# may think that the classifier is really good. But this is mostly
# because there are few positives (5s) compared to the negatives
# (non-5s). In contrast, the PR curve makes it clear that the classifier
# has room for improvement (the curve could be closer to the topright
# corner).

# Let’s train a RandomForestClassifier and compare its ROC curve and ROC AUC
# score to the SGDClassifier.The RandomForestClassi
# fier class does not have a decision_function() method. Instead it has a pre
# dict_proba() method. Scikit-Learn classifiers generally have one or the other.The
# predict_proba() method returns an array containing a row per instance and a column
# per class, each containing the probability that the given instance belongs to the
# given class (e.g., 70% chance that the image represents a 5):

# In[45]:

from sklearn.ensemble import RandomForestClassifier


# In[46]:

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")


# In[47]:

#Simple solution to get scores is just use the positive class's probability as the score:
y_scores_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


# In[48]:

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plt.legend(loc="lower right", fontsize=16)
plt.show()


# In[49]:

roc_auc_score(y_train_5,y_scores_forest)


# In[50]:

y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
precision_score(y_train_5, y_train_pred_forest)


# In[51]:

recall_score(y_train_5, y_train_pred_forest)


# So this is much better than the SGD classifier. 

# ## Multiclass Classification 

# Some algorithms (such as Random Forest classifiers or naive Bayes classifiers) are
# capable of handling multiple classes directly. Others (such as Support Vector Machine
# classifiers or Linear classifiers) are strictly binary classifiers. However, there are various
# strategies that you can use to perform multiclass classification using multiple
# binary classifiers.
# 
# For example, one way to create a system that can classify the digit images into 10
# classes (from 0 to 9) is to train 10 binary classifiers, one for each digit (a 0-detector, a
# 1-detector, a 2-detector, and so on). Then when you want to classify an image, you get
# the decision score from each classifier for that image and you select the class whose
# classifier outputs the highest score. This is called the one-versus-all (OvA) strategy
# (also called one-versus-the-rest).
# 
# Can also train a binary classifier for every pair of digit, one-versus-one (OvO) strategy. Can mean training a lot of classifiers but the main advantage
# of OvO is that each classifier only needs to be trained on the part of the training
# set for the two classes that it must distinguish.
# 
# Some algorithms (such as Support Vector Machine classifiers) scale poorly with the
# size of the training set, so for these algorithms OvO is preferred since it is faster to
# train many classifiers on small training sets than training few classifiers on large
# training sets. For most binary classification algorithms, however, OvA is preferred.
# 
# Scikit-Learn detects when you try to use a binary classification algorithm for a multiclass
# classification task, and it automatically runs OvA (except for SVM classifiers for
# which it uses OvO). Let’s try this with the SGDClassifier:

# In[52]:

sgd_clf.fit(X_train, y_train) # y_train, not y_train_5
sgd_clf.predict([some_digit])


# Under the hood,
# Scikit-Learn actually trained 10 binary classifiers, got their decision scores for the
# image, and selected the class with the highest score. To see that this is indeed the case, you can call the decision_function() method and see that the highest score is for class 5 

# In[53]:

some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores


# In[54]:

np.argmax(some_digit_scores)


# In[55]:

sgd_clf.classes_


# In[56]:

sgd_clf.classes_[5]


# If you want to force ScikitLearn to use one-versus-one or one-versus-all, you can use
# the OneVsOneClassifier or OneVsRestClassifier classes.

# In[57]:

from sklearn.multiclass import OneVsOneClassifier


# In[58]:

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))


# In[59]:

ovo_clf.fit(X_train,y_train)


# In[60]:

ovo_clf.predict([some_digit])


# In[61]:

len(ovo_clf.estimators_)


# Training a random forest is just as easy:

# In[62]:

forest_clf.fit(X_train, y_train)


# In[63]:

forest_clf.predict([some_digit])


# This time Scikit-Learn did not have to run OvA or OvO because Random Forest
# classifiers can directly classify instances into multiple classes. You can call
# predict_proba() to get the list of probabilities that the classifier assigned to each
# instance for each class:

# In[64]:

forest_clf.predict_proba([some_digit])


# 90% probability that the image is a 5. 
# 
# To evaluate, as usual you can use cross validation

# In[65]:

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")


# It gets over 84% on all test folds. If you used a random classifier, you would get 10%
# accuracy, so this is not such a bad score, but you can still do much better. For example,
# simply scaling the inputs (as discussed in Chapter 2) increases accuracy above
# 90%:

# In[66]:

from sklearn.preprocessing import StandardScaler


# In[67]:

scaler = StandardScaler()


# In[68]:

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))


# In[69]:

cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


# ## Error Analysis

# Of course, if this were a real project, you would follow the steps in your Machine
# Learning project checklist (see Appendix B): exploring data preparation options, trying
# out multiple models, shortlisting the best ones and fine-tuning their hyperparameters
# using GridSearchCV, and automating as much as possible, as you did in the
# previous chapter. Here, we will assume that you have found a promising model and
# you want to find ways to improve it. One way to do this is to analyze the types of
# errors it makes.

# ### Confusion Matrix

# In[70]:

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)


# In[71]:

conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx


# That’s a lot of numbers. It’s often more convenient to look at an image representation
# of the confusion matrix, using Matplotlib’s matshow() function:

# In[72]:

plt.matshow(conf_mx, cmap = plt.cm.gray)
plt.show()


# This confusion matrix looks fairly good, since most images are on the main diagonal,
# which means that they were classified correctly. The 5s look slightly darker than the
# other digits, which could mean that there are fewer images of 5s in the dataset or that
# the classifier does not perform as well on 5s as on other digits. In fact, you can verify
# that both are the case.
# 
# Let’s focus the plot on the errors. First, you need to divide each value in the confusion
# matrix by the number of images in the corresponding class, so you can compare error
# rates instead of absolute number of errors (which would make abundant classes look
# unfairly bad):

# In[73]:

row_sums = conf_mx.sum(axis=1,keepdims=True)
norm_conf_mx = conf_mx/ row_sums


# Now let’s fill the diagonal with zeros to keep only the errors, and let’s plot the result

# In[74]:

np.fill_diagonal(norm_conf_mx, 0)


# In[75]:

plt.matshow(norm_conf_mx, cmap = plt.cm.gray)
plt.show()


# Now you can clearly see the kinds of errors the classifier makes. Remember that rows
# represent actual classes, while columns represent predicted classes.
# 
# The columns for
# classes 8 and 9 are quite bright, which tells you that many images get misclassified as
# 8s or 9s. Similarly, the rows for classes 8 and 9 are also quite bright, telling you that 8s
# and 9s are often confused with other digits.
# 
# Conversely, some rows are pretty dark,
# such as row 1: this means that most 1s are classified correctly (a few are confused
# with 8s, but that’s about it). Notice that the errors are not perfectly symmetrical; for
# example, there are more 5s misclassified as 8s than the reverse.
# 
# Analyzing the confusion matrix can often give you insights on ways to improve your
# classifier. Looking at this plot, it seems that your efforts should be spent on improving
# classification of 8s and 9s, as well as fixing the specific 3/5 confusion.
# 
# For example,
# you could try to gather more training data for these digits. Or you could engineer
# new features that would help the classifier—for example, writing an algorithm to
# count the number of closed loops (e.g., 8 has two, 6 has one, 5 has none). Or you
# could preprocess the images (e.g., using Scikit-Image, Pillow, or OpenCV) to make
# some patterns stand out more, such as closed loops.
# 
# Analyzing individual errors can also be a good way to gain insights on what your
# classifier is doing and why it is failing, but it is more difficult and time-consuming. For example of 3s and 5s:

# In[76]:

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train==cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train==cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train==cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train==cl_b) & (y_train_pred == cl_b)]


# In[77]:

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


# In[78]:

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()


# The two 5×5 blocks on the left show digits classified as 3s, and the two 5×5 blocks on
# the right show images classified as 5s. Some of the digits that the classifier gets wrong
# (i.e., in the bottom-left and top-right blocks) are so badly written that even a human
# would have trouble classifying them (e.g., the 5 on the 8th row and 1st column truly
# looks like a 3).
# 
# However, most misclassified images seem like obvious errors to us,
# and it’s hard to understand why the classifier made the mistakes it did.3 The reason is
# that we used a simple SGDClassifier, which is a linear model. All it does is assign a
# weight per class to each pixel, and when it sees a new image it just sums up the weighted
# pixel intensities to get a score for each class. So since 3s and 5s differ only by a few
# pixels, this model will easily confuse them.
# 
# The main difference between 3s and 5s is the position of the small line that joins the
# top line to the bottom arc. If you draw a 3 with the junction slightly shifted to the left,
# the classifier might classify it as a 5, and vice versa. In other words, this classifier is
# quite sensitive to image shifting and rotation. So one way to reduce the 3/5 confusion
# would be to preprocess the images to ensure that they are well centered and not too
# rotated. This will probably help reduce other errors as well.

# ## Mulitlabel Classification

# Until now each instance has always been assigned to just one class. In some cases you
# may want your classifier to output multiple classes for each instance. For example,
# consider a face-recognition classifier: what should it do if it recognizes several people
# on the same picture? Of course it should attach one label per person it recognizes. First a simpler example sticking with digits.

# In[79]:

from sklearn.neighbors import KNeighborsClassifier


# In[80]:

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 ==1)
y_multilabel = np.c_[y_train_large, y_train_odd]


# In[81]:

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)


# This code creates a y_multilabel array containing two target labels for each digit
# image: the first indicates whether or not the digit is large (7, 8, or 9) and the second
# indicates whether or not it is odd. The next lines create a KNeighborsClassifier
# instance (which supports multilabel classification, but not all classifiers do) and we
# train it using the multiple targets array. Now you can make a prediction, and notice
# that it outputs two labels:

# In[82]:

knn_clf.predict([some_digit])


# The digit 5 is indeed not large (False) and odd (True).

# There are many ways to evaluate a multilabel classifier, and selecting the right metric
# really depends on your project. For example, one approach is to measure the F1 score
# for each individual label (or any other binary classifier metric discussed earlier), then
# simply compute the average score. This code computes the average F1 score across all
# labels:

# In[ ]:

#does not run, or at least takes a long time
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv = 3)


# In[ ]:

f1_score(y_train, y_train_knn_pred, average = "macro")


# This assumes that all labels are equally important, which may not be the case.One simple option isto give each label a weight equal to its support (i.e., the number of instances with that
# target label). To do this, simply set average="weighted" in the preceding code.

# ## Multioutput Classification

# The last type of classification task we are going to discuss here is called multioutputmulticlass
# classification (or simply multioutput classification). It is simply a generalization
# of multilabel classification where each label can be multiclass (i.e., it can have
# more than two possible values).
# 
# To illustrate this, let’s build a system that removes noise from images. It will take as
# input a noisy digit image, and it will (hopefully) output a clean digit image, represented
# as an array of pixel intensities, just like the MNIST images. Notice that the
# classifier’s output is multilabel (one label per pixel) and each label can have multiple
# values (pixel intensity ranges from 0 to 255). It is thus an example of a multioutput
# classification system.
# 
# **NOTE**: The line between classification and regression is sometimes blurry,
# such as in this example. Arguably, predicting pixel intensity is more
# akin to regression than to classification. Moreover, multioutput
# systems are not limited to classification tasks; you could even have
# a system that outputs multiple labels per instance, including both
# class labels and value labels.
# 
# Let’s start by creating the training and test sets by taking the MNIST images and
# adding noise to their pixel intensities using NumPy’s randint() function. The target
# images will be the original images:

# In[84]:

noise = np.random.randint(0,100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0,100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test


# In[85]:

knn_clf.fit(X_train_mod, y_train_mod)


# In[89]:

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")


# In[90]:

some_index = 5500
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)


# Now to compare this reconstructed image to the modified and the original unaltered test set. Below it looks like we managed to remove the noise and reconstruct the image nicely. 

# In[91]:

plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
plt.show()


# ## KNN Classifier

# In[92]:

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)
knn_clf.fit(X_train, y_train)


# In[93]:

y_knn_pred = knn_clf.predict(X_test)


# In[94]:

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_knn_pred)


# In[ ]:



