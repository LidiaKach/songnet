from songsearch.nn2 import faceRecoModel
from songsearch import config
from songsearch import utils
import seaborn as sns;sns.set()
import os
from sklearn import metrics
import pickle
from sklearn import svm



x_train, y_train = utils.load_data(os.path.join(os.getcwd(), config.DATA_PATH), "data.npz")
x_test, y_test = utils.load_data(os.path.join(os.getcwd(), config.DATA_PATH), "data_test.npz")

trained_model = utils.reconstruct_net(faceRecoModel, "500_model_weights.h5", config.SPEC_SHAPE, config.EMBEDDING_DIM)


X_train_trm = trained_model.predict(x_train)
X_test_trm = trained_model.predict(x_test)

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train_trm, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test_trm)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

with open(os.path.join(os.getcwd(), config.CLASSIFIER_PATH, "svm_classifier.pickle"), 'wb') as f:
    pickle.dump(clf, f)


