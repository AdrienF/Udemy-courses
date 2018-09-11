# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

#%% Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding Country label to value
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# Encoding Gender label to value
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Categorical variables (Country and Gender) are independant and
# should be transformed to N-1 dummy variables
country_idx = 1
onehotencoder = OneHotEncoder(categorical_features = [country_idx])
# Gender does not need to be transfomed as we will simply replace it
# with a binary dummy variable (eg. is_male)
X = onehotencoder.fit_transform(X).toarray()
# From here :
#    X[0:3] will code the country
#    X[4] will code the gender
# As Country are mutually exclusive, we remove the first column of X,
# to avoid falling in the "dummy variable trap"
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

modelId = 0

if modelId==0:
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    # loss is binary_crossentropy because output is binary
    # in case of categories, we can choose categorical_crossentropy
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
elif modelId==1:
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
else:
    print("/!\\ Select a classifier /!\\")

# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Plotting the training historyplt.plot(history.history['acc'])
print(history.history.keys())
plt.plot(history.history['acc'], 'r', label='accuraccy')
plt.plot(history.history['loss'], 'b', label = 'loss')
plt.title('model training')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

#%% Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#%% Exercise
# UseANN to predict if the following client is likely to leave the bank
client = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]], dtype=object)
client[:,1] = labelencoder_X_1.transform(client[:,1])
client[:,2] = labelencoder_X_2.transform(client[:,2])
client_as_array = onehotencoder.transform(client).toarray()[:, 1:]
#scale data
client_as_array = sc.transform(client_as_array)
likely_to_leave = classifier.predict(client_as_array)
print("Chance of client to leave : {:.1f}%%".format(likely_to_leave[0,0]*100))

#%% Part 4 Evaluating model performance
# K-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 64, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 6)

# Drop-out : avoid overfitting

def build_classifier_with_dropout():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#%% Parameter tuning to get 86% accuracy
from sklearn.model_selection import GridSearchCV
def build_classifier_with_hyper_parameters(optimizer_name):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer_name, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier_with_hyper_parameters)
hyper_params = {
        'batch_size' : [25, 32],
        'epochs' : [100, 500],
        'optimizer_name' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, param_grid = hyper_params, cv = 10, n_jobs = 4, scoring = 'accuracy')
grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
