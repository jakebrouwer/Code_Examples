from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier


X = pd.read_csv("C:\\Users\\jrbro\\Downloads\\mushrooms.csv")
Y = X.pop('class')
X = pd.get_dummies(X, prefix_sep='_')
X = StandardScaler().fit_transform(X)
Y = LabelEncoder().fit_transform(Y)





input_layer = keras.layers.Input(shape=X.shape[1])
encoded = keras.layers.Dense(3,activation='relu')(input_layer)
decoded = keras.layers.Dense(X.shape[1],activation='relu')(encoded)
ae = keras.models.Model(input_layer,decoded)
ae.compile(optimizer='adam',loss='binary_crossentropy')

X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=101)

ae.fit(
    X1,
    Y1,
    epochs=100,
    batch_size=300,
    shuffle=True,
    verbose = 30,
    validation_data=(X2, Y2)
)

encoder = keras.models.Model(input_layer, encoded)
X_ae = encoder.predict(X)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_ae, Y, 
                                                    test_size = 0.30, 
                                                    random_state = 101)
trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train,Y_Train)
predictionforest = trainedforest.predict(X_Test)
print(predictionforest)
print(confusion_matrix(Y_Test,predictionforest))
print(classification_report(Y_Test,predictionforest))

