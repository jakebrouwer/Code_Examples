from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from .data_loader import DataLoader


class AutoEncoder(DataLoader):
    """
    A class designed to set up an AutoEncoder for unsupervised data classification

    The resulting autoencoder data will be tested against a classical machine learning model for comparison
    """

    def __init__(self,data:str = 'breast_cancer'):
        """
        The constructor for the AutoEncoder class

        Args:
            data (str, optional): the dataset to load and train the autoencoder on. Defaults to 'breast_cancer'.
        """
        #set initial data for check later on
        self._initial_data = data
        #run _load_data() inherited from DataLoader
        self._load_data(data,train_ratio=0.8)
        #set up additional class variables
        self.confusion_matrix = None
        self.classification_report = None
        self.autoencoder_data = None

    def _build_rfc(self,X_ae,y):
        """
        Build and RandomForestClassifier to test the Autoencoder data on

        Args:
            X_ae (array_like): The data built from the latent space of the autoencoder
            y (array_like): The labeled data to test against X_ae. the autoencoder will never see this data.
        """
        #set up train test split
        X_Train, X_Test, y_Train, y_Test = train_test_split(
            X_ae,
            y,
            test_size = 0.30,
            random_state = 101
            )
        #train a RandomForestClassifier on a training set of the autoencoder data
        trainedforest = RandomForestClassifier(n_estimators=200).fit(X_Train,y_Train)
        #predict on a test set of the autoencoder data
        predictionforest = trainedforest.predict(X_Test)
        #get results
        self.confusion_matrix = confusion_matrix(y_Test,predictionforest)
        self.classification_report = classification_report(y_Test,predictionforest)
        print(self.confusion_matrix)
        print(self.classification_report)

    def build_autoencoder(self):
        """
        Builds the autoencoder itself.
        """

        X = self._X_train
        y = self._y_train
        if "mushroom" in self._initial_data:
            X = pd.get_dummies(X, prefix_sep='_')
            X = StandardScaler().fit_transform(X)
            y = LabelEncoder().fit_transform(y)

        #build ae
        input_layer = keras.layers.Input(shape=X.shape[1])
        #encoder
        first_encoder = keras.layers.Dense(64,activation='relu')(input_layer) #reduce dimensionality down to 64
        second_encoder = keras.layers.Dense(16,activation='relu')(first_encoder) #now at 16 dimensions
        encoded = keras.layers.Dense(3,activation='relu')(second_encoder) #dimensionality is reduced to 3 dimensions
        #decoder
        first_decoder = keras.layers.Dense(16,activation='relu')(encoded)#slowly build dimensionality back up
        second_decoder = keras.layers.Dense(64,activation='relu')(first_decoder)
        decoded = keras.layers.Dense(X.shape[1],activation='relu')(second_decoder) #back to original dimensions
        #build model
        ae = keras.models.Model(input_layer,decoded)
        #compile
        ae.compile(optimizer='adam',loss='binary_crossentropy')

        #get two random samples of X 
        ae_X_train, ae_X_test = train_test_split(X,test_size=0.8, random_state=42)
        #fit the autoencoder. ae_X_train is given twice because we are trying to find an output that is a close reconstruction of the input data
        # the autoencoder is trying to learn how to reconstruct ae_X_train with three dimensional data
        ae.fit(
            ae_X_train,
            ae_X_train,
            epochs=100,
            batch_size=300,
            shuffle=True,
            verbose = 30,
            validation_data=(ae_X_test, ae_X_test)
        )
        #build a model of JUST the fitted latent space layer of the autoencoder
        encoder = keras.models.Model(input_layer, encoded)

        #predict get the predition data from the latent space
        self.autoencoder_data = encoder.predict(X) #this is now the new, reduced data with the same size as X

        #test the latent space data against the random forest classifier
        self._build_rfc(self.autoencoder_data,y)
        #plug new data into a classical model to evaluate how successful the dimensionality reduction was
        #high scores indicate not alot of data was lost and autoencoder was successful at determining classes

        #this data could also be clustered to verify class determination but I've not got the time to do so at present
