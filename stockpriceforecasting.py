import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import L2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Optional
import matplotlib.dates as mdates
import os
import sys
import warnings
warnings.filterwarnings("ignore")
# Set the logging level to ERROR to suppress informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class LoadData:
    """Load stock data"""

    def __init__(self, ticker: str, start_date: str = "2000-01-01") -> None:
        """
        Initialize the model with the given stock ticker and start date.
        
        Args:
        - ticker (str): Stock ticker symbol.
        - start_date (str): Start date for collecting data.
        
        Returns:
        - None
        """
        self.ticker = ticker
        self.start_date = start_date
        self.today = date.today().strftime("%Y-%m-%d")
        
    def load_data(self) -> pd.DataFrame:
        """
        Load stock data from Yahoo Finance.

        Args:
        -None
        
        Returns:
        - pd.DataFrame: DataFrame containing the loaded stock data.
        """
        data: pd.DataFrame = yf.download(self.ticker, self.start_date, self.today)
        data.reset_index(inplace=True)
        return data

class PrepareData:
    """Prepare the data for modeling"""

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the model with MinMaxScaler.

        Args:
        - data (pd.DataFrame): The input data.
        
        Returns:
        - None
        """
        self.data = data
        self.scaler = MinMaxScaler(feature_range=(0,1))

    def feature_extraction(self) -> None:
        """
        Extract features such as EMA and MACD.
        
        Args:
        -None
        
        Returns:
        - None
        """
        self.data.set_index('Date', inplace=True)
        self.data['ema_12'] = self.data['Close'].ewm(span=12, adjust=False).mean()
        self.data['ema_26'] = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['macd'] = self.data['ema_12'] - self.data['ema_26']
        self.data['signal_line'] = self.data['macd'].ewm(span=9, adjust=False).mean()
        for lag in range(1, 6):
            self.data[f'macd_{lag}_dyago'] = self.data['macd'].shift(lag).bfill()
        self.data.drop(self.data.columns[[0,1,2,4,5]], axis=1, inplace=True)
    
    def train_test_val_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, test, and validation sets. (70:20:10)

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test data and list of dates.
        """
        training_data_len = int(np.ceil(len(self.data) * 0.70))
        val_data_len = int(np.ceil(len(self.data) * 0.90))
        train_data = self.data.iloc[0:training_data_len, :]
        val_data = self.data.iloc[training_data_len-100:val_data_len, :]
        test_data = self.data.iloc[val_data_len - 100:, :]
        dates = test_data.index.to_numpy()
        return train_data, val_data, test_data, dates
    
    def normalize_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize data using MinMaxScaler.

        Args:
        - train_data (pd.DataFrame): Train data.
        - val_data (pd.DataFrame): Validation data.
        - test_data (pd.DataFrame): Test data.

        Returns:
        - Tuple[np.ndarray, np.ndarray, np.ndarray]: Normalized train, validation, and test data.
        """
        train_scaled_data = self.scaler.fit_transform(train_data)
        val_scaled_data = self.scaler.transform(val_data)
        test_scaled_data = self.scaler.transform(test_data)
        return train_scaled_data, val_scaled_data, test_scaled_data
    
    def build_features_and_target(self, scaled_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build features and target variables using time steps.

        Args:
        - scaled_data (np.ndarray): Either train, val, or test data.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Features and target variables.
        """
        X, y = [], []
        for i in range(100, len(scaled_data)):
            X.append(scaled_data[i-100:i, 1:])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

class ModelTraining:
    """Model Training"""

    def __init__(self, l2_reg: float = 9e-4, learning_rate: float = 7.5e-6, epochs: int = 300, batch_size: int = 1024) -> None:
        """
        Initialize the model.

        Args:
        - l2_reg (float): L2 regularization parameter.
        - learning_rate (float): Learning rate for the optimizer.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        
        Returns:
        - None
        """
        self.l2_reg = l2_reg
        self.model = None
        self.history = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def build_model(self) -> Sequential:
        """
        Build LSTM model.

        Args:
        -None
        
        Returns:
        - Sequential: The constructed LSTM model.
        """
        model = Sequential()
        model.add(LSTM(512, return_sequences=True, input_shape=(100, 9), kernel_regularizer=L2(self.l2_reg)))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=False, kernel_regularizer=L2(self.l2_reg)))
        model.add(Dropout(0.3))
        model.add(Dense(25))
        model.add(Dense(1))
        return model
    
    def train_model(self, model: Sequential, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train the LSTM model.
        
        Args:
        - model (Sequential): The LSTM model to train.
        - X_train (np.ndarray): The training features.
        - y_train (np.ndarray): The training target.
        - X_val (np.ndarray): The validation features.
        - y_val (np.ndarray): The validation target.
        
        Returns:
        - None
        """
        optimizer = Adam(learning_rate=self.learning_rate)
        def root_mean_squared_error(y_true, y_pred):
            return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

        model.compile(optimizer = optimizer, loss = root_mean_squared_error, metrics = ['MAE'])
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batch_size, verbose=True)
        self.history = history
        self.model = model
        self.model.save("model/model.keras")
        print("Model saved successfully.")

    def plot_loss_graphs(self) -> None:
        """
        Plot training and validation loss graphs.
        
        Args:
        -None
        
        Returns:
        - None
        """
        history = self.history
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['MAE'], label='Train MAE')
        plt.plot(history.history['val_MAE'], label='Validation MAE')
        plt.title('Model MAE')
        plt.ylabel('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

class ModelTesting:
    """Model Testing"""

    def __init__(self, model: Optional[tf.keras.Model] = None, model_path: str = "model/model.keras") -> None:
        """
        Initialize the ModelTesting class with a trained model.

        Args:
        - model (Optional[tf.keras.Model]): Trained model object (optional). If not provided, the model will be loaded from the specified path.
        - model_path (str): Path to load the model from if `model` is not provided.
        """
        self.y_pred: Optional[np.ndarray] = None
        def root_mean_squared_error(y_true, y_pred):
            return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
        if model is not None:
            self.model = model
        else:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
            else:
                raise FileNotFoundError(f"Model file '{model_path}' not found.")
                
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float]:
        """
        Evaluate the model performance.
        
        Args:
        - X_test (np.ndarray): Test features.
        - y_test (np.ndarray): Observed test values.
        
        Returns:
        - Tuple[float, float, float]: Mean squared error, Mean absolute error, R-squared.
        """
        
        def root_mean_squared_error(y_true, y_pred):
            return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
        self.y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, self.y_pred)
        r2 = r2_score(y_test, self.y_pred)
        return rmse, mae, r2
        
    def predicted_observed(self, y_test: np.ndarray, dates: np.ndarray) -> None:
        """
        Plot predicted vs. observed values.

        Args:
        - y_test (np.ndarray): Observed values.
        """

        dates = dates[100:]
        plt.plot(dates, y_test, label='Observed')
        plt.plot(dates, self.y_pred, label='Predicted')
        plt.title('Predicted vs Observed Stock Price')
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

        # Set ticks at specific interval, e.g., every 5th date
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=100))

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.xlabel('Date')
        plt.ylabel('Normalized Stock Price')
        plt.legend()
        plt.tight_layout() 
        plt.show()

        
def main(training: bool) -> None:
    
    if training:
        # Initialize the model

        print("\n-----------------Data loading started-------------------------\n")
        loading = LoadData('GOOG')

        # Load data
        data = loading.load_data()

        print("\n----------------------Data loaded successfully----------------------")

        print("\n-----------------Data preparation started----------------------")

        # Prepare data
        preparing = PrepareData(data)

        # Feature extraction
        preparing.feature_extraction()

        # Split data
        train_data, val_data, test_data, dates = preparing.train_test_val_split()

        # Normalize data
        train_scaled_data, val_scaled_data, test_scaled_data = preparing.normalize_data(train_data, val_data, test_data)

        # Build features and target
        X_train, y_train = preparing.build_features_and_target(train_scaled_data)
        X_val, y_val = preparing.build_features_and_target(val_scaled_data)
        X_test, y_test = preparing.build_features_and_target(test_scaled_data)

        print("\n----------------------Data preparation successfully------------------")

        print("\n-----------------Model Training started-------------------------")
        modelling = None
        # Initialize modeling
        modelling = ModelTraining()

        # Build model
        lstm_model = modelling.build_model()

        # Train model
        modelling.train_model(lstm_model, X_train, y_train, X_val, y_val)

        # Plot loss graphs
        modelling.plot_loss_graphs()

        print("\n----------------------Model Training completed successfully------------")

        print("\n-----------------Model testing started--------------------")

        # Test model
        testing = ModelTesting(lstm_model)

        # Evaluate model
        rmse, mae, r2 = testing.evaluate_model(X_test, y_test)

        print("\n----------------------Model Testing completed successfully---------------")

        print("\nHere are the model results.\n")

        print("Root Mean Squared Error on Test Data  :", rmse)
        print("Mean Absolute Error on Test Data      :", mae)
        print("R-squared on Test Data                :", r2)

        testing.predicted_observed(y_test, dates)
    
    else:        
        
        print("\n-----------------Data loading started-------------------------\n")
        loading = LoadData('GOOG')

        # Load data
        data = loading.load_data()

        print("\n----------------------Data loaded successfully----------------------")

        print("\n-----------------Data preparation started----------------------")

        # Prepare data
        preparing = PrepareData(data)

        # Feature extraction
        preparing.feature_extraction()

        # Split data
        train_data, val_data, test_data, dates = preparing.train_test_val_split()

        # Normalize data
        train_scaled_data, val_scaled_data, test_scaled_data = preparing.normalize_data(train_data, val_data, test_data)

        # Build features and target
        X_train, y_train = preparing.build_features_and_target(train_scaled_data)
        X_val, y_val = preparing.build_features_and_target(val_scaled_data)
        X_test, y_test = preparing.build_features_and_target(test_scaled_data)

        print("\n----------------------Data preparation successfully------------------")
        

        print("\n-----------------Model testing started--------------------")

        # Test model
        testing = ModelTesting(None)

        # Evaluate model
        rmse, mae, r2 = testing.evaluate_model(X_test, y_test)

        print("\n----------------------Model Testing completed successfully---------------")

        print("\nHere are the model results.\n")

        print("Root Mean Squared Error on Test Data  :", rmse)
        print("Mean Absolute Error on Test Data      :", mae)
        print("R-squared on Test Data                :", r2)

        testing.predicted_observed(y_test, dates)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        temp = sys.argv[1]
        if temp == 'training':
            main(training=True)
        else:
            main(training=False)
    else:
        main(training=True)    
    
