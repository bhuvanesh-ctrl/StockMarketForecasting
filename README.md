# README

## Stock Price Forecasting using Deep Learning

This project aims to predict stock prices using a Long Short-Term Memory (LSTM) model. The model is trained and tested on stock data fetched from Yahoo Finance. The primary components include data loading, preprocessing, feature extraction, model building, training, and evaluation.

### Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Model Testing](#model-testing)
7. [Results](#results)

### Installation

1. Clone the repository:
    ```bash
    https://github.com/bhuvanesh-ctrl/StockMarketForecasting.git
    cd StockMarketForecasting
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run the project, use the following command:
```bash
python main.py training  # For training and testing the model
python main.py testing   # For testing the model only
```

### Project Structure

- `main.py`: Main script to run the training and testing process.
- `data/`: Directory to store data.
- `model/`: Directory to save the trained model.
- `src/`: Source code directory.
    - `load_data.py`: Contains `LoadData` class for loading stock data.
    - `prepare_data.py`: Contains `PrepareData` class for preprocessing and feature extraction.
    - `model_training.py`: Contains `ModelTraining` class for building and training the model.
    - `model_testing.py`: Contains `ModelTesting` class for evaluating the model.
- `README.md`: This file.

### Data Preparation

The data preparation involves the following steps:
1. **Loading Data**: Fetches stock data from Yahoo Finance using `yfinance` library.
2. **Feature Extraction**: Computes technical indicators like EMA, MACD, and signal line.
3. **Data Splitting**: Splits data into training (70%), validation (20%), and testing (10%) sets.
4. **Normalization**: Normalizes the data using `MinMaxScaler`.
5. **Building Features and Targets**: Constructs features and target variables using time steps.

### Model Training

The LSTM model is built and trained as follows:
1. **Building the Model**: Constructs an LSTM model with specified layers and regularization.
2. **Training the Model**: Trains the model using the training data and validates it using the validation data.
3. **Saving the Model**: Saves the trained model to `model/model.keras`.

### Model Testing

The trained model is evaluated using the test data:
1. **Loading the Model**: Loads the trained model from the specified path.
2. **Evaluating the Model**: Computes evaluation metrics like RMSE, MAE, and R².
3. **Plotting Predictions**: Plots the predicted vs. observed stock prices.

### Results

The model evaluation metrics are printed and the predicted vs. observed stock prices are plotted.

---

For further details, refer to the comments in the code and the inline documentation within each class and function.
