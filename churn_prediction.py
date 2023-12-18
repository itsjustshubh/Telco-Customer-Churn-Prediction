import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, concatenate
from tensorflow.keras.models import Model


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset.

    Parameters:
    filepath (str): The file path to the dataset.

    Returns:
    tuple: Returns preprocessed features (X), target variable (y),
           a preprocessor object, and the original DataFrame.
    """
    df = pd.read_csv(filepath)

    # Convert 'TotalCharges' to numeric and handle non-numeric entries
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Now, you can safely impute missing values
    imputer = SimpleImputer(strategy='mean')
    df['TotalCharges'] = imputer.fit_transform(df[['TotalCharges']])

    # Defining categorical and numerical features
    categorical_features = ['gender', 'PaymentMethod', 'InternetService']
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)])

    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    return X, y, preprocessor, df


def perform_eda(df):
    """
    Perform exploratory data analysis (EDA) on the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame on which to perform EDA.

    Returns:
    None
    """
    print(df.describe())

    # Select only numeric columns for correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_cols.corr(), annot=True)
    plt.show()
    # Add more EDA as needed


def create_nn_model(input_shape):
    """
    Create a Neural Network model.

    Parameters:
    input_shape (tuple): Shape of the input data.

    Returns:
    Model: A compiled TensorFlow neural network model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def create_rnn_model(input_shape):
    """
    Create a Recurrent Neural Network (RNN) model.

    Parameters:
    input_shape (tuple): Shape of the input data, specifically for RNN.

    Returns:
    Model: A compiled TensorFlow RNN model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.SimpleRNN(50),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_model(input_shape):
    """
    Create a Convolutional Neural Network (CNN) model.

    Parameters:
    input_shape (tuple): Shape of the input data, specifically for CNN.

    Returns:
    Model: A compiled TensorFlow CNN model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape(target_shape=input_shape + (1,), input_shape=input_shape),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_lstm_model(cnn_input_shape, lstm_input_shape):
    """
    Create a hybrid CNN-LSTM model.

    Parameters:
    cnn_input_shape (tuple): Shape of the input data for the CNN part of the model.
    lstm_input_shape (tuple): Shape of the input data for the LSTM part of the model.

    Returns:
    Model: A compiled TensorFlow hybrid CNN-LSTM model.
    """
    # CNN branch
    cnn_input = Input(shape=cnn_input_shape)
    cnn = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_input)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)

    # LSTM branch
    lstm_input = Input(shape=lstm_input_shape)
    lstm = LSTM(50, return_sequences=True)(lstm_input)
    lstm = LSTM(50)(lstm)

    # Concatenate both branches
    combined = concatenate([cnn, lstm])

    # Fully connected layers
    z = Dense(64, activation='relu')(combined)
    z = Dense(1, activation='sigmoid')(z)

    # Create model
    model = Model(inputs=[cnn_input, lstm_input], outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train and evaluate different machine learning models.

    Parameters:
    X_train (array-like): Training features.
    X_test (array-like): Testing features.
    y_train (array-like): Training target variable.
    y_test (array-like): Testing target variable.

    Returns:
    None
    """
    # Logistic Regression
    lr_classifier = LogisticRegression()
    lr_classifier.fit(X_train, y_train)
    lr_predictions = lr_classifier.predict(X_test)
    print("Logistic Regression accuracy: ", accuracy_score(y_test, lr_predictions))
    print(classification_report(y_test, lr_predictions))

    # Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    dt_predictions = dt_classifier.predict(X_test)
    print("Decision Tree accuracy: ", accuracy_score(y_test, dt_predictions))
    print(classification_report(y_test, dt_predictions))

    # Random Forest Classifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_test)
    print("Random Forest accuracy: ", accuracy_score(y_test, rf_predictions))
    print(classification_report(y_test, rf_predictions))

    # Support Vector Machine (SVM)
    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)
    svm_predictions = svm_classifier.predict(X_test)
    print("SVM accuracy: ", accuracy_score(y_test, svm_predictions))
    print(classification_report(y_test, svm_predictions))

    # K-Nearest Neighbors (KNN)
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train, y_train)
    knn_predictions = knn_classifier.predict(X_test)
    print("KNN accuracy: ", accuracy_score(y_test, knn_predictions))
    print(classification_report(y_test, knn_predictions))

    # Gradient Boosting Classifier
    gb_classifier = GradientBoostingClassifier()
    gb_classifier.fit(X_train, y_train)
    gb_predictions = gb_classifier.predict(X_test)
    print("Gradient Boosting Classifier accuracy: ", accuracy_score(y_test, gb_predictions))
    print(classification_report(y_test, gb_predictions))

    # Naive Bayes Classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    nb_predictions = nb_classifier.predict(X_test)
    print("Naive Bayes accuracy: ", accuracy_score(y_test, nb_predictions))
    print(classification_report(y_test, nb_predictions))

    # Neural Network - Assuming X_train and X_test are already preprocessed
    nn_model = create_nn_model(input_shape=(X_train.shape[1],))
    nn_model.fit(X_train, y_train, epochs=10, batch_size=10)
    nn_predictions = nn_model.predict(X_test)
    nn_predictions = (nn_predictions > 0.5).astype(int)
    print("Neural Network accuracy: ", accuracy_score(y_test, nn_predictions))
    print(classification_report(y_test, nn_predictions))

    # Reshape input for RNN - Assuming X_train and X_test are 2D arrays
    X_train_rnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_rnn = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # RNN
    rnn_model = create_rnn_model(input_shape=(1, X_train.shape[1]))
    rnn_model.fit(X_train_rnn, y_train, epochs=10, batch_size=10)
    rnn_predictions = rnn_model.predict(X_test_rnn)
    rnn_predictions = (rnn_predictions > 0.5).astype(int)
    print("RNN accuracy: ", accuracy_score(y_test, rnn_predictions))
    print(classification_report(y_test, rnn_predictions))

    # Convolutional Neural Network
    # CNN - Assuming X_train and X_test are 2D arrays
    cnn_input_shape = (X_train.shape[1],)
    cnn_model = create_cnn_model(cnn_input_shape)
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=10)
    cnn_predictions = cnn_model.predict(X_test)
    cnn_predictions = (cnn_predictions > 0.5).astype(int)
    print("CNN accuracy: ", accuracy_score(y_test, cnn_predictions))
    print(classification_report(y_test, cnn_predictions))

    # Convolutional Neural Network - LSTM
    # Prepare data for CNN-LSTM model
    cnn_input_shape = (X_train.shape[1], 1)  # Reshape for CNN
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    lstm_input_shape = (X_train.shape[1], 1)  # Reshape for LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # CNN-LSTM model
    cnn_lstm_model = create_cnn_lstm_model(cnn_input_shape, lstm_input_shape)
    cnn_lstm_model.fit([X_train_cnn, X_train_lstm], y_train, epochs=10, batch_size=10)
    cnn_lstm_predictions = cnn_lstm_model.predict([X_test_cnn, X_test_lstm])
    cnn_lstm_predictions = (cnn_lstm_predictions > 0.5).astype(int)
    print("CNN-LSTM Model accuracy: ", accuracy_score(y_test, cnn_lstm_predictions))
    print(classification_report(y_test, cnn_lstm_predictions))


def tune_hyperparameters(X_train, y_train):
    """
    Perform hyperparameter tuning for Logistic Regression.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training target variable.

    Returns:
    Model: The best estimator after hyperparameter tuning.
    """
    lr_pipeline = Pipeline(steps=[
        ('classifier', LogisticRegression(solver='lbfgs'))
    ])

    param_grid = {
        'classifier__C': [0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(lr_pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best parameters for Logistic Regression: ", grid_search.best_params_)

    return grid_search.best_estimator_


def main():
    """
    Main function to execute the data analysis and modeling process.

    Parameters:
    None

    Returns:
    None
    """
    filepath = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'  # Update with your dataset path
    X, y, preprocessor, df = load_and_preprocess_data(filepath)

    # Perform EDA
    perform_eda(df)

    # Apply preprocessing to the entire dataset
    X_transformed = preprocessor.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # Handling class imbalance with SMOTE on the transformed training data
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train and evaluate models on the resampled data
    train_and_evaluate(X_train_resampled, X_test, y_train_resampled, y_test)

    # Hyperparameter tuning on the resampled data
    best_lr_model = tune_hyperparameters(X_train_resampled, y_train_resampled)

    # Model Interpretation
    # Adjust this part based on the model and your requirements
    importance = best_lr_model.named_steps['classifier'].coef_[0]
    # Update feature_names assignment to match the transformed data
    feature_names = preprocessor.get_feature_names_out()
    for i, v in enumerate(importance):
        print('Feature:', feature_names[i], 'Score:', v)

# Run the main function
if __name__ == "__main__":
    main()
