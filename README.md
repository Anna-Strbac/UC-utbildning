# Bitcoin Price Prediction using LSTM

This project aims to build a machine learning model and application for predicting Bitcoin prices using LSTM (Long Short-Term Memory) networks. The data is processed and cleaned, with feature engineering applied to identify key factors influencing price movements.

The application is built in Python and uses libraries like Pandas and NumPy for data manipulation, Scikit-learn for model evaluation, and TensorFlow (Keras) for training the LSTM model. Matplotlib is used for visualizing results, and joblib allows saving and loading the trained models.

The objective is to create a robust price prediction model that can aid in decision-making for trading or financial analysis.
The backtesting for 2024 framework allows for evaluating the model's performance in a real-world simulation.


## Technologies Used
- **Python**: The primary programming language used for this project.
- **TensorFlow**: For building and training the LSTM model.
- **Keras**: High-level neural network API used for creating the LSTM model.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For splitting data into training and test sets and evaluating model performance.
- **SHAP**: For model explainability (optional).


## Requirements
To run this project, install the required Python libraries listed in the requirements.txt file.


## Instructions to set up application

To get the updated dataset, download the CSV file from CoinCodex https://coincodex.com/crypto/bitcoin/historical-data/, covering the range from 2010-07-10 to today. After downloading the file:

Replace the file_path in Part 1 of the .ipynb file with the location of the downloaded CSV file on your computer. Download file Bitcoin_Historical_Data and change the path to it in Part 3. This is necessary to handle missing Volume values in dataset.

To run the application (app.py), update the file paths in the script:

Model file: Download the pre-trained model bitcoin_model_t1_augmented.h5 and set its file path in the script.
Data files: The files bitcoin_data_scaled.csv and bitcoin_scaler_close.pkl will be created if you run the code in the .ipynb notebook up to Part 5: "Model Training". After generating these files, move them to your desired location and update their file paths in the script (data_path and scaler_path).

Once the setup is complete, the application (app.py) will provide the next day's Bitcoin closing price prediction based on the model.

As an alternative, you can download the last 10 days of OHLCV and Market Cap values from CoinCodex to predict the value for the next day.


## Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch.
Make your changes and commit them.
Push to your branch.
Open a pull request.
Please follow the existing code style and include tests where applicable.


## Licence
This project is licensed under the MIT License. See the LICENSE file for details.


## Contact
Name: Anna Strbac
Email: ann.strbac@gmail.com
GitHub: Anna-Strbac