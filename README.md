# ⚡ Electricity Consumption Forecasting Project

A machine learning project that predicts hourly electricity consumption using time-series features and historical usage patterns. This project includes data preprocessing, feature engineering, model training with hyperparameter tuning, and a Streamlit web application for real-time predictions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [License](#license)

## 🎯 Overview

This project demonstrates a complete machine learning workflow for time-series forecasting. It predicts next-hour electricity consumption based on:
- **Temporal features**: hour, day, weekday, month
- **Lag features**: previous 1-3 hours of consumption
- **Rolling statistics**: 3-hour moving average

The project includes:
1. **Exploratory Data Analysis** in Jupyter Notebook
2. **Feature Engineering** for time-series data
3. **Model Training** with Ridge Regression, Random Forest, and Gradient Boosting
4. **Hyperparameter Tuning** using GridSearchCV
5. **Interactive Web App** built with Streamlit

## ✨ Features

- **Time-Series Feature Engineering**: Extracts temporal patterns and creates lag features
- **Multiple ML Models**: Compares Ridge Regression, Random Forest, and Gradient Boosting
- **Hyperparameter Optimization**: Uses GridSearchCV for optimal model tuning
- **Model Evaluation**: Comprehensive metrics (MAE, RMSE, R²) with visualizations
- **Model Persistence**: Saves trained model using joblib
- **Web Application**: User-friendly Streamlit interface for predictions

## 📊 Dataset

The dataset (`electricity.csv`) contains hourly electricity consumption data with the following structure:
- **datetime**: Timestamp of the measurement
- **energy_kwh**: Electricity consumption in kilowatt-hours (kWh)

Sample data format:
```
datetime,energy_kwh
2020-01-01 00:00:00,3.45
2020-01-01 01:00:00,3.22
...
```

## 🔬 Methodology

### 1. Data Preprocessing
- Parse datetime column
- Handle missing values by dropping rows with NaN

### 2. Feature Engineering
**Temporal Features:**
- `hour`: Hour of the day (0-23)
- `day`: Day of the month (1-31)
- `weekday`: Day of the week (0-6)
- `month`: Month of the year (1-12)

**Lag Features:**
- `lag1`: Energy consumption 1 hour ago
- `lag2`: Energy consumption 2 hours ago
- `lag3`: Energy consumption 3 hours ago

**Rolling Statistics:**
- `rolling_mean`: 3-hour moving average of energy consumption

### 3. Train-Test Split
- 80% training data
- 20% testing data
- Maintains temporal order (no shuffling)

### 4. Model Pipeline
- **Standardization**: StandardScaler for feature normalization
- **Regression Model**: Ridge, Random Forest, or Gradient Boosting

### 5. Hyperparameter Tuning
- GridSearchCV with 5-fold cross-validation
- Optimizes for negative mean absolute error
- Tests alpha values: [0.001, 0.01, 0.1, 1, 10, 100]

### 6. Model Evaluation
**Metrics:**
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors
- **R² Score**: Proportion of variance explained

**Visualizations:**
- Actual vs Predicted time-series plot
- Residuals plot
- Scatter plot with ideal prediction line

### 7. Model Comparison
Compares three algorithms:
- Ridge Regression (with optimal alpha)
- Random Forest Regressor
- Gradient Boosting Regressor

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Sumit-Agnihotri/Electricity_Consumption_Forecasting_Project.git
cd Electricity_Consumption_Forecasting_Project
```

2. **Install required packages:**
```bash
pip install pandas numpy matplotlib scikit-learn streamlit joblib
```

Or create a `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Jupyter Notebook

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Electricity Consumption Forecasting.ipynb`

3. Run all cells to:
   - Load and preprocess data
   - Engineer features
   - Train and tune models
   - Evaluate performance
   - Save the best model

### Running the Streamlit Web App

1. Ensure the model file exists:
```bash
# The notebook saves the model as electricity_forecaster.pkl
```

2. Launch the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

4. Input parameters:
   - Select date and time
   - Enter last 3 hours of energy consumption
   - Click "Predict Next-Hour Usage"

## 📈 Model Performance

### Ridge Regression (Tuned)
- **Best Alpha**: 0.001
- **MAE**: ~X.XX kWh
- **RMSE**: ~X.XX kWh
- **R² Score**: ~0.XX

### Model Comparison Results
| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Ridge Regression | X.XX | X.XX | 0.XX |
| Random Forest | X.XX | X.XX | 0.XX |
| Gradient Boosting | X.XX | X.XX | 0.XX |

*Note: Run the notebook to see actual performance metrics*

## 📁 Project Structure

```
Electricity_Consumption_Forecasting_Project/
│
├── electricity.csv                          # Dataset
├── Electricity Consumption Forecasting.ipynb # Main analysis notebook
├── app.py                                   # Streamlit web application
├── electricity_forecaster.pkl               # Saved trained model
├── README.md                                # Project documentation
└── LICENSE                                  # License file
```

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning models and preprocessing
  - StandardScaler
  - Ridge Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - GridSearchCV
- **Streamlit**: Web application framework
- **Joblib**: Model serialization

## 🔮 Future Improvements

### Model Enhancements
- [ ] Implement LSTM/GRU for deep learning approach
- [ ] Add more temporal features (season, holidays, weekends)
- [ ] Include external features (weather, temperature)
- [ ] Experiment with XGBoost and LightGBM
- [ ] Implement time-series cross-validation

### Application Features
- [ ] Add confidence intervals for predictions
- [ ] Display historical trends visualization
- [ ] Enable batch predictions from CSV upload
- [ ] Add model retraining capability
- [ ] Include anomaly detection
- [ ] Deploy to cloud platform (Heroku, AWS, Azure)

### Code Quality
- [ ] Add unit tests
- [ ] Implement logging
- [ ] Create configuration file for parameters
- [ ] Add data validation checks
- [ ] Improve error handling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sumit Agnihotri**
- GitHub: [@Sumit-Agnihotri](https://github.com/Sumit-Agnihotri)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Sumit-Agnihotri/Electricity_Consumption_Forecasting_Project/issues).

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

---

**Happy Forecasting! ⚡📊**