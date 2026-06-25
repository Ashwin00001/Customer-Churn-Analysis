Telecom Customer Churn Prediction using Artificial Neural Networks

## Overview

Customer churn is one of the biggest challenges faced by telecom companies, as retaining existing customers is often more cost-effective than acquiring new ones. This project uses an Artificial Neural Network (ANN) to predict whether a customer is likely to leave the service based on their demographic information, subscription details, and usage patterns.

The model is trained using TensorFlow/Keras and deployed as an interactive web application using Streamlit, allowing users to enter customer information and receive instant churn predictions.

---

## Features

- Predicts customer churn using an Artificial Neural Network (ANN)
- Interactive web interface built with Streamlit
- Data preprocessing with feature scaling and one-hot encoding
- Real-time prediction with churn probability
- Clean and user-friendly interface

---

## Tech Stack

**Programming Language**
- Python

**Libraries & Frameworks**
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Pickle

---

## Project Structure

```
Telecom-Churn-ANN/
│── app.py
│── train_model.py
│── telecom_data.csv
│── churn_ann.h5
│── scaler.pkl
│── requirements.txt
└── README.md
```

---

## How It Works

The project follows a standard machine learning pipeline:

1. Load the telecom customer dataset.
2. Clean and preprocess the data.
3. Encode categorical variables using one-hot encoding.
4. Scale numerical features using StandardScaler.
5. Split the dataset into training and testing sets.
6. Train an Artificial Neural Network (ANN).
7. Save the trained model and scaler.
8. Deploy the model using a Streamlit web application.
9. Accept customer details as input and predict churn probability.

---

## Model Architecture

| Layer | Configuration |
|--------|---------------|
| Dense | 64 Neurons (ReLU) |
| Dropout | 0.3 |
| Dense | 32 Neurons (ReLU) |
| Output | 1 Neuron (Sigmoid) |

**Training Configuration**

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 15
- Batch Size: 32

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Telecom-Churn-ANN.git
```

Navigate to the project folder:

```bash
cd Telecom-Churn-ANN
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train the Model

```bash
python train_model.py
```

This generates:

- `churn_ann.h5`
- `scaler.pkl`

### Launch the Web Application

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

## Input Parameters

The application accepts customer information such as:

- Gender
- Senior Citizen
- Dependents
- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Phone Service
- Online Security
- Online Backup
- Tech Support
- Streaming TV
- Streaming Movies
- Multiple Lines
- Payment Method
- Paperless Billing

---

## Output

The model predicts whether a customer is likely to churn and displays:

- **Low Risk of Churn**
- **High Risk of Churn**

along with the prediction probability.

---

## Future Enhancements

- Improve model performance through hyperparameter tuning
- Implement Early Stopping to reduce overfitting
- Deploy the application on Streamlit Community Cloud
- Add visual dashboards for customer insights
- Store prediction history using a database

