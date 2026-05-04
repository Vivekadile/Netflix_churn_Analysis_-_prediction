# 📊 Customer Churn Prediction (Deep Learning + Streamlit)

## 🚀 Project Overview

This project predicts whether a customer is likely to churn based on their behavior and subscription details.
A deep learning model (ANN) is trained on user activity data and deployed using Streamlit for real-time predictions.

---

## 🧠 Problem Statement

Customer churn is a major issue for subscription-based platforms.
The goal is to identify customers at risk of leaving so that businesses can take preventive action.

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* Streamlit

---

## 📂 Project Structure

```
customer_churn/
│── data/                  # dataset
│
│── app.py                # Streamlit app
│── model.h5              # trained model
│── scaler.pkl           # saved scaler
│── requirements.txt
│── README.md
```

---

## 🔄 Workflow

1. **Data Preprocessing**

   * Removed unnecessary columns (e.g., `customer_id`)
   * Encoded categorical variables using `pd.get_dummies`
   * Scaled features using `StandardScaler`

2. **Model Building**

   * Artificial Neural Network (ANN)
   * Dense layers with Dropout for regularization
   * Binary classification (churn / no churn)

3. **Training**

   * Optimizer: Adam
   * Loss: Binary Crossentropy
   * Achieved ~89% accuracy on test data

4. **Evaluation**

   * Confusion Matrix
   * Precision, Recall, F1-score
   * Balanced performance across both classes

5. **Deployment**

   * Built a Streamlit UI for user input
   * Applied same preprocessing pipeline
   * Real-time churn prediction

---

## 📈 Model Performance

* Accuracy: ~89%
* Precision: 0.89
* Recall: 0.89
* F1 Score: 0.89

Confusion Matrix:

```
[[445  53]
 [ 53 449]]
```

---

## 🧪 How to Run

### 1. Clone repository

```
git clone <your-repo-link>
cd customer_churn
```

### 2. Create virtual environment

```
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run Streamlit app

```
streamlit run app.py
```

---

## 📊 Features Used

* Age
* Watch Hours
* Last Login Days
* Subscription Type
* Device Type
* Payment Method
* Number of Profiles
* Average Watch Time
* Favorite Genre
* Region

---

## ⚠️ Key Learnings

* Proper preprocessing is critical (encoding + scaling)
* Neural networks can become overconfident without tuning
* Feature interactions matter more than individual features
* Deployment requires consistent preprocessing pipeline

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Model calibration for better probability outputs
* Try tree-based models (Random Forest, XGBoost)
* Add more user-friendly UI

---

## 👨‍💻 Author

Vivek Kumar Adile
Engineering Student | AI/ML Enthusiast
