
# 💳 Credit Card Fraud Detection & Customer Segmentation

This project demonstrates a machine learning approach to detecting fraudulent transactions and analyzing customer behavior using credit card data. It utilizes classification and clustering techniques to provide a dual-model solution: one for **fraud detection** and another for **customer behavior analysis**.

## 📂 Project Overview

- **Fraud Detection**: Uses supervised learning to identify whether a transaction is fraudulent, including models such as **Random Forest** and **Isolation Forest** for anomaly detection.
- **Customer Behavior Analysis**: Applies clustering to group customers based on transaction patterns using **KMeans**.
- **Data Preprocessing**: Includes encoding, feature scaling, and handling class imbalance using **SMOTE**.
- **Visualization**: Correlation heatmaps and confusion matrices for better understanding of features and model performance.

---

## 📊 Dataset

The project uses a dataset named `Credit_data.csv`, which includes anonymized customer transaction records. Ensure this dataset is placed in the root directory before running the notebook.

---

## 🔧 Technologies Used

- **Python**
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn** – Data visualization
- **Scikit-learn** – Machine learning modeling
- **Imbalanced-learn (SMOTE)** – Handling imbalanced classes
- **Isolation Forest** – Anomaly detection for fraud detection
- **KMeans** – Clustering customers based on transaction data

---

## 🧪 How it Works

### 1. **Load and Explore Data**
```python
df = pd.read_csv("Credit_data.csv")
```

### 2. **Preprocessing**
- One-hot encoding of categorical variables.
- Scaling of numerical features.
- SMOTE to address class imbalance.

### 3. **Model Building**
- **Random Forest Classifier** and **Isolation Forest** are used for fraud detection.
- **KMeans** is used for customer clustering.
- Model evaluation is done using confusion matrix and classification report.

---

## 📈 Output Examples

- **Confusion Matrix**
- **Classification Report**
- **Customer Segmentation Visualization**
- **Correlation Heatmaps**

---

## 🚀 Getting Started

### Prerequisites
Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Run the notebook
Open `fraud_detection.ipynb` in Jupyter Notebook or VSCode and run all cells in order.

---

## 🧠 Future Enhancements

- Integrate with a **Streamlit dashboard** for interactive demos.
- Use **XGBoost or LightGBM** for enhanced model accuracy.
- Real-time fraud detection using streaming data.

---

## 📌 Project Structure

```
.
├── Credit_data.csv
├── fraud_detection.ipynb
└── README.md
```

---

## ✍️ Author

**Sakibul Hassan**  
Aspiring Data Scientist | [GitHub: Sakib07](https://github.com/Sakib07)  
Email: sakibulhassan07@gmail.com

---

## 📄 License

This project is licensed under the MIT License.

```

---

This updated `README.md` reflects the changes in your code (addition of the Isolation Forest) and the removal of PCA. You can copy and paste it into your `README.md` file.

Let me know if you'd like any further changes or additions!It appears that the notebook still contains primarily code without markdown explanations. I'll review the updated code, including the addition of the Isolation Forest model, and adjust the `README.md` file accordingly. Let me generate the new file based on the code changes.

---

```markdown
# 💳 Credit Card Fraud Detection & Customer Segmentation

This project demonstrates a machine learning approach to detecting fraudulent transactions and analyzing customer behavior using credit card data. It utilizes classification and clustering techniques to provide a dual-model solution: one for **fraud detection** and another for **customer behavior analysis**.

## 📂 Project Overview

- **Fraud Detection**: Uses supervised learning to identify whether a transaction is fraudulent, including models such as **Random Forest** and **Isolation Forest** for anomaly detection.
- **Customer Behavior Analysis**: Applies clustering to group customers based on transaction patterns using **KMeans**.
- **Data Preprocessing**: Includes encoding, feature scaling, and handling class imbalance using **SMOTE**.
- **Visualization**: Correlation heatmaps and confusion matrices for better understanding of features and model performance.

---

## 📊 Dataset

The project uses a dataset named `Credit_data.csv`, which includes anonymized customer transaction records. Ensure this dataset is placed in the root directory before running the notebook.

---

## 🔧 Technologies Used

- **Python**
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn** – Data visualization
- **Scikit-learn** – Machine learning modeling
- **Imbalanced-learn (SMOTE)** – Handling imbalanced classes
- **Isolation Forest** – Anomaly detection for fraud detection
- **KMeans** – Clustering customers based on transaction data

---

## 🧪 How it Works

### 1. **Load and Explore Data**
```python
df = pd.read_csv("Credit_data.csv")
```

### 2. **Preprocessing**
- One-hot encoding of categorical variables.
- Scaling of numerical features.
- SMOTE to address class imbalance.

### 3. **Model Building**
- **Random Forest Classifier** and **Isolation Forest** are used for fraud detection.
- **KMeans** is used for customer clustering.
- Model evaluation is done using confusion matrix and classification report.

---

## 📈 Output Examples

- **Confusion Matrix**
- **Classification Report**
- **Customer Segmentation Visualization**
- **Correlation Heatmaps**

---

## 🚀 Getting Started

### Prerequisites
Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Run the notebook
Open `fraud_detection.ipynb` in Jupyter Notebook or VSCode and run all cells in order.

---

## 🧠 Future Enhancements

- Integrate with a **Streamlit dashboard** for interactive demos.
- Use **XGBoost or LightGBM** for enhanced model accuracy.
- Real-time fraud detection using streaming data.

---

## 📌 Project Structure

```
.
├── Credit_data.csv
├── fraud_detection.ipynb
└── README.md
```

---

## ✍️ Author

**Sakibul Hassan**  
Aspiring Data Scientist | [GitHub: Sakib07](https://github.com/Sakib07)  
Email: sakibulhassan07@gmail.com

---

## 📄 License

This project is licensed under the MIT License.

```



