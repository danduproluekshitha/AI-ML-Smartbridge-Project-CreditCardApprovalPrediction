# 💳 Automated Credit Card Approval System
### AI-Powered Credit Card Eligibility Prediction

The Automated Credit Card Approval System is a Machine Learning-powered web application that predicts whether a credit card application is likely to be approved based on an applicant's financial and personal information. By leveraging predictive analytics, the system helps streamline the approval process, reduce manual effort, and support faster, data-driven decision-making.

The application combines a Flask backend with an interactive web interface, allowing users to enter applicant details and receive instant approval predictions along with confidence scores and feature-based insights.

---

## 🚀 Features

- 🔐 User-Friendly Web Interface
- 🤖 Machine Learning-Based Credit Approval Prediction
- 📊 Instant Approval/Rejection Results
- 📈 Prediction Confidence Score
- 🧠 Feature Importance & SHAP Explanations
- ⚡ Fast Flask Backend
- 📋 Clean and Responsive Dashboard
- 💾 Pre-trained Machine Learning Model

---

## 🛠 Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap

### Backend
- Flask
- Python

### Machine Learning
- Scikit-learn
- Pandas
- NumPy
- SHAP

### Database
- CSV Dataset

### Development Tools
- Visual Studio Code
- Git
- GitHub
- Jupyter Notebook

---

## 📂 Project Structure

```text
Credit-Card-Approval-System/
│
├── app.py
├── model/
│   ├── best_model.pkl
│   ├── label_encoders.pkl
│   └── feature_columns.pkl
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── dataset/
│   └── clean_dataset.csv
├── notebooks/
│   └── model_training.ipynb
├── docs/
│   ├── SRS.pdf
│   ├── Project_Report.pdf
│   ├── Presentation.pdf
│   ├── Data_Flow_Diagram.pdf
│   └── Flowchart.pdf
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/danduproluekshitha/Credit-Card-Approval-System.git

cd Credit-Card-Approval-System
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the Application

```bash
python app.py
```

The application will start at:

```
http://127.0.0.1:5000
```

---

## 🤖 Machine Learning Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Label Encoding
5. Train-Test Split
6. Model Training
7. Hyperparameter Tuning
8. Model Evaluation
9. Credit Approval Prediction
10. SHAP-Based Model Explanation

---

## 📊 Machine Learning Algorithms

The following algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The best-performing model was selected based on evaluation metrics and deployed in the web application.

---

## 📁 Documentation

The `docs/` folder contains:

- Software Requirements Specification (SRS)
- Project Report
- Project Presentation
- Data Flow Diagram
- System Flowchart
- ER Diagram
- Screenshots
- Other Supporting Documents

---

## 📸 Screenshots

The project documentation includes screenshots of:

- Home Page
- Applicant Details Form
- Prediction Result
- SHAP Explanation
- Dashboard
- Model Performance

---

## 📈 Dataset Features

The model uses the following applicant information:

- Gender
- Age
- Marital Status
- Employment Status
- Years Employed
- Income
- Debt
- Credit Score
- Prior Default History
- Bank Customer Status
- Industry
- Citizenship

---

## 🔮 Future Enhancements

- User Authentication
- Credit Score Prediction
- Loan Eligibility Prediction
- Explainable AI Dashboard
- Cloud Deployment
- Docker Support
- PDF Report Generation
- REST API Integration
- Mobile Responsive UI

---

## Demo Link
https://drive.google.com/file/d/1O-DPSBp1f3qhfNnjOQd4xMP2Fg1itJ7K/view?usp=drive_link

## 👥 Team Members

This project was developed as an academic team project by:

- **Ekshitha Danduprolu**

---

## 📄 License

This project is developed for educational and learning purposes.

---

## ⭐ Support

If you found this project useful, consider giving it a **Star ⭐** on GitHub.
