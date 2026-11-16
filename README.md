# Statiscal_Anamoly_Detection_and_machine-_failure_prediction
This project focuses on identifying abnormal operating conditions in industrial machines and predicting potential machine failures using Machine Learning and Statistical Anomaly Detection techniques.

The main objective is to detect unusual patterns in machine operating parameters and identify possible machine failures before they occur.

The project uses the AI4I 2020 Predictive Maintenance Dataset to analyse machine performance, detect anomalies, and understand the factors contributing to machine failure.


### OBJECTIVES
* Perform Exploratory Data Analysis (EDA) to understand machine operating conditions.
* Identify abnormal observations using statistical techniques.
* Detect outliers using the IQR method and Z-score.
* Analyse relationships between machine parameters and failure conditions.
* Develop Machine Learning models for machine failure prediction.
* Build an interactive dashboard to visualise machine conditions and anomaly detection results.
* Support predictive maintenance by identifying potential machine risks.

### TECHNOLOGIES USED
* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Jupyter Notebook
  
### PROJECT STRUCTURE

Machine-Failure-Prediction/
│
├── Dataset/
│   └── ai4i2020.csv
│
├── ML_PROJECT_MFP/
│   ├── statiscal_anamoly_detection_and_machine_failure_prediction.ipynb
│
├── app.py
│
├── requirements.txt
│
└── README.md

### WORKFLOW

1. Data Collection
Collected the AI4I 2020 Predictive Maintenance dataset for analysis.

2. Data Cleaning and Preprocessing
   Checked missing values .Removed unnecessary columns where appropriate.Handled categorical variables.Checked duplicate records.Prepared data for analysis and modelling.

3. Exploratory Data Analysis (EDA)
Analysed machine operating parameters.Studied feature distributions.Identified relationships between variables.Visualised machine failure patterns.

4. Statistical Anomaly Detection
Applied statistical techniques to identify unusual observations.
IQR (Interquartile Range):
Used to identify observations outside the lower and upper statistical boundaries.

Z-score:
Used to measure how far an observation is from the mean in terms of standard deviations.
Created anomaly indicators:
IQR Flagged

Z-score Flagged
These indicators help identify potentially abnormal machine operating conditions.

5. Machine Learning
Machine Learning techniques are used to analyse machine operating parameters and predict potential machine failures.
The model performance is evaluated using appropriate classification metrics.

6. Dashboard Development
Developed an interactive dashboard to display:Machine operating parameters.Statistical anomaly detection results.
IQR and Z-score flagged observations.Machine failure information.Risk-related insights and visualisations.

### Limitations

* The project uses the publicly available AI4I 2020 dataset.

* The dataset is synthetic and does not represent direct measurements from a specific real-world industrial machine.

* Actual industrial deployment requires validation using real machine sensor data and manufacturer-defined operating limits.

* Anomaly detection does not necessarily mean that a machine will fail.
