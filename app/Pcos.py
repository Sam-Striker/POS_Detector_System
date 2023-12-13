import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Step 1: Data Cleaning
data = pd.read_csv('PCOS_data.csv')

# Drop irrelevant columns
data.drop(['Sl. No', 'Patient File No.'], axis=1, inplace=True)

# Handle missing values (example: remove rows with missing values)
data.dropna(inplace=True)

# Step 2: Descriptive Analysis
# Compute basic statistics
statistics = data.describe()

# Calculate frequency of unique values for categorical features
categorical_features = ['PCOS (Y/N)', 'Blood Group', 'Cycle(R/I)', 'Marraige Status (Yrs)', 'Pregnant(Y/N)',
                        'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)',
                        'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)']

frequency_counts = data[categorical_features].apply(pd.Series.value_counts)

# Step 3: Classification Algorithm Comparison
# Preprocess the data
X = data.drop('PCOS (Y/N)', axis=1)
y = data['PCOS (Y/N)']

label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train and evaluate classification algorithms
classifiers = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC()
]

best_classifier = None
best_accuracy = 0

for classifier in classifiers:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = classifier

# Step 4: Model Development
# Train the best classifier on the entire dataset
X_scaled = scaler.transform(X_encoded)  # Scale the entire dataset

best_classifier.fit(X_scaled, y)

# Save the trained model as a joblib file
joblib.dump(best_classifier, 'pcos_model.joblib')

# Example prediction on new data
new_data = pd.DataFrame({
    'Age (yrs)': [30],
    'Weight (Kg)': [60],
    'Height(Cm)': [160],
    'BMI': [23],
    'Blood Group': ['B+'],
    'Pulse rate(bpm)': [70],
    'RR (breaths/min)': [18],
    'Hb(g/dl)': [12],
    'Cycle(R/I)': ['R'],
    'Cycle length(days)': [28],
    'Marraige Status (Yrs)': [5],
    'Pregnant(Y/N)': ['N'],
    'No. of abortions': [0],
    'I   beta-HCG(mIU/mL)': [1.99],
    'II    beta-HCG(mIU/mL)': [1.99],
    'FSH(mIU/mL)': [5],
    'LH(mIU/mL)': [3],
    'FSH/LH': [1.66],
    'Hip(inch)': [36],
    'Waist(inch)': [30],
    'Waist:Hip Ratio': [0.83],
    'TSH (mIU/L)': [2],
    'AMH(ng/mL)': [3.5]
})

new_data_encoded = new_data.apply(label_encoder.transform)
new_data_scaled = scaler.transform(new_data_encoded)

# Load the trained model from the joblib file
loaded_model = joblib.load('PCOS_Pred.joblib')

prediction = loaded_model.predict(new_data_scaled)
print("Prediction:", prediction)
