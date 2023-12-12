import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 1. Load and clean data
data = pd.read_csv("PCOS_data.csv")

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Impute missing values (replace NaN with mean/median)
data.fillna(data.mean(), inplace=True)

# 2. Descriptive analysis
print("Descriptive statistics:")
print(data.describe())

# 3. Feature selection
features = ["Age (yrs)", "Weight(Kg)", "Height(Cm)", "BMI", "Pulse rate(bpm)", "Hb(g/dl)", "Cycle(R/I)",
            "Cycle lenght(days)", "No. of abortions", "I beta-HCG(mIU/mL)", "II beta-HCG(mIU/mL)", "FSH(mIU/mL)",
            "LH(mIU/mL)", "FCH/LH", "Hip(inch)", "Waist(inch)", "Waist:Hip Ration", "TSH(mIU/mL)",
            "AMH(ng/mL)", "PRL(ng/mL)", "Vit D3(ng/mL)", "PRG(ng/mL)", "rbs(mg/dl)"]

target = "Have PC0S"

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, data[target], test_size=0.2, random_state=0)

# 4. Model comparison
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {accuracy:.4f}")

# Choose best model based on accuracy score
best_model = models["Random Forest"]

# 5. Final model development
best_model.fit(X_train, y_train)
y_pred_final = best_model.predict(X_test)

accuracy_final = accuracy_score(y_test, y_pred_final)
confusion_matrix_final = confusion_matrix(y_test, y_pred_final)

print(f"Final accuracy: {accuracy_final:.4f}")
print(f"Confusion matrix:\n{confusion_matrix_final}")

# Save the model for future use
import joblib

joblib.dump(best_model, "PCOS_Predictor_SVM.joblib")

print("Model saved successfully!")
