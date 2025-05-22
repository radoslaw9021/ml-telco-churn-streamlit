import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Wczytanie danych
df = pd.read_csv('../data/telco.csv')

# Konwersja TotalCharges na float i usunięcie braków
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Mapowanie zmiennej docelowej
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Wybór cech (prosty zestaw przykładowy)
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

# Podział na zbiory
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ewaluacja
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Zapis modelu
joblib.dump(model, '../model/model.pkl')
print("Model zapisany do ../model/model.pkl")
