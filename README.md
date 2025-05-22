# 📊 Telco Customer Churn – Streamlit App

Interaktywna aplikacja predykcji odejścia klienta oparta o dane z telekomunikacji. Stworzona w Pythonie z użyciem Streamlit.

## 🔍 Funkcje:
- Przewidywanie churnu na podstawie:
  - liczby miesięcy z firmą (tenure)
  - miesięcznych opłat
  - łącznej zapłaconej kwoty
- Wbudowane wizualizacje EDA
- Prosty interfejs użytkownika

## 🚀 Uruchomienie lokalne

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## 📂 Struktura katalogów

```
├─ app/
│  └─ app.py
├─ data/
│  └─ telco.csv
├─ model/
│  └─ model.pkl
├─ scripts/
│  └─ train_model.py
├─ requirements.txt
```

## 🌐 Online Demo

Jeśli wdrożono: [Otwórz aplikację na Streamlit Cloud](https://share.streamlit.io/your-username/ml-telco-churn-streamlit/main/app/app.py)

---

© 2025 Tymon
