import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytaj model i dane
model = joblib.load('model/model.pkl')
df = pd.read_csv('data/telco.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Tytuł
st.title('🔮 Przewidywanie odejścia klienta (Churn)')

with st.expander("ℹ️ Co oznaczają te dane?", expanded=False):
    st.markdown("""
    - **tenure**: liczba miesięcy z firmą
    - **MonthlyCharges**: miesięczny koszt usług
    - **TotalCharges**: suma opłat (obliczana automatycznie)
    - **PaymentMethod**: sposób płatności (informacyjnie)
    """)

# Dane wejściowe
tenure = st.selectbox("📅 Liczba miesięcy z firmą (tenure)", [1, 3, 6, 12, 24, 36, 48, 60, 72], index=2)
monthly_charges = st.selectbox("💳 Miesięczna opłata (MonthlyCharges)", [20, 40, 60, 70, 80, 100, 120], index=3)
payment_method = st.selectbox("💰 Metoda płatności (informacyjnie)", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
], index=0)

# Automatyczne obliczenie TotalCharges
total_charges = tenure * monthly_charges
st.info(f"📊 Obliczona łączna kwota zapłacona: **{total_charges:.2f}**")

# Predykcja
if st.button("🔍 Przewiduj"):
    input_df = pd.DataFrame([[tenure, monthly_charges, total_charges]],
                            columns=['tenure', 'MonthlyCharges', 'TotalCharges'])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.markdown("### 📈 Wynik predykcji:")
    if prediction == 1:
        st.error(f"❌ Klient prawdopodobnie ODEJDZIE. (Prawdopodobieństwo: {proba:.2%})")
    else:
        st.success(f"✅ Klient prawdopodobnie ZOSTANIE. (Prawdopodobieństwo: {1 - proba:.2%})")

    st.progress(int(proba * 100))

# EDA i wnioski
with st.expander("📊 Eksploracja danych – wykresy"):
    st.subheader("1. Churn vs MonthlyCharges")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax1)
    ax1.set_title("Monthly Charges vs Churn")
    st.pyplot(fig1)

    st.subheader("2. Rozkład długości umowy (tenure)")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['tenure'], bins=30, kde=True, ax=ax2)
    ax2.set_title("Rozkład długości umowy")
    st.pyplot(fig2)

    st.markdown("🔎 **Wniosek:** najwięcej klientów odchodzi w ciągu pierwszych 5 miesięcy.")

if st.checkbox("📋 Zobacz przykładowe dane klientów"):
    st.dataframe(df[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].sample(5))

with st.expander("📊 Dodatkowe wykresy"):
    st.subheader("3. Churn vs PaymentMethod")
    fig3, ax3 = plt.subplots()
    churn_by_payment = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index')
    churn_by_payment.plot(kind='bar', stacked=True, ax=ax3, colormap='viridis')
    ax3.set_ylabel("Proporcja")
    ax3.set_title("Churn wg metody płatności")
    ax3.legend(["Został", "Odszedł"])
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.subheader("4. Znaczenie cech (feature importance)")
    importances = model.feature_importances_
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig4, ax4 = plt.subplots()
    ax4.barh(features, importances, color='skyblue')
    ax4.set_title("Które cechy mają największy wpływ?")
    st.pyplot(fig4)

with st.expander("📌 Wnioski z analizy danych", expanded=False):
    st.markdown("""
- Klienci o krótkim stażu i wyższych kosztach są bardziej skłonni do odejścia.
- Sposób płatności ma znaczenie – elektroniczne przelewy wskazują na wyższy churn.
- Najwięcej klientów traci się w pierwszych 3–5 miesiącach współpracy.
    """)
