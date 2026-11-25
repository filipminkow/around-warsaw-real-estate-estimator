import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Wczytujemy gotowy potok z dysku
pipeline = joblib.load('pipeline_mieszkania.pkl')

preprocessor = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['model']

# 2. Konfiguracja nagłówka strony
st.title("Warsaw Real Estate Estimator")
st.write("Wprowadź parametry mieszkania wokół Warszawy, aby oszacować jego cenę.")

# 3. Panel boczny lub główny z suwakami dla użytkownika
miasto = st.selectbox("Miejscowość:", ['Blonie', 'Grodzisk Mazowiecki', 'Minsk Mazowiecki', 'Piaseczno', 'Pruszkow'])
metraz = st.slider("Metraż (m²):", min_value=35.0, max_value=100.0, value=35.0, step=0.5)
pokoje = st.slider("Liczba pokoi:", min_value=1, max_value=4, value=1, step=1)
wiek_budynku = st.slider("Wiek budynku (lata):", min_value=0, max_value=50, value=0, step=1)
odleglosc_pkp = st.slider("Odległość od stacji PKP (km):", min_value=0.1, max_value=5.0, value=0.1, step=0.1)


if st.button("Oblicz szacowaną cenę"):
    dane_wejsciowe = pd.DataFrame([{'metraz': metraz,'pokoje': pokoje,'wiek_budynku': wiek_budynku,'odleglosc_pkp_km': odleglosc_pkp,'miasto': miasto,}])
    cena = pipeline.predict(dane_wejsciowe)[0]
    col1, col2 = st.columns(2)
    cena_za_m2 = cena/metraz
    col1.metric("Szacowana cena całkowita", f"{cena:,.0f} zł")
    col2.metric("Cena za 1 m²", f"{cena_za_m2:,.0f} zł/m²")

    dane_przetworzone = preprocessor.transform(dane_wejsciowe)
    nazwy_cech = preprocessor.get_feature_names_out()

    st.subheader("Dlaczego model wycenił mieszkanie akurat na tyle?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(dane_przetworzone)

    # Przypisujemy czytelne nazwy kolumn, żeby na wykresie nie było "Feature 0, Feature 1"
    shap_values.feature_names = nazwy_cech

    # Rysujemy wykres w Matplotlib i wrzucamy na stronę Streamlit
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)