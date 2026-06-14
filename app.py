import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json

# ── Загрузка модели ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_names.json") as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, feature_names = load_model()

# Маппинг ocean_proximity → числовой код (LabelEncoder сортирует по алфавиту)
OCEAN_MAP = {
    "<1H OCEAN": 0,
    "INLAND":    1,
    "ISLAND":    2,
    "NEAR BAY":  3,
    "NEAR OCEAN": 4,
}

# ── Интерфейс ────────────────────────────────────────────────────
st.set_page_config(page_title="Цены на жильё в Калифорнии", page_icon="🏠", layout="centered")

st.title("🏠 Прогноз стоимости жилья в Калифорнии")
st.markdown(
    "Введите характеристики района — модель **Gradient Boosting** предскажет "
    "медианную стоимость дома."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Расположение")
    longitude = st.slider("Долгота", -124.35, -114.31, -119.0, step=0.01)
    latitude  = st.slider("Широта",   32.54,   41.95,   35.5,  step=0.01)
    ocean_prox = st.selectbox("Близость к океану", list(OCEAN_MAP.keys()))

    st.subheader("🏘️ Район")
    housing_median_age = st.slider("Медианный возраст домов (лет)", 1, 52, 20)
    median_income      = st.slider("Медианный доход (×$10 000)", 0.5, 15.0, 3.5, step=0.1)

with col2:
    st.subheader("👥 Население и жильё")
    households   = st.number_input("Домохозяйств в районе",    min_value=1,  max_value=6000, value=500)
    population   = st.number_input("Численность населения",    min_value=1,  max_value=35000, value=1200)
    total_rooms  = st.number_input("Всего комнат в районе",    min_value=1,  max_value=40000, value=2500)
    total_bedrooms = st.number_input("Всего спален в районе",  min_value=1,  max_value=7000,  value=500)

st.divider()

# ── Вычисляем производные признаки ───────────────────────────────
rooms_per_household      = total_rooms / max(households, 1)
bedrooms_per_room        = total_bedrooms / max(total_rooms, 1)
population_per_household = population / max(households, 1)
ocean_enc                = OCEAN_MAP[ocean_prox]

input_data = pd.DataFrame([[
    longitude, latitude, housing_median_age,
    total_rooms, total_bedrooms, population, households,
    median_income,
    rooms_per_household, bedrooms_per_room, population_per_household,
    ocean_enc
]], columns=feature_names)

# ── Предсказание ─────────────────────────────────────────────────
if st.button("🔮 Рассчитать стоимость", use_container_width=True, type="primary"):
    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)

    st.success(f"### Прогнозируемая стоимость: **${prediction:,.0f}**")

    st.markdown("#### Введённые параметры района:")
    summary = {
        "Долгота / Широта": f"{longitude} / {latitude}",
        "Близость к океану": ocean_prox,
        "Возраст домов": f"{housing_median_age} лет",
        "Медианный доход": f"${median_income * 10_000:,.0f}",
        "Домохозяйств": households,
        "Население": population,
        "Комнат / домохозяйство": f"{rooms_per_household:.1f}",
        "Спален / комнату": f"{bedrooms_per_room:.2f}",
        "Чел. / домохозяйство": f"{population_per_household:.1f}",
    }
    st.table(pd.DataFrame(summary.items(), columns=["Параметр", "Значение"]))

st.divider()
st.caption("Модель обучена на California Housing Dataset (перепись 1990 г.). Алгоритм: Gradient Boosting Regressor.")
