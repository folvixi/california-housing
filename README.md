# 🏠 Прогнозирование цен на недвижимость в Калифорнии

Веб-приложение для предсказания медианной стоимости жилья по характеристикам района.

## Описание

- **Задача:** регрессия — предсказание `median_house_value`
- **Данные:** California Housing Dataset (перепись населения 1990 г., 20 640 записей)
- **Лучшая модель:** Gradient Boosting Regressor (Test R² ≈ 0.83, RMSE ≈ $47 000)

## Структура репозитория

```
├── app.py                            # Streamlit веб-приложение
├── california_housing_regression.ipynb  # Ноутбук: EDA, обучение, анализ моделей
├── california_housing.csv            # Датасет
├── model.pkl                         # Обученная модель
├── scaler.pkl                        # Стандартизатор признаков
├── feature_names.json                # Список признаков
└── requirements.txt                  # Зависимости
```

## Сравнение моделей

| Модель | Test RMSE | Test R2 | Переобучение |
|:---|---:|---:|:---|
| Linear Regression | 68 000 $ | 0.64 | Нет (недообучение) |
| Ridge Regression | 68 000 $ | 0.64 | Нет (недообучение) |
| Decision Tree | 71 000 $ | 0.61 | Сильное |
| Random Forest | 51 000 $ | 0.80 | Умеренное |
| **Gradient Boosting (tuned)** | **47 000 $** | **0.83** | **Минимальное** |

## Запуск локально

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Деплой

Приложение развёрнуто на [Streamlit Cloud](https://streamlit.io/cloud).
