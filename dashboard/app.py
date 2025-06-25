# dashboard/app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud

# Попытаемся импортировать seasonal_decompose
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False

# Определяем корень проекта
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Пути к данным
PATH_SALES_PARQUET   = os.path.join(BASE_DIR, "data", "processed", "processed.parquet")
PATH_SENTIMENT_CSV   = os.path.join(BASE_DIR, "data", "reviews", "sentiment.csv")
PATH_DL_COMPARISON   = os.path.join(BASE_DIR, "results_dl", "dl_comparison.csv")

# Настройка страницы
st.set_page_config(
    page_title="Walmart Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_sales():
    """Загрузка подготовленных parquet-данных продаж."""
    if not os.path.exists(PATH_SALES_PARQUET):
        st.error(f"Файл не найден: {PATH_SALES_PARQUET}")
        return pd.DataFrame()
    return pd.read_parquet(PATH_SALES_PARQUET)

@st.cache_data
def load_reviews():
    """Загрузка разметки отзывов из sentiment.csv."""
    if not os.path.exists(PATH_SENTIMENT_CSV):
        return pd.DataFrame(columns=["text", "label"])
    df = pd.read_csv(PATH_SENTIMENT_CSV)
    # ожидаем колонки 'text' и 'label'
    if "text" not in df.columns or "label" not in df.columns:
        return pd.DataFrame(columns=["text", "label"])
    return df[["text", "label"]]

@st.cache_data
def load_dl_metrics():
    """Загрузка результатов глубокого обучения."""
    if not os.path.exists(PATH_DL_COMPARISON):
        return pd.DataFrame()
    return pd.read_csv(PATH_DL_COMPARISON)

# --- Загрузка данных ---
df_sales = load_sales()
df_reviews = load_reviews()
df_dl    = load_dl_metrics()

# --- Sidebar: фильтры ---
st.sidebar.title("Фильтры продаж")
if not df_sales.empty:
    min_date, max_date = df_sales.Date.min().date(), df_sales.Date.max().date()
    date_range = st.sidebar.date_input(
        "Период", (min_date, max_date),
        min_value=min_date, max_value=max_date
    )
    if len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_sales = df_sales[df_sales.Date.between(start, end)]

    stores = ["Все"] + sorted(df_sales.Store.unique().tolist())
    sel_store = st.sidebar.selectbox("Магазин", stores)
    if sel_store != "Все":
        df_sales = df_sales[df_sales.Store == sel_store]

    depts = ["Все"] + sorted(df_sales.Dept.unique().tolist())
    sel_dept = st.sidebar.selectbox("Отдел", depts)
    if sel_dept != "Все":
        df_sales = df_sales[df_sales.Dept == sel_dept]
else:
    st.sidebar.warning("Нет данных продаж для фильтров")

# --- Раздел 1: Столбчатая диаграмма продаж ---
st.title("📊 Еженедельные продажи Walmart")
if "Weekly_Sales" in df_sales.columns:
    agg = df_sales.groupby(["Date", "IsHoliday"])["Weekly_Sales"].sum().reset_index()
    fig = px.bar(
        agg, x="Date", y="Weekly_Sales", color="IsHoliday",
        barmode="stack",
        labels={"Weekly_Sales": "Сумма продаж", "IsHoliday": "Праздник"},
        title="Продажи с разделением по праздничным неделям"
    )
    fig.update_layout(xaxis_title=None, yaxis_title="Сумма продаж")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Колонка `Weekly_Sales` отсутствует")

# --- Раздел 2: Сезонная декомпозиция ---
st.subheader("🔍 Сезонная декомпозиция суммарных продаж")
if _has_statsmodels and "Weekly_Sales" in df_sales.columns:
    daily = df_sales.groupby("Date")["Weekly_Sales"].sum().reset_index()
    if len(daily) >= 104:
        decomp = seasonal_decompose(daily.set_index("Date")["Weekly_Sales"], model="additive", period=52)
        comp = make_subplots(rows=4, cols=1, shared_xaxes=True,
                             subplot_titles=["Наблюдения", "Тренд", "Сезонность", "Остатки"])
        comp.add_trace(go.Scatter(x=daily.Date, y=daily.Weekly_Sales), row=1, col=1)
        comp.add_trace(go.Scatter(x=daily.Date, y=decomp.trend), row=2, col=1)
        comp.add_trace(go.Scatter(x=daily.Date, y=decomp.seasonal), row=3, col=1)
        comp.add_trace(go.Scatter(x=daily.Date, y=decomp.resid), row=4, col=1)
        comp.update_layout(height=800, showlegend=False)
        st.plotly_chart(comp, use_container_width=True)
    else:
        st.info("Недостаточно данных для декомпозиции")
else:
    st.info("Установите `statsmodels` для сезонной декомпозиции")

# --- Раздел 3: Корреляционная матрица ---
st.subheader("📈 Корреляционная матрица признаков")
features = ["Weekly_Sales", "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
available = [f for f in features if f in df_sales.columns]
sel = st.multiselect("Выберите признаки", available, default=available)
if sel:
    corr = df_sales[sel].corr()
    hm = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", origin="lower")
    hm.update_layout(height=400, margin=dict(t=40))
    st.plotly_chart(hm, use_container_width=True)
else:
    st.info("Выберите хотя бы один признак")

# --- Раздел 4: Scatter с выбором объёма и цели ---
st.subheader("🔀 Scatter: зависимости")
num_cols = df_sales.select_dtypes(include="number").columns.tolist()
if num_cols:
    target = st.selectbox("Ось Y (цель)", num_cols, index=num_cols.index("Weekly_Sales") if "Weekly_Sales" in num_cols else 0)
    x_feats = st.multiselect("Ось X", [c for c in num_cols if c!=target], default=[c for c in num_cols if c not in ("Store","Dept","IsHoliday","Date",target)][:3])
    n = len(df_sales)
    opts = [1000, 5000, 10000]
    opts = [o for o in opts if o<n] + [n]
    sample_n = st.select_slider("Точек", opts, value=min(5000,n))
    if x_feats:
        df_sub = df_sales.sample(sample_n, random_state=1)
        cols = 3
        for i in range(0, len(x_feats), cols):
            row = st.columns(min(cols, len(x_feats)-i))
            for j, feat in enumerate(x_feats[i:i+cols]):
                fig = px.scatter(df_sub, x=feat, y=target, color="IsHoliday" if "IsHoliday" in df_sub.columns else None,
                                 title=f"{feat} vs {target} (n={sample_n})")
                fig.update_traces(marker=dict(size=4, opacity=0.6))
                row[j].plotly_chart(fig, use_container_width=True)
    else:
        st.info("Выберите X-признаки")
else:
    st.info("Нет числовых признаков для scatter")

# --- Раздел 5: География ---
if {"Latitude","Longitude"}.issubset(df_sales.columns):
    st.subheader("🌎 География магазинов")
    loc = df_sales.groupby("Store")[["Latitude","Longitude","Weekly_Sales"]].sum().reset_index()
    mfig = px.scatter_mapbox(loc, lat="Latitude", lon="Longitude", size="Weekly_Sales",
                             hover_name="Store", mapbox_style="carto-positron", zoom=3)
    st.plotly_chart(mfig, use_container_width=True)

# --- Раздел 6: Анализ отзывов ---
st.title("💬 Анализ отзывов пользователей")
if not df_reviews.empty:
    wc = WordCloud(width=800, height=300, background_color="white").generate(" ".join(df_reviews.text))
    st.image(wc.to_array(), use_column_width=True)
    df_reviews["length"] = df_reviews.text.str.len()
    h = px.histogram(df_reviews, x="length", nbins=30, title="Длина отзывов")
    st.plotly_chart(h, use_container_width=True)
    # Pie chart sentiment
    cnt = df_reviews.label.value_counts().rename(index={0:"Negative",1:"Positive"})
    p = px.pie(names=cnt.index, values=cnt.values, title="Доли классов отзывов")
    st.plotly_chart(p, use_container_width=True)
else:
    st.info("Нет размеченных отзывов")

# --- Раздел 7: Результаты DL ---
st.subheader("🤖 Сравнение DL моделей")
if not df_dl.empty:
    st.dataframe(df_dl, use_container_width=True)
    acc = px.bar(df_dl, x="model", y="accuracy", title="Accuracy моделей")
    st.plotly_chart(acc, use_container_width=True)
    f1 = df_dl.melt(id_vars="model", value_vars=["f1_neg","f1_pos"], var_name="class", value_name="f1")
    f1_fig = px.bar(f1, x="model", y="f1", color="class", barmode="group", title="F1-score по классам")
    st.plotly_chart(f1_fig, use_container_width=True)
else:
    st.info("Нет результатов DL (results_dl/dl_comparison.csv)")
