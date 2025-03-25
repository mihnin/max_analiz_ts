import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Расширенный анализатор прогнозируемости временных рядов",
    page_icon="📊",
    layout="wide"
)

# Заголовки
st.title("Расширенный анализатор прогнозируемости временных рядов")
st.markdown("""
Это приложение анализирует ваш файл и определяет, какие столбцы подходят для прогнозирования, 
а какие нет. Просто загрузите CSV или Excel файл с данными временного ряда.

_Приложение проводит комплексный анализ, включая расчет метрик и визуализации, которые помогут 
оценить потенциал прогнозирования для каждого числового столбца._
""")

# Создаем боковую панель с параметрами
st.sidebar.header("Параметры анализа")

# Пороговые значения для определения прогнозируемости
cv_threshold = st.sidebar.slider(
    "Порог коэффициента вариации (CV)", 
    min_value=0.1, 
    max_value=5.0, 
    value=1.5, 
    step=0.1,
    help="CV > 1.5 указывает на плохую прогнозируемость"
)

zero_percentage_threshold = st.sidebar.slider(
    "Порог процента нулевых значений", 
    min_value=10, 
    max_value=90, 
    value=40, 
    step=5,
    help="Высокий процент нулей затрудняет прогнозирование"
)

autocorr_threshold = st.sidebar.slider(
    "Порог автокорреляции", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.3, 
    step=0.1,
    help="Низкая автокорреляция указывает на слабую временную зависимость"
)

stationarity_p_value = st.sidebar.slider(
    "P-значение для теста на стационарность", 
    min_value=0.01, 
    max_value=0.1, 
    value=0.05, 
    step=0.01,
    help="Низкое p-значение указывает на стационарность (легче прогнозировать)"
)

# Метод деления для оценки прогнозируемости
test_size = st.sidebar.slider(
    "Размер тестовой выборки", 
    min_value=0.1, 
    max_value=0.5, 
    value=0.2, 
    step=0.05,
    help="Доля данных для оценки точности простых моделей"
)

# Выбор декомпозиции
decomposition_model = st.sidebar.selectbox(
    "Метод декомпозиции временного ряда",
    ["additive", "multiplicative"],
    index=0,
    help="Additive для рядов с постоянной амплитудой сезонности, multiplicative для рядов с изменяющейся амплитудой"
)

# Функции для анализа прогнозируемости

def calculate_cv(series):
    """Расчет коэффициента вариации (CV)"""
    series_no_null = series.dropna()
    if len(series_no_null) == 0:
        return float('inf')
    
    mean = np.mean(series_no_null)
    if mean == 0 or np.isclose(mean, 0):
        return float('inf')
    
    return np.std(series_no_null) / abs(mean)

def calculate_entropy(series):
    """Расчет энтропии как меры хаотичности"""
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    
    # Дискретизация данных для расчета энтропии
    hist, bin_edges = np.histogram(series, bins=min(20, len(np.unique(series))), density=True)
    
    # Расчет энтропии по Шеннону
    entropy = 0
    for p in hist:
        if p > 0:
            entropy -= p * np.log2(p)
    
    # Нормализация энтропии (0 - низкая хаотичность, 1 - высокая)
    max_entropy = np.log2(len(hist))
    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = np.nan
    
    return normalized_entropy

def estimate_lyapunov(series, lag=1, min_points=10):
    """Приближенная оценка метрики Ляпунова"""
    series = series.dropna()
    if len(series) < min_points:
        return np.nan
    
    try:
        # Создание временного ряда и его сдвинутой версии
        y = series.values[:-lag]
        y_lagged = series.values[lag:]
        
        # Расчет расстояний между соседними точками
        divergence = np.abs(np.diff(y))
        divergence_lagged = np.abs(np.diff(y_lagged))
        
        # Расчет среднего логарифма отношений
        valid_indices = (divergence != 0) & np.isfinite(divergence) & np.isfinite(divergence_lagged)
        if np.sum(valid_indices) < min_points // 2:
            return np.nan
        
        log_ratios = np.log(divergence_lagged[valid_indices] / divergence[valid_indices])
        lyapunov_exp = np.mean(log_ratios)
        
        return lyapunov_exp
    except:
        return np.nan

def calculate_autocorrelation(series, lag=1):
    """Расчет автокорреляции временного ряда"""
    series = series.dropna()
    if len(series) <= lag:
        return np.nan
    try:
        autocorrelation = acf(series, nlags=lag)[lag]
        return autocorrelation
    except:
        return np.nan

def calculate_zero_percentage(series):
    """Расчет процента нулевых значений"""
    series = series.dropna()
    if len(series) == 0:
        return 100.0
    return (series == 0).sum() / len(series) * 100

def calculate_sign_changes(series):
    """Расчет количества изменений знака"""
    # Удаляем нулевые значения для корректного определения знака
    series = series.dropna()
    non_zero_series = series[series != 0]
    if len(non_zero_series) <= 1:
        return 0, 0
    
    signs = np.sign(non_zero_series)
    sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
    return sign_changes, sign_changes / (len(non_zero_series) - 1) * 100

def test_stationarity(series):
    """Тест Дики-Фуллера на стационарность"""
    series = series.dropna()
    if len(series) < 8:  # Минимальное количество точек для теста
        return None, np.nan
    
    try:
        result = adfuller(series)
        return result, result[1]  # возвращаем полный результат и p-значение
    except:
        return None, np.nan

def detect_date_column(df):
    """Определение столбца с датой"""
    potential_date_cols = []
    
    # Проверяем столбцы с объектным типом или строками
    for col in df.select_dtypes(include=['object', 'string']).columns:
        try:
            # Пробуем преобразовать к дате первые несколько элементов
            sample = df[col].dropna().head(5)
            success = 0
            for item in sample:
                try:
                    pd.to_datetime(item)
                    success += 1
                except:
                    pass
            
            # Если более 60% элементов успешно преобразованы, считаем столбец датой
            if success / len(sample) > 0.6:
                potential_date_cols.append(col)
        except:
            pass
    
    # Проверяем столбцы с типом datetime
    date_cols = list(df.select_dtypes(include=['datetime64']).columns)
    potential_date_cols.extend(date_cols)
    
    # Если есть потенциальные столбцы с датой, возвращаем первый
    if potential_date_cols:
        return potential_date_cols[0]
    
    return None

def evaluate_simple_models(df, date_col, value_col, test_size=0.2):
    """Оценка точности простых моделей прогнозирования"""
    try:
        # Подготавливаем данные
        df_sorted = df.sort_values(by=date_col).copy()
        df_sorted['numeric_date'] = pd.to_numeric(df_sorted[date_col])
        
        # Удаляем строки с пропущенными значениями
        df_clean = df_sorted.dropna(subset=[value_col, 'numeric_date'])
        
        # Проверяем, достаточно ли данных
        if len(df_clean) < 10:
            return {"error": "Недостаточно данных для оценки моделей"}
        
        # Создаем признаки: предыдущие значения
        df_clean['prev_value'] = df_clean[value_col].shift(1)
        df_clean = df_clean.dropna()
        
        # Разделяем данные на обучающую и тестовую выборки
        train_size = int(len(df_clean) * (1 - test_size))
        train_df = df_clean.iloc[:train_size]
        test_df = df_clean.iloc[train_size:]
        
        if len(train_df) < 5 or len(test_df) < 3:
            return {"error": "Недостаточно данных для обучения и тестирования моделей"}
        
        # 1. Наивная модель (предыдущее значение)
        y_naive = test_df['prev_value'].values
        y_true = test_df[value_col].values
        
        naive_mae = mean_absolute_error(y_true, y_naive)
        naive_mse = mean_squared_error(y_true, y_naive)
        naive_rmse = np.sqrt(naive_mse)
        naive_mape = np.mean(np.abs((y_true - y_naive) / np.where(y_true != 0, y_true, 1))) * 100
        
        # 2. Линейная регрессия
        X_train = train_df[['numeric_date', 'prev_value']]
        y_train = train_df[value_col]
        X_test = test_df[['numeric_date', 'prev_value']]
        
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        y_lr = lr_model.predict(X_test)
        
        lr_mae = mean_absolute_error(y_true, y_lr)
        lr_mse = mean_squared_error(y_true, y_lr)
        lr_rmse = np.sqrt(lr_mse)
        lr_mape = np.mean(np.abs((y_true - y_lr) / np.where(y_true != 0, y_true, 1))) * 100
        
        # 3. Random Forest (для нелинейных зависимостей)
        try:
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_model.fit(X_train, y_train)
            
            y_rf = rf_model.predict(X_test)
            
            rf_mae = mean_absolute_error(y_true, y_rf)
            rf_mse = mean_squared_error(y_true, y_rf)
            rf_rmse = np.sqrt(rf_mse)
            rf_mape = np.mean(np.abs((y_true - y_rf) / np.where(y_true != 0, y_true, 1))) * 100
        except:
            rf_mae, rf_rmse, rf_mape = np.nan, np.nan, np.nan
        
        # Сравнение моделей
        best_model_name = "Наивная модель"
        best_mae = naive_mae
        
        if lr_mae < best_mae:
            best_model_name = "Линейная регрессия"
            best_mae = lr_mae
            
        if not np.isnan(rf_mae) and rf_mae < best_mae:
            best_model_name = "Random Forest"
            best_mae = rf_mae
        
        # Оценка прогнозируемости на основе точности моделей
        mape_benchmark = 25  # MAPE ниже 25% считается хорошим результатом
        best_mape = min(naive_mape, lr_mape, np.inf if np.isnan(rf_mape) else rf_mape)
        
        forecasting_quality = "Высокая" if best_mape < mape_benchmark else "Средняя" if best_mape < 2 * mape_benchmark else "Низкая"
        
        return {
            "naive_mae": naive_mae,
            "naive_rmse": naive_rmse,
            "naive_mape": naive_mape,
            "lr_mae": lr_mae,
            "lr_rmse": lr_rmse,
            "lr_mape": lr_mape,
            "rf_mae": rf_mae if not np.isnan(rf_mae) else None,
            "rf_rmse": rf_rmse if not np.isnan(rf_rmse) else None,
            "rf_mape": rf_mape if not np.isnan(rf_mape) else None,
            "best_model": best_model_name,
            "best_mape": best_mape,
            "forecasting_quality": forecasting_quality
        }
    except Exception as e:
        return {"error": f"Ошибка при оценке моделей: {str(e)}"}

def analyze_forecasting_potential(series, stats_results, model_evaluation=None):
    """Анализ потенциала прогнозирования временного ряда с учетом всех метрик"""
    if len(series.dropna()) < 4:
        return {
            'cv': np.nan,
            'autocorr': np.nan,
            'zero_percentage': np.nan,
            'sign_changes': np.nan,
            'sign_changes_percentage': np.nan,
            'entropy': np.nan,
            'lyapunov': np.nan,
            'stationarity_p_value': np.nan,
            'conclusion': 'Недостаточно данных для анализа',
            'reasons': ['Временной ряд содержит менее 4 точек'],
            'score': 0,
            'model_evaluation': model_evaluation
        }
    
    # Извлекаем метрики
    cv = stats_results['cv']
    autocorr = stats_results['autocorr']
    zero_percentage = stats_results['zero_percentage']
    sign_changes = stats_results['sign_changes']
    sign_changes_percentage = stats_results['sign_changes_percentage']
    entropy = stats_results['entropy']
    lyapunov = stats_results['lyapunov']
    stationarity_p_value = stats_results['stationarity_p_value']
    
    # Оценка прогнозируемости
    reasons = []
    
    # Базовая оценка на основе статистических метрик
    statistical_score = 0
    
    # Оценка по коэффициенту вариации
    if np.isnan(cv) or cv > cv_threshold * 2:
        statistical_score += 0
        cv_str = f"{cv:.2f}" if not np.isnan(cv) else "∞"
        reasons.append(f'Очень высокий коэффициент вариации (CV = {cv_str} > {cv_threshold * 2})')
    elif cv > cv_threshold:
        statistical_score += 10
        reasons.append(f'Высокий коэффициент вариации (CV = {cv:.2f} > {cv_threshold})')
    else:
        statistical_score += 25
    
    # Оценка по автокорреляции
    if np.isnan(autocorr) or abs(autocorr) < autocorr_threshold / 2:
        statistical_score += 0
        autocorr_str = f"{abs(autocorr):.2f}" if not np.isnan(autocorr) else "?"
        reasons.append(f'Крайне слабая автокорреляция (|r| = {autocorr_str} < {autocorr_threshold / 2})')
    elif abs(autocorr) < autocorr_threshold:
        statistical_score += 10
        reasons.append(f'Слабая автокорреляция (|r| = {abs(autocorr):.2f} < {autocorr_threshold})')
    else:
        statistical_score += 25
    
    # Оценка по проценту нулевых значений
    if np.isnan(zero_percentage) or zero_percentage > zero_percentage_threshold * 1.5:
        statistical_score += 0
        zero_percentage_str = f"{zero_percentage:.1f}" if not np.isnan(zero_percentage) else "?"
        reasons.append(f'Крайне высокий процент нулевых значений ({zero_percentage_str}% > {zero_percentage_threshold * 1.5}%)')
    elif zero_percentage > zero_percentage_threshold:
        statistical_score += 5
        reasons.append(f'Высокий процент нулевых значений ({zero_percentage:.1f}% > {zero_percentage_threshold}%)')
    else:
        statistical_score += 15
    
    # Оценка по проценту смены знака
    if np.isnan(sign_changes_percentage) or sign_changes_percentage > 50:
        statistical_score += 0
        sign_changes_str = f"{sign_changes_percentage:.1f}" if not np.isnan(sign_changes_percentage) else "?"
        reasons.append(f'Частое чередование знаков ({sign_changes_str}% значений)')
    elif sign_changes_percentage > 30:
        statistical_score += 5
        reasons.append(f'Умеренное чередование знаков ({sign_changes_percentage:.1f}% значений)')
    else:
        statistical_score += 15
    
    # Оценка по энтропии
    if np.isnan(entropy) or entropy > 0.8:
        statistical_score += 0
        entropy_str = f"{entropy:.2f}" if not np.isnan(entropy) else "?"
        reasons.append(f'Высокая энтропия (хаотичность) ряда ({entropy_str})')
    elif entropy > 0.5:
        statistical_score += 5
        reasons.append(f'Повышенная энтропия (хаотичность) ряда ({entropy:.2f})')
    else:
        statistical_score += 10
    
    # Оценка по стационарности
    stationarity_threshold = 0.05  # Стандартный порог для p-value в тесте на стационарность
    if np.isnan(stationarity_p_value) or stationarity_p_value > stationarity_threshold * 2:
        statistical_score += 0
        p_value_str = f"{stationarity_p_value:.3f}" if not np.isnan(stationarity_p_value) else "?"
        reasons.append(f'Временной ряд не стационарен (p = {p_value_str})')
    elif stationarity_p_value > stationarity_threshold:
        statistical_score += 5
    else:
        statistical_score += 10
        
    # Нормализация статистической оценки до 100 баллов
    max_statistical_score = 25 + 25 + 15 + 15 + 10 + 10
    normalized_statistical_score = (statistical_score / max_statistical_score) * 100
    
    # Оценка на основе точности моделей (если доступна)
    model_score = 0
    if model_evaluation and "error" not in model_evaluation:
        best_mape = model_evaluation.get('best_mape', float('inf'))
        best_model = model_evaluation.get('best_model', 'Нет данных')
        
        # Оценка по MAPE
        if best_mape < 10:
            model_score = 100  # Отличная точность (MAPE < 10%)
        elif best_mape < 20:
            model_score = 75   # Хорошая точность (MAPE < 20%)
        elif best_mape < 30:
            model_score = 50   # Средняя точность (MAPE < 30%)
        elif best_mape < 50:
            model_score = 25   # Низкая точность (MAPE < 50%)
        else:
            model_score = 0    # Очень низкая точность (MAPE >= 50%)
        
        if best_mape > 30:
            reasons.append(f'Низкая точность прогнозных моделей (MAPE = {best_mape:.1f}% > 30%)')
    
    # Итоговая оценка: комбинация статистической оценки и оценки моделей
    # Если оценка моделей доступна, учитываем её с весом 60%, статистическую с весом 40%
    # Если оценка моделей недоступна, используем только статистическую оценку
    if model_evaluation and "error" not in model_evaluation:
        final_score = 0.4 * normalized_statistical_score + 0.6 * model_score
    else:
        final_score = normalized_statistical_score
    
    # Формирование заключения
    if final_score >= 70:
        conclusion = 'Хорошо прогнозируемый ряд'
    elif final_score >= 40:
        conclusion = 'Средне прогнозируемый ряд'
    else:
        conclusion = 'Плохо прогнозируемый ряд'
    
    # Если причин нет, но прогнозируемость хорошая
    if not reasons and conclusion == 'Хорошо прогнозируемый ряд':
        reasons.append('Стабильный временной ряд с предсказуемыми паттернами')
    
    return {
        'cv': cv,
        'autocorr': autocorr,
        'zero_percentage': zero_percentage,
        'sign_changes': sign_changes,
        'sign_changes_percentage': sign_changes_percentage,
        'entropy': entropy,
        'lyapunov': lyapunov,
        'stationarity_p_value': stationarity_p_value,
        'conclusion': conclusion,
        'reasons': reasons,
        'score': final_score,
        'model_evaluation': model_evaluation
    }

def create_time_series_plot(df, date_col, value_col, title=None):
    """Создание графика временного ряда с помощью Plotly"""
    try:
        fig = px.line(df, x=date_col, y=value_col, title=title or f"Временной ряд: {value_col}")
        fig.update_layout(height=400, showlegend=False, template="plotly_white")
        
        # Добавляем скользящее среднее для визуального сглаживания
        if len(df) > 5:
            df_valid = df.dropna(subset=[value_col])
            if len(df_valid) > 5:
                window_size = max(3, len(df_valid) // 10)
                df_valid['rolling_mean'] = df_valid[value_col].rolling(window=window_size, center=True).mean()
                fig.add_scatter(x=df_valid[date_col], y=df_valid['rolling_mean'],
                               mode='lines', line=dict(color='red', width=1.5, dash='dash'),
                               name='Скользящее среднее')
        
        return fig
    except Exception as e:
        # В случае ошибки создаем пустой график с сообщением
        fig = go.Figure()
        fig.update_layout(
            title=f"Не удалось построить график для {value_col}",
            annotations=[dict(
                text=f"Ошибка: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )],
            height=400
        )
        return fig

def create_distribution_plot(series, title=None):
    """Создание гистограммы распределения с помощью Plotly"""
    try:
        fig = px.histogram(series.dropna(), title=title or "Распределение значений",
                          nbins=min(30, len(np.unique(series.dropna()))))
        
        # Добавляем кривую плотности
        fig.update_traces(opacity=0.7)
        fig.update_layout(height=300, showlegend=False, template="plotly_white")
        
        return fig
    except Exception as e:
        # В случае ошибки создаем пустой график с сообщением
        fig = go.Figure()
        fig.update_layout(
            title="Не удалось построить гистограмму",
            annotations=[dict(
                text=f"Ошибка: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )],
            height=300
        )
        return fig

def create_acf_plot(series, lags=10, title=None):
    """Создание графика автокорреляции с помощью Plotly"""
    # Рассчитываем автокорреляцию
    try:
        series_clean = series.dropna()
        if len(series_clean) < lags + 1:
            lags = max(1, len(series_clean) - 1)
            
        acf_values = acf(series_clean, nlags=lags)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(lags + 1)),
            y=acf_values,
            name='ACF'
        ))
        
        # Добавляем границы доверительного интервала (95%)
        confidence = 1.96 / np.sqrt(len(series_clean))
        fig.add_trace(go.Scatter(
            x=list(range(lags + 1)),
            y=[confidence] * (lags + 1),
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='95% Confidence'
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(lags + 1)),
            y=[-confidence] * (lags + 1),
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='95% Confidence'
        ))
        
        fig.update_layout(
            title=title or 'График автокорреляции',
            xaxis_title='Лаг',
            yaxis_title='Значение ACF',
            height=300,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        # В случае ошибки создаем пустой график с сообщением
        fig = go.Figure()
        fig.update_layout(
            title="Не удалось построить график автокорреляции",
            annotations=[dict(
                text=f"Ошибка: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )],
            height=300
        )
        return fig

def create_lag_plot(series, lag=1, title=None):
    """Создание диаграммы рассеяния значений (t) и (t-lag)"""
    try:
        # Создаем DataFrame с текущими и запаздывающими значениями
        series_clean = series.dropna()
        if len(series_clean) <= lag:
            raise ValueError(f"Длина ряда ({len(series_clean)}) меньше или равна лагу ({lag})")
        
        current_values = series_clean.iloc[lag:].values
        lagged_values = series_clean.iloc[:-lag].values
        
        fig = go.Figure()
        
        # Добавляем диаграмму рассеяния
        fig.add_trace(go.Scatter(
            x=lagged_values,
            y=current_values,
            mode='markers',
            marker=dict(
                color='blue',
                size=8,
                opacity=0.6
            ),
            name='Lag Plot'
        ))
        
        # Добавляем линию y=x для сравнения
        min_val = min(np.min(lagged_values), np.min(current_values))
        max_val = max(np.max(lagged_values), np.max(current_values))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(
                color='red',
                width=1,
                dash='dash'
            ),
            name='y=x'
        ))
        
        # Расчет коэффициента корреляции
        correlation = np.corrcoef(lagged_values, current_values)[0, 1]
        annotation_text = f"Корреляция: {correlation:.2f}"
        
        fig.update_layout(
            title=title or f'Lag Plot (lag={lag}): Зависимость значений от предыдущих',
            xaxis_title=f'Значение (t-{lag})',
            yaxis_title='Значение (t)',
            height=400,
            showlegend=False,
            template="plotly_white",
            annotations=[
                dict(
                    x=0.95,
                    y=0.05,
                    xref="paper",
                    yref="paper",
                    text=annotation_text,
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            ]
        )
        
        return fig
    except Exception as e:
        # В случае ошибки создаем пустой график с сообщением
        fig = go.Figure()
        fig.update_layout(
            title="Не удалось построить Lag Plot",
            annotations=[dict(
                text=f"Ошибка: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )],
            height=400
        )
        return fig

def create_box_plot(series, title=None):
    """Создание ящика с усами (Box Plot) для визуализации распределения данных"""
    try:
        series_clean = series.dropna()
        if len(series_clean) < 4:
            raise ValueError(f"Недостаточно данных для построения Box Plot (необходимо минимум 4 точки)")
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=series_clean,
            name='',
            boxpoints='all',  # Показывать все точки
            jitter=0.5,       # Разброс точек по горизонтали для лучшей видимости
            pointpos=0,       # Позиция точек: 0 - по центру (что соответствует боксу)
            marker=dict(
                color='blue',
                size=6,
                opacity=0.7
            ),
            line=dict(
                color='darkblue', 
                width=2
            ),
            fillcolor='lightblue',
            whiskerwidth=0.8,
            notched=False
        ))
        
        fig.update_layout(
            title=title or 'Box Plot: Распределение значений',
            yaxis_title='Значение',
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        # В случае ошибки создаем пустой график с сообщением
        fig = go.Figure()
        fig.update_layout(
            title="Не удалось построить Box Plot",
            annotations=[dict(
                text=f"Ошибка: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )],
            height=400
        )
        return fig

def create_qq_plot(series, title=None):
    """Создание QQ-Plot для сравнения с нормальным распределением"""
    try:
        from scipy import stats as scipy_stats
        
        series_clean = series.dropna()
        if len(series_clean) < 4:
            raise ValueError(f"Недостаточно данных для построения QQ-Plot (необходимо минимум 4 точки)")
        
        # Расчет квантилей
        sorted_data = np.sort(series_clean)
        n = len(sorted_data)
        quantiles = np.arange(1, n + 1) / (n + 1)  # Quantiles from 1/(n+1) to n/(n+1)
        theoretical_quantiles = scipy_stats.norm.ppf(quantiles)
        
        fig = go.Figure()
        
        # Добавляем диаграмму рассеяния с точками
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_data,
            mode='markers',
            marker=dict(
                color='blue',
                size=6,
                opacity=0.7
            ),
            name='Data Points'
        ))
        
        # Добавляем линию y=ax+b
        slope, intercept = np.polyfit(theoretical_quantiles, sorted_data, 1)
        line_x = np.array([min(theoretical_quantiles), max(theoretical_quantiles)])
        line_y = slope * line_x + intercept
        
        fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            line=dict(
                color='red',
                width=1.5
            ),
            name='Fitted Line'
        ))
        
        fig.update_layout(
            title=title or 'QQ-Plot: Сравнение с нормальным распределением',
            xaxis_title='Теоретические квантили',
            yaxis_title='Фактические квантили',
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        # В случае ошибки создаем пустой график с сообщением
        fig = go.Figure()
        fig.update_layout(
            title="Не удалось построить QQ-Plot",
            annotations=[dict(
                text=f"Ошибка: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )],
            height=400
        )
        return fig

def create_decomposition_plot(df, date_col, value_col, model="additive", title=None):
    """Создание графика декомпозиции временного ряда на тренд, сезонность и остатки"""
    try:
        # Проверяем, достаточно ли данных для декомпозиции
        df_valid = df.dropna(subset=[date_col, value_col]).sort_values(by=date_col)
        if len(df_valid) < 8:
            raise ValueError(f"Недостаточно данных для декомпозиции (необходимо минимум 8 точек)")
        
        # Проверяем, равномерно ли распределены даты
        # Установка индекса
        ts = df_valid.set_index(date_col)[value_col]
        
        # Определяем период на основе частоты данных
        # При месячных данных период=12, при квартальных период=4
        timedelta = pd.Series(ts.index).diff().mode()[0]
        if timedelta.days > 364 and timedelta.days < 366:
            # Годовые данные
            period = 1
        elif timedelta.days > 89 and timedelta.days < 93:
            # Квартальные данные
            period = 4
        elif timedelta.days > 27 and timedelta.days < 32:
            # Месячные данные
            period = 12
        else:
            # Если данные не месячные и не квартальные, устанавливаем период как 25% от длины данных
            period = max(2, int(len(ts) * 0.25))
        
        # Если данных меньше чем 2*период, уменьшаем период
        if len(ts) < 2 * period:
            period = max(2, len(ts) // 2)
        
        # Делаем декомпозицию
        result = seasonal_decompose(
            ts, 
            model=model, 
            period=period,
            extrapolate_trend='freq'
        )
        
        # Создаем фигуру с подграфиками
        fig = make_subplots(
            rows=4, 
            cols=1,
            subplot_titles=('Исходный ряд', 'Тренд', 'Сезонность', 'Остатки'),
            vertical_spacing=0.1
        )
        
        # Добавляем исходный ряд
        fig.add_trace(
            go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Исходный ряд'),
            row=1, col=1
        )
        
        # Добавляем тренд
        fig.add_trace(
            go.Scatter(x=ts.index, y=result.trend, mode='lines', name='Тренд', line=dict(color='red')),
            row=2, col=1
        )
        
        # Добавляем сезонность
        fig.add_trace(
            go.Scatter(x=ts.index, y=result.seasonal, mode='lines', name='Сезонность', line=dict(color='green')),
            row=3, col=1
        )
        
        # Добавляем остатки
        fig.add_trace(
            go.Scatter(x=ts.index, y=result.resid, mode='lines', name='Остатки', line=dict(color='purple')),
            row=4, col=1
        )
        
        # Добавляем нулевую линию для сезонности и остатков
        fig.add_trace(
            go.Scatter(x=[ts.index.min(), ts.index.max()], y=[0, 0], mode='lines', 
                      line=dict(color='black', width=0.5, dash='dash'), showlegend=False),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=[ts.index.min(), ts.index.max()], y=[0, 0], mode='lines', 
                      line=dict(color='black', width=0.5, dash='dash'), showlegend=False),
            row=4, col=1
        )
        
        # Обновляем макет
        fig.update_layout(
            title=title or f'Декомпозиция временного ряда ({model}): {value_col}',
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        # В случае ошибки создаем пустой график с сообщением
        fig = go.Figure()
        fig.update_layout(
            title="Не удалось построить декомпозицию временного ряда",
            annotations=[dict(
                text=f"Ошибка: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )],
            height=800
        )
        return fig

def create_seasonality_heatmap(df, date_col, value_col, title=None):
    """Создание тепловой карты сезонности (годы по строкам, месяцы по столбцам)"""
    try:
        # Подготавливаем данные
        df_valid = df.dropna(subset=[date_col, value_col])
        if len(df_valid) < 12:
            raise ValueError(f"Недостаточно данных для тепловой карты сезонности (нужно минимум 12 точек)")
        
        # Определяем частоту данных
        df_sorted = df_valid.sort_values(by=date_col)
        date_diffs = pd.Series(df_sorted[date_col]).diff().dropna()
        
        if len(date_diffs) == 0:
            raise ValueError("Невозможно определить частоту данных")
        
        # Определяем модальную разницу между датами
        modal_diff = date_diffs.mode()[0]
        
        # Определяем тип сезонности в зависимости от частоты данных
        if modal_diff.days <= 1:
            # Дневные или более частые данные - группируем по месяцам и дням месяца
            df_valid['season_x'] = df_valid[date_col].dt.day
            df_valid['season_y'] = df_valid[date_col].dt.month
            x_label = "День месяца"
            y_label = "Месяц"
        elif modal_diff.days <= 7:
            # Недельные данные - группируем по дням недели и месяцам
            df_valid['season_x'] = df_valid[date_col].dt.dayofweek
            df_valid['season_y'] = df_valid[date_col].dt.month
            x_label = "День недели"
            y_label = "Месяц"
            # Заменяем числовые значения дней недели на их названия
            day_names = {0: 'Пн', 1: 'Вт', 2: 'Ср', 3: 'Чт', 4: 'Пт', 5: 'Сб', 6: 'Вс'}
            df_valid['season_x'] = df_valid['season_x'].map(day_names)
        elif modal_diff.days <= 31:
            # Месячные данные - группируем по месяцам и годам
            df_valid['season_x'] = df_valid[date_col].dt.month
            df_valid['season_y'] = df_valid[date_col].dt.year
            x_label = "Месяц"
            y_label = "Год"
            # Заменяем числовые значения месяцев на их названия
            month_names = {1: 'Янв', 2: 'Фев', 3: 'Мар', 4: 'Апр', 5: 'Май', 6: 'Июн', 
                          7: 'Июл', 8: 'Авг', 9: 'Сен', 10: 'Окт', 11: 'Ноя', 12: 'Дек'}
            df_valid['season_x'] = df_valid['season_x'].map(month_names)
        elif modal_diff.days <= 92:
            # Квартальные данные - группируем по кварталам и годам
            df_valid['season_x'] = df_valid[date_col].dt.quarter
            df_valid['season_y'] = df_valid[date_col].dt.year
            x_label = "Квартал"
            y_label = "Год"
            # Заменяем числовые значения кварталов на их названия
            quarter_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
            df_valid['season_x'] = df_valid['season_x'].map(quarter_names)
        else:
            # Годовые данные или более редкие - группируем по годам и декадам
            df_valid['season_x'] = df_valid[date_col].dt.year % 10  # Год внутри декады
            df_valid['season_y'] = df_valid[date_col].dt.year // 10 * 10  # Декада (2020, 2010, etc.)
            x_label = "Год в декаде"
            y_label = "Декада"
        
        # Создаем сводную таблицу для тепловой карты
        pivot_table = df_valid.pivot_table(
            values=value_col,
            index='season_y',
            columns='season_x',
            aggfunc='mean'
        )
        
        # Сортируем индексы и колонки
        pivot_table = pivot_table.sort_index(axis=0)
        
        # Если есть пропущенные значения, заполняем их средними по столбцу
        # но только если в столбце есть хотя бы одно непропущенное значение
        for col in pivot_table.columns:
            if pivot_table[col].count() > 0:
                pivot_table[col] = pivot_table[col].fillna(pivot_table[col].mean())
        
        # Если после заполнения средними все еще остались пропуски, заполняем их нулями
        pivot_table = pivot_table.fillna(0)
        
        # Определяем цветовую шкалу в зависимости от данных
        # Если есть и положительные и отрицательные значения, используем RdBu_r
        # Иначе используем однонаправленную шкалу
        if (pivot_table.min().min() < 0) and (pivot_table.max().max() > 0):
            color_scale = 'RdBu_r'
        elif pivot_table.min().min() < 0:
            color_scale = 'Blues_r'  # Для отрицательных значений
        else:
            color_scale = 'Reds'  # Для положительных значений
        
        # Создаем тепловую карту с улучшенными метками
        fig = px.imshow(
            pivot_table,
            labels=dict(x=x_label, y=y_label, color=value_col),
            x=pivot_table.columns,
            y=pivot_table.index,
            color_continuous_scale=color_scale,
            aspect='auto',
            text_auto='.1f'  # Автоматически показываем значения до 1 десятичного знака
        )
        
        # Обновляем макет для лучшей читаемости
        fig.update_layout(
            title=title or f'Тепловая карта сезонности: {value_col}',
            height=400,
            template="plotly_white",
            coloraxis_colorbar=dict(
                title=value_col,
                tickformat='.1f'
            ),
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(pivot_table.columns))),
                ticktext=[str(col) for col in pivot_table.columns]
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(pivot_table.index))),
                ticktext=[str(idx) for idx in pivot_table.index]
            )
        )
        
        return fig
    except Exception as e:
        # В случае ошибки создаем пустой график с сообщением
        fig = go.Figure()
        fig.update_layout(
            title="Не удалось построить тепловую карту сезонности",
            annotations=[dict(
                text=f"Ошибка: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )],
            height=400
        )
        return fig

def create_model_evaluation_plot(df, date_col, value_col, model_results, test_size=0.2, title=None):
    """Создание графика фактических и прогнозных значений на тестовой выборке"""
    try:
        if "error" in model_results:
            raise ValueError(model_results["error"])
        
        # Подготавливаем данные
        df_sorted = df.sort_values(by=date_col).copy()
        df_sorted['numeric_date'] = pd.to_numeric(df_sorted[date_col])
        
        # Удаляем строки с пропущенными значениями
        df_clean = df_sorted.dropna(subset=[value_col, 'numeric_date'])
        
        # Проверяем, достаточно ли данных
        if len(df_clean) < 10:
            raise ValueError("Недостаточно данных для оценки моделей")
        
        # Создаем признаки: предыдущие значения
        df_clean['prev_value'] = df_clean[value_col].shift(1)
        df_clean = df_clean.dropna()
        
        # Разделяем данные на обучающую и тестовую выборки
        train_size = int(len(df_clean) * (1 - test_size))
        train_df = df_clean.iloc[:train_size]
        test_df = df_clean.iloc[train_size:]
        
        if len(train_df) < 5 or len(test_df) < 3:
            raise ValueError("Недостаточно данных для обучения и тестирования моделей")
        
        # Получаем фактические значения
        actual_values = test_df[value_col].values
        
        # Получаем прогнозные значения наивной модели
        naive_pred = test_df['prev_value'].values
        
        # Получаем прогнозные значения линейной регрессии
        X_train = train_df[['numeric_date', 'prev_value']]
        y_train = train_df[value_col]
        X_test = test_df[['numeric_date', 'prev_value']]
        
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        # Получаем прогнозные значения Random Forest (если доступны)
        if model_results.get('rf_mae') is not None:
            try:
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
            except:
                rf_pred = None
        else:
            rf_pred = None
        
        # Создаем график фактических и прогнозных значений
        fig = go.Figure()
        
        # Добавляем фактические значения
        fig.add_trace(go.Scatter(
            x=test_df[date_col],
            y=actual_values,
            mode='lines+markers',
            name='Фактические значения',
            line=dict(color='blue', width=2)
        ))
        
        # Добавляем прогнозные значения наивной модели
        fig.add_trace(go.Scatter(
            x=test_df[date_col],
            y=naive_pred,
            mode='lines+markers',
            name='Наивная модель',
            line=dict(color='green', width=1.5, dash='dot')
        ))
        
        # Добавляем прогнозные значения линейной регрессии
        fig.add_trace(go.Scatter(
            x=test_df[date_col],
            y=lr_pred,
            mode='lines+markers',
            name='Линейная регрессия',
            line=dict(color='red', width=1.5, dash='dash')
        ))
        
        # Добавляем прогнозные значения Random Forest (если доступны)
        if rf_pred is not None:
            fig.add_trace(go.Scatter(
                x=test_df[date_col],
                y=rf_pred,
                mode='lines+markers',
                name='Random Forest',
                line=dict(color='purple', width=1.5, dash='dashdot')
            ))
        
        # Добавляем аннотацию с метриками
        annotations = []
        
        annotations.append(dict(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"Лучшая модель: {model_results['best_model']} (MAPE = {model_results['best_mape']:.1f}%)",
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        ))
        
        fig.update_layout(
            title=title or f'Сравнение фактических и прогнозных значений: {value_col}',
            xaxis_title='Дата',
            yaxis_title='Значение',
            height=500,
            showlegend=True,
            template="plotly_white",
            annotations=annotations
        )
        
        return fig
    except Exception as e:
        # В случае ошибки создаем пустой график с сообщением
        fig = go.Figure()
        fig.update_layout(
            title="Не удалось построить график сравнения моделей",
            annotations=[dict(
                text=f"Ошибка: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )],
            height=500
        )
        return fig

# Загрузка файла
uploaded_file = st.file_uploader("Выберите CSV или Excel файл", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    st.success("Файл успешно загружен. Анализирую данные...")
    
    # Определяем тип файла и загружаем данные
    try:
        if uploaded_file.name.endswith('.csv'):
            # Попробуем разные разделители
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(uploaded_file, sep=sep)
                    if len(df.columns) > 1:  # Если успешно разделено на колонки
                        break
                except:
                    continue
        else:  # Excel
            df = pd.read_excel(uploaded_file)
        
        # Сбрасываем указатель файла для повторного использования
        uploaded_file.seek(0)
        
        # Показываем превью данных
        st.subheader("Предварительный просмотр данных")
        st.dataframe(df.head())
        
        # Определяем столбец с датой
        date_col = detect_date_column(df)
        
        if date_col:
            # Преобразуем столбец с датой
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            st.info(f"Обнаружен столбец с датой: {date_col}")
            
            # Позволяем пользователю выбрать столбец с датой, если автоопределение некорректно
            date_col = st.selectbox(
                "Выберите столбец с датой",
                options=[date_col] + [col for col in df.columns if col != date_col],
                index=0
            )
            
            # Преобразуем выбранный столбец с датой
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Определяем числовые столбцы
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != date_col]
            
            if not numeric_cols:
                st.error("В файле не найдены числовые столбцы для анализа.")
            else:
                st.success(f"Найдено {len(numeric_cols)} числовых столбцов для анализа.")
                
                # Опционально: выбор столбцов для анализа
                selected_cols = st.multiselect(
                    "Выберите столбцы для анализа (если не выбрано, будут проанализированы все числовые столбцы)",
                    options=numeric_cols,
                    default=[]
                )
                
                if selected_cols:
                    numeric_cols = selected_cols
                
                # Показываем общую информацию
                st.subheader("Общая информация о данных")
                
                # Количество строк, период данных
                total_rows = len(df)
                date_range = f"с {df[date_col].min().strftime('%d.%m.%Y')} по {df[date_col].max().strftime('%d.%m.%Y')}"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Количество строк", total_rows)
                with col2:
                    st.metric("Период данных", date_range)
                
                # Создаем группы для хорошо и плохо прогнозируемых рядов
                good_forecasting = []
                medium_forecasting = []
                bad_forecasting = []
                
                # Словарь для хранения результатов анализа
                analysis_results = {}
                
                # Прогресс-бар для отслеживания анализа
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Анализируем каждый числовой столбец
                for i, col in enumerate(numeric_cols):
                    # Обновляем прогресс
                    progress = (i + 1) / len(numeric_cols)
                    progress_bar.progress(progress)
                    progress_text.text(f"Анализирую столбец {i+1} из {len(numeric_cols)}: {col}")
                    
                    # Подготавливаем данные
                    series = df[col].copy()
                    
                    # Рассчитываем статистические метрики
                    stats_results = {
                        'cv': calculate_cv(series),
                        'autocorr': calculate_autocorrelation(series, lag=1),
                        'zero_percentage': calculate_zero_percentage(series),
                        'entropy': calculate_entropy(series),
                        'lyapunov': estimate_lyapunov(series),
                    }
                    
                    # Рассчитываем изменения знака
                    sign_changes, sign_changes_percentage = calculate_sign_changes(series)
                    stats_results['sign_changes'] = sign_changes
                    stats_results['sign_changes_percentage'] = sign_changes_percentage
                    
                    # Тест на стационарность
                    adf_result, p_value = test_stationarity(series)
                    stats_results['stationarity_test'] = adf_result
                    stats_results['stationarity_p_value'] = p_value
                    
                    # Оцениваем простые модели прогнозирования
                    model_evaluation = evaluate_simple_models(df, date_col, col, test_size=test_size)
                    
                    # Анализируем потенциал прогнозирования
                    analysis = analyze_forecasting_potential(series, stats_results, model_evaluation)
                    
                    # Сохраняем результаты анализа
                    analysis_results[col] = {
                        'stats': stats_results,
                        'analysis': analysis,
                        'model_evaluation': model_evaluation
                    }
                    
                    # Определяем группу прогнозируемости
                    if analysis['conclusion'] == 'Хорошо прогнозируемый ряд':
                        good_forecasting.append(col)
                    elif analysis['conclusion'] == 'Средне прогнозируемый ряд':
                        medium_forecasting.append(col)
                    else:
                        bad_forecasting.append(col)
                
                # Скрываем прогресс-бар и текст
                progress_bar.empty()
                progress_text.empty()
                
                # Сводная информация
                st.subheader("Сводная информация по всем столбцам")
                
                # Создаем DataFrame с результатами анализа для сравнения
                summary_data = []
                for col in numeric_cols:
                    result = analysis_results[col]
                    summary_data.append({
                        'Столбец': col,
                        'CV': f"{result['stats']['cv']:.2f}" if not np.isnan(result['stats']['cv']) else "∞",
                        'Автокорреляция': f"{result['stats']['autocorr']:.2f}" if not np.isnan(result['stats']['autocorr']) else "N/A",
                        '% нулей': f"{result['stats']['zero_percentage']:.1f}%" if not np.isnan(result['stats']['zero_percentage']) else "N/A",
                        'Энтропия': f"{result['stats']['entropy']:.2f}" if not np.isnan(result['stats']['entropy']) else "N/A",
                        'Стационарность (p)': f"{result['stats']['stationarity_p_value']:.3f}" if not np.isnan(result['stats']['stationarity_p_value']) else "N/A",
                        'Оценка': f"{result['analysis']['score']:.1f}",
                        'Заключение': result['analysis']['conclusion']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                
                # Отображаем результаты в виде таблицы с цветовой кодировкой
                def highlight_conclusion(val):
                    if val == 'Хорошо прогнозируемый ряд':
                        return 'background-color: lightgreen'
                    elif val == 'Средне прогнозируемый ряд':
                        return 'background-color: khaki'
                    else:
                        return 'background-color: lightcoral'
                
                # Стилизуем DataFrame
                styled_df = summary_df.style.applymap(
                    highlight_conclusion, 
                    subset=['Заключение']
                )
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Добавляем кнопку для выгрузки результатов в Excel
                if len(summary_df) > 0:
                    # Создаем буфер для Excel-файла
                    excel_buffer = io.BytesIO()
                    
                    # Создаем Excel-writer
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        # Записываем сводную таблицу на первый лист
                        summary_df.to_excel(writer, sheet_name='Сводная информация', index=False)
                        
                        # Записываем детальную информацию на отдельные листы для каждого столбца
                        for col in numeric_cols:
                            # Создаем DataFrame с детальными метриками
                            stats = analysis_results[col]['stats']
                            analysis = analysis_results[col]['analysis']
                            model_eval = analysis_results[col]['model_evaluation']
                            
                            metrics = []
                            values = []
                            
                            # Добавляем статистические метрики
                            metrics.append('Коэффициент вариации (CV)')
                            values.append(f"{stats['cv']:.2f}" if not np.isnan(stats['cv']) else "∞")
                            
                            metrics.append('Автокорреляция')
                            values.append(f"{stats['autocorr']:.2f}" if not np.isnan(stats['autocorr']) else "N/A")
                            
                            metrics.append('% нулевых значений')
                            values.append(f"{stats['zero_percentage']:.1f}%" if not np.isnan(stats['zero_percentage']) else "N/A")
                            
                            metrics.append('% смены знака')
                            values.append(f"{stats['sign_changes_percentage']:.1f}%" if not np.isnan(stats['sign_changes_percentage']) else "N/A")
                            
                            metrics.append('Энтропия')
                            values.append(f"{stats['entropy']:.2f}" if not np.isnan(stats['entropy']) else "N/A")
                            
                            metrics.append('Коэф. Ляпунова')
                            values.append(f"{stats['lyapunov']:.2f}" if not np.isnan(stats['lyapunov']) else "N/A")
                            
                            metrics.append('P-значение стационарности')
                            values.append(f"{stats['stationarity_p_value']:.3f}" if not np.isnan(stats['stationarity_p_value']) else "N/A")
                            
                            metrics.append('Итоговая оценка')
                            values.append(f"{analysis['score']:.1f}")
                            
                            # Добавляем метрики моделей если есть
                            if "error" not in model_eval:
                                metrics.append('Наивная модель (MAPE)')
                                values.append(f"{model_eval.get('naive_mape', 'N/A'):.1f}%" if model_eval.get('naive_mape') is not None else "N/A")
                                
                                metrics.append('Линейная регрессия (MAPE)')
                                values.append(f"{model_eval.get('lr_mape', 'N/A'):.1f}%" if model_eval.get('lr_mape') is not None else "N/A")
                                
                                metrics.append('Random Forest (MAPE)')
                                if model_eval.get('rf_mape') is not None:
                                    values.append(f"{model_eval['rf_mape']:.1f}%")
                                else:
                                    values.append("N/A")
                                
                                metrics.append('Лучшая модель')
                                values.append(model_eval.get('best_model', 'N/A'))
                            
                            metrics.append('Заключение')
                            values.append(analysis['conclusion'])
                            
                            # Добавляем причины если есть
                            if analysis['reasons']:
                                metrics.append('Причины')
                                values.append("; ".join(analysis['reasons']))
                            
                            # Создаем и записываем DataFrame
                            detail_df = pd.DataFrame({'Метрика': metrics, 'Значение': values})
                            sheet_name = col[:30]  # Ограничиваем длину имени листа для Excel
                            detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Предлагаем пользователю скачать файл
                    excel_buffer.seek(0)
                    st.download_button(
                        label="Скачать анализ в Excel",
                        data=excel_buffer,
                        file_name="Анализ_прогнозируемости.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                
                # Создаем три секции с разной прогнозируемостью
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🟢 Хорошо прогнозируемые")
                    if good_forecasting:
                        for col in good_forecasting:
                            st.markdown(f"- **{col}** (оценка: {analysis_results[col]['analysis']['score']:.1f})")
                    else:
                        st.markdown("*Нет столбцов в этой категории*")
                
                with col2:
                    st.markdown("### 🟠 Средне прогнозируемые")
                    if medium_forecasting:
                        for col in medium_forecasting:
                            st.markdown(f"- **{col}** (оценка: {analysis_results[col]['analysis']['score']:.1f})")
                    else:
                        st.markdown("*Нет столбцов в этой категории*")
                
                with col3:
                    st.markdown("### 🔴 Плохо прогнозируемые")
                    if bad_forecasting:
                        for col in bad_forecasting:
                            st.markdown(f"- **{col}** (оценка: {analysis_results[col]['analysis']['score']:.1f})")
                    else:
                        st.markdown("*Нет столбцов в этой категории*")
                
                # Переход к детальному анализу
                st.markdown("---")
                st.subheader("Детальный анализ по каждому столбцу")
                
                # Создаем вкладки для каждого столбца
                column_tabs = st.tabs(numeric_cols)
                
                for i, col in enumerate(numeric_cols):
                    with column_tabs[i]:
                        # Получаем результаты анализа
                        analysis = analysis_results[col]['analysis']
                        stats = analysis_results[col]['stats']
                        model_evaluation = analysis_results[col]['model_evaluation']
                        
                        # Определяем цвет для заключения
                        if analysis['conclusion'] == 'Хорошо прогнозируемый ряд':
                            conclusion_color = 'green'
                        elif analysis['conclusion'] == 'Средне прогнозируемый ряд':
                            conclusion_color = 'orange'
                        else:
                            conclusion_color = 'red'
                        
                        # Отображаем заключение
                        st.markdown(f"### Заключение: <span style='color:{conclusion_color}'>{analysis['conclusion']}</span>", unsafe_allow_html=True)
                        
                        # Отображаем обоснование
                        if analysis['reasons']:
                            st.markdown("#### Причины:")
                            for reason in analysis['reasons']:
                                st.markdown(f"- {reason}")
                        
                        # Отображаем метрики в две строки
                        metrics_row1_col1, metrics_row1_col2, metrics_row1_col3, metrics_row1_col4 = st.columns(4)
                        with metrics_row1_col1:
                            st.metric("Коэффициент вариации (CV)", f"{stats['cv']:.2f}" if not np.isnan(stats['cv']) else "∞")
                        with metrics_row1_col2:
                            st.metric("Автокорреляция", f"{stats['autocorr']:.2f}" if not np.isnan(stats['autocorr']) else "N/A")
                        with metrics_row1_col3:
                            st.metric("% нулевых значений", f"{stats['zero_percentage']:.1f}%" if not np.isnan(stats['zero_percentage']) else "N/A")
                        with metrics_row1_col4:
                            st.metric("% смены знака", f"{stats['sign_changes_percentage']:.1f}%" if not np.isnan(stats['sign_changes_percentage']) else "N/A")
                        
                        metrics_row2_col1, metrics_row2_col2, metrics_row2_col3, metrics_row2_col4 = st.columns(4)
                        with metrics_row2_col1:
                            st.metric("Энтропия", f"{stats['entropy']:.2f}" if not np.isnan(stats['entropy']) else "N/A")
                        with metrics_row2_col2:
                            st.metric("Коэф. Ляпунова", f"{stats['lyapunov']:.2f}" if not np.isnan(stats['lyapunov']) else "N/A")
                        with metrics_row2_col3:
                            st.metric("P-значение стационарности", f"{stats['stationarity_p_value']:.3f}" if not np.isnan(stats['stationarity_p_value']) else "N/A")
                        with metrics_row2_col4:
                            st.metric("Итоговая оценка", f"{analysis['score']:.1f}")
                        
                        # Показываем результаты оценки моделей
                        if model_evaluation and "error" not in model_evaluation:
                            st.markdown("#### Оценка точности простых моделей прогнозирования:")
                            
                            model_col1, model_col2, model_col3 = st.columns(3)
                            with model_col1:
                                st.metric("Наивная модель (MAPE)", f"{model_evaluation['naive_mape']:.1f}%")
                            with model_col2:
                                st.metric("Линейная регрессия (MAPE)", f"{model_evaluation['lr_mape']:.1f}%")
                            with model_col3:
                                if model_evaluation.get('rf_mape') is not None:
                                    st.metric("Random Forest (MAPE)", f"{model_evaluation['rf_mape']:.1f}%")
                                else:
                                    st.metric("Random Forest (MAPE)", "N/A")
                            
                            st.markdown(f"**Лучшая модель: {model_evaluation['best_model']}** (MAPE = {model_evaluation['best_mape']:.1f}%)")
                        
                        # Создаем четыре вкладки для визуализаций
                        viz_tabs = st.tabs(["Базовая визуализация", "Распределение", "Автокорреляция", "Декомпозиция", "Сравнение моделей"])
                        
                        # Вкладка 1: Основные визуализации
                        with viz_tabs[0]:
                            # График временного ряда
                            st.plotly_chart(
                                create_time_series_plot(df, date_col, col, title=f"Временной ряд: {col}"), 
                                use_container_width=True
                            )
                            
                            # Box Plot
                            st.plotly_chart(
                                create_box_plot(df[col], title=f"Box Plot: {col}"), 
                                use_container_width=True
                            )
                        
                        # Вкладка 2: Визуализации распределения
                        with viz_tabs[1]:
                            # Гистограмма распределения
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(
                                    create_distribution_plot(df[col], title=f"Гистограмма распределения: {col}"), 
                                    use_container_width=True
                                )
                            
                            # QQ-Plot
                            with col2:
                                st.plotly_chart(
                                    create_qq_plot(df[col], title=f"QQ-Plot: {col}"), 
                                    use_container_width=True
                                )
                        
                        # Вкладка 3: Автокорреляция и Lag-Plot
                        with viz_tabs[2]:
                            col1, col2 = st.columns(2)
                            with col1:
                                # График автокорреляции
                                st.plotly_chart(
                                    create_acf_plot(df[col], title=f"Автокорреляция: {col}"), 
                                    use_container_width=True
                                )
                            
                            with col2:
                                # Lag Plot
                                st.plotly_chart(
                                    create_lag_plot(df[col], title=f"Lag Plot: {col}"), 
                                    use_container_width=True
                                )
                        
                        # Вкладка 4: Декомпозиция
                        with viz_tabs[3]:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                # Декомпозиция временного ряда
                                st.plotly_chart(
                                    create_decomposition_plot(df, date_col, col, model=decomposition_model, title=f"Декомпозиция: {col}"), 
                                    use_container_width=True
                                )
                            
                            with col2:
                                # Тепловая карта сезонности
                                st.plotly_chart(
                                    create_seasonality_heatmap(df, date_col, col, title=f"Тепловая карта сезонности: {col}"), 
                                    use_container_width=True
                                )
                        
                        # Вкладка 5: Сравнение моделей
                        with viz_tabs[4]:
                            if model_evaluation and "error" not in model_evaluation:
                                st.plotly_chart(
                                    create_model_evaluation_plot(df, date_col, col, model_evaluation, test_size=test_size, title=f"Сравнение моделей: {col}"), 
                                    use_container_width=True
                                )
                            else:
                                st.warning("Недостаточно данных для оценки моделей прогнозирования.")
                        
                        # Рекомендации
                        st.markdown("#### Рекомендации:")
                        if analysis['conclusion'] == 'Хорошо прогнозируемый ряд':
                            st.markdown("""
                            - ✅ **Рекомендуемые модели**: ARIMA, ETS, линейная регрессия, нейронные сети
                            - ✅ **Горизонт прогнозирования**: Возможен долгосрочный прогноз
                            - ✅ **Частота обновления**: Стандартная (месяц/квартал)
                            - ✅ **Действия**: Включить в основной прогноз, мониторить точность
                            """)
                        elif analysis['conclusion'] == 'Средне прогнозируемый ряд':
                            st.markdown("""
                            - ⚠️ **Рекомендуемые модели**: Ансамблевые методы (XGBoost, Random Forest), экспоненциальное сглаживание
                            - ⚠️ **Подготовка данных**: Рассмотрите агрегацию данных до более высокого уровня
                            - ⚠️ **Горизонт прогнозирования**: Краткосрочные прогнозы (1-3 периода вперед)
                            - ⚠️ **Действия**: Добавьте внешние переменные, используйте ансамбли моделей
                            """)
                        else:
                            st.markdown("""
                            - ❌ **Рекомендуемые подходы**: Сценарный анализ вместо точечного прогноза
                            - ❌ **Альтернативы прогнозированию**: Бюджетирование на основе средних значений
                            - ❌ **Горизонт планирования**: Только самый краткосрочный (1 период)
                            - ❌ **Действия**: Исключить из регулярного прогноза, применять экспертные оценки
                            """)
                        
                        st.markdown("---")
        else:
            st.error("Не удалось определить столбец с датой. Пожалуйста, убедитесь, что в файле есть столбец с датами.")
            
            # Предлагаем выбрать столбец с датой вручную
            date_col_options = list(df.columns)
            if date_col_options:
                manual_date_col = st.selectbox(
                    "Выберите столбец с датой вручную",
                    options=date_col_options
                )
                
                if st.button("Продолжить анализ с выбранным столбцом даты"):
                    # Продолжаем анализ с выбранным вручную столбцом
                    try:
                        df[manual_date_col] = pd.to_datetime(df[manual_date_col], errors='coerce')
                        st.experimental_rerun()
                    except:
                        st.error(f"Не удалось преобразовать столбец {manual_date_col} в формат даты.")
            
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")
        st.info("Попробуйте другой формат файла или проверьте данные.")

# Добавляем справочную информацию в сайдбар
with st.sidebar.expander("Справка по метрикам", expanded=False):
    st.markdown("""
    ### 1. Коэффициент вариации (CV)
    Отношение стандартного отклонения к среднему значению. Показывает относительный разброс данных.
    - CV < 0.3: Стабильные, предсказуемые данные
    - 0.3 ≤ CV < 1.0: Умеренная вариативность
    - 1.0 ≤ CV < 2.0: Высокая вариативность
    - CV ≥ 2.0: Экстремальная вариативность, плохо прогнозируемые данные
    
    ### 2. Автокорреляция
    Мера зависимости значений временного ряда от своих предыдущих значений.
    - |r| > 0.6: Сильная временная зависимость, высокая прогнозируемость
    - 0.3 < |r| ≤ 0.6: Умеренная временная зависимость
    - |r| ≤ 0.3: Слабая временная зависимость, низкая прогнозируемость
    
    ### 3. Процент нулевых значений
    Доля нулевых значений во временном ряде.
    - < 10%: Нормальный ряд
    - 10-40%: Требует внимания
    - > 40%: Проблематичный для прогнозирования
    
    ### 4. Процент смены знака
    Как часто значения временного ряда меняют знак с положительного на отрицательный и наоборот.
    - < 10%: Стабильный ряд с редкими изменениями знака
    - 10-30%: Умеренные колебания
    - > 30%: Частые изменения знака, трудно прогнозируемый ряд
    
    ### 5. Энтропия
    Мера хаотичности или непредсказуемости временного ряда.
    - < 0.3: Низкая хаотичность, предсказуемый ряд
    - 0.3-0.6: Средняя хаотичность
    - > 0.6: Высокая хаотичность, непредсказуемый ряд
    
    ### 6. Метрика Ляпунова
    Показатель чувствительности к начальным условиям (хаотичности).
    - < 0: Стабильный ряд
    - > 0: Хаотичный ряд (чем больше, тем более хаотичный)
    
    ### 7. Тест на стационарность (p-значение)
    Тест Дики-Фуллера для определения стационарности ряда.
    - p < 0.05: Ряд стационарен (легче прогнозировать)
    - p ≥ 0.05: Ряд нестационарен (труднее прогнозировать)
    
    ### 8. MAPE (средняя абсолютная процентная ошибка)
    Показатель точности прогнозной модели.
    - < 10%: Высокая точность
    - 10-20%: Хорошая точность
    - 20-30%: Приемлемая точность
    - > 30%: Низкая точность
    """)

with st.sidebar.expander("Как улучшить прогнозируемость", expanded=False):
    st.markdown("""
    ### Для средне прогнозируемых рядов
    1. **Агрегируйте данные**
       - Перейдите с месячных на квартальные или годовые данные
       - Используйте скользящее среднее для сглаживания
    
    2. **Трансформируйте данные**
       - Логарифмическое преобразование для данных с экспоненциальным ростом
       - Box-Cox или Yeo-Johnson преобразования для стабилизации дисперсии
       - Удаление или замена выбросов
    
    3. **Добавьте дополнительные переменные**
       - Включите индикаторы сезонности (месяц, квартал)
       - Добавьте макроэкономические показатели
       - Используйте связанные бизнес-метрики как предикторы
    
    ### Для плохо прогнозируемых рядов
    1. **Сценарный подход**
       - Вместо точечного прогноза используйте диапазоны (оптимистичный/пессимистичный)
       - Моделируйте экстремальные сценарии
       - Используйте стохастическое моделирование (Монте-Карло)
    
    2. **Декомпозиция временного ряда**
       - Разделите ряд на тренд, сезонность и случайную компоненту
       - Прогнозируйте только предсказуемые компоненты
       - Для случайной компоненты используйте доверительные интервалы
    
    3. **Переопределение задачи**
       - Вместо прогнозирования конкретных значений сфокусируйтесь на выявлении аномалий
       - Прогнозируйте направление изменения вместо абсолютных значений
       - Используйте бюджетирование на основе средних значений
    
    4. **Экспертные оценки**
       - Дополняйте статистические модели экспертными корректировками
       - Используйте качественные методы прогнозирования (метод Дельфи)
       - Комбинируйте результаты разных моделей и экспертных оценок
    """)

with st.sidebar.expander("Рекомендуемые модели", expanded=False):
    st.markdown("""
    ### Для хорошо прогнозируемых рядов
    - **ARIMA, SARIMA** - для рядов с трендом и сезонностью
    - **ETS (Exponential Smoothing)** - для рядов с мультипликативной сезонностью
    - **Prophet** - для рядов с нелинейными трендами и праздничными эффектами
    - **Нейронные сети (LSTM, GRU)** - для сложных паттернов и длинных рядов
    
    ### Для средне прогнозируемых рядов
    - **Ансамблевые методы** - XGBoost, Random Forest, Gradient Boosting
    - **Гибридные модели** - комбинации статистических методов и машинного обучения
    - **Регрессионные модели с регуляризацией** - Ridge, LASSO для борьбы с переобучением
    - **Экспоненциальное сглаживание с демпфированием** - для рядов с затухающими трендами
    
    ### Для плохо прогнозируемых рядов
    - **Наивные модели** - среднее, медиана, последнее значение
    - **Модели на основе медианы** - устойчивы к выбросам
    - **Квантильная регрессия** - для оценки интервалов прогноза
    - **Байесовские методы** - для работы с неопределенностью
    """)

with st.sidebar.expander("О модели VAR", expanded=False):
    st.markdown("""
    ### Vector Autoregression (VAR)
    
    Модель VAR (Vector Autoregression) является расширением одномерной авторегрессионной модели для многомерных временных рядов.
    
    #### Когда использовать VAR:
    
    1. **Для хорошо прогнозируемых рядов**:
       - Когда ряды имеют очевидную взаимозависимость
       - Когда необходимо моделировать взаимное влияние нескольких показателей
       - Для получения комплексного прогноза взаимосвязанных метрик
       - Когда ряды имеют схожую частоту и стационарность
    
    2. **Для средне прогнозируемых рядов**:
       - Когда добавление информации из других рядов может улучшить прогноз
       - В сочетании с преобразованиями (разности, логарифмирование) для достижения стационарности
       - Для более точного моделирования краткосрочных зависимостей
       - Когда между рядами наблюдается причинно-следственная связь по Грейнджеру
    
    3. **Не рекомендуется для плохо прогнозируемых рядов**:
       - VAR требует стационарность данных, которая часто отсутствует у плохо прогнозируемых рядов
       - Объединение хаотичных рядов может только усилить шум
       - Высокий риск переобучения на нестабильных данных
       - Высокая сложность модели не оправдана при низком потенциале прогнозирования
    
    #### Требования для применения VAR:
    - Стационарность рядов (или возможность их преобразования к стационарному виду)
    - Одинаковая длина всех рядов
    - Отсутствие пропущенных значений
    - Одинаковая частота данных
    - Разумное количество параметров (рекомендуется не более 5-7 переменных)
    """)

# Информация о приложении
st.sidebar.markdown("---")
st.sidebar.info("""
### О приложении
Разработано для автоматического определения прогнозируемости временных рядов.

**Версия 2.0** - Расширенный анализ с дополнительными метриками и визуализациями.
""")
