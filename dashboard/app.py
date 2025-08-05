# 1. 라이브러리 임포트
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- 2. 데이터 및 Figure 준비 ---
PROJECT_ROOT = Path(__file__).parent.parent
SYMBOLS = ['NVDA', 'AMD', 'INTC', 'SMH']

# EDA 데이터 로드
dataframes = {}
for symbol in SYMBOLS:
    file_path = PROJECT_ROOT / 'data' / 'processed' / f'{symbol}_daily.csv'
    dataframes[symbol] = pd.read_csv(file_path, index_col=0, parse_dates=True)

# 예측 결과 데이터 로드
arima_df = pd.read_csv(PROJECT_ROOT / 'data/processed/arima_forecast_result.csv', index_col=0, parse_dates=True)
lstm_df = pd.read_csv(PROJECT_ROOT / 'data/processed/lstm_forecast_result.csv', index_col=0, parse_dates=True)

# --- Figure 생성 ---
# 탭 1 Figures
all_adj_close = pd.DataFrame({s: df['adj_close'] for s, df in dataframes.items()})
all_adj_close_since_2015 = all_adj_close[all_adj_close.index >= '2015-01-01']
normalized_prices = (all_adj_close_since_2015 / all_adj_close_since_2015.iloc[0]) * 100
fig_normalized = px.line(normalized_prices, title='Normalized Stock Performance')

# 탭 2 Figures
daily_returns = all_adj_close.pct_change().dropna()
rolling_volatility = daily_returns.rolling(window=30).std() * (252**0.5)
fig_volatility = px.line(rolling_volatility, title='30-Day Rolling Volatility (Risk)')
correlation_matrix = daily_returns.corr()
fig_heatmap = px.imshow(correlation_matrix, text_auto=True, title='Correlation Matrix of Daily Returns')

# 탭 3 Figures
fig_arima = go.Figure()
fig_arima.add_trace(go.Scatter(x=arima_df.index, y=arima_df['Historical Price'], mode='lines', name='Historical'))
fig_arima.add_trace(go.Scatter(x=arima_df.index, y=arima_df['Forecast'], mode='lines', name='ARIMA Forecast'))
fig_arima.add_trace(go.Scatter(x=arima_df.index, y=arima_df['Upper Confidence'], mode='lines', name='Upper Confidence', line=dict(width=0)))
fig_arima.add_trace(go.Scatter(x=arima_df.index, y=arima_df['Lower Confidence'], mode='lines', name='Lower Confidence', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)'))
fig_arima.update_layout(title='ARIMA Forecast (Next 30 Days)')

fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(x=lstm_df.index, y=lstm_df['Historical Price'], mode='lines', name='Train'))
fig_lstm.add_trace(go.Scatter(x=lstm_df.index, y=lstm_df['Validation (Actual)'], mode='lines', name='Actual'))
fig_lstm.add_trace(go.Scatter(x=lstm_df.index, y=lstm_df['LSTM Predictions'], mode='lines', name='Predicted'))
fig_lstm.update_layout(title='LSTM Model: Actual vs. Predicted')


# --- 3. Dash 앱 생성 및 레이아웃 정의 ---
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1('AI Semiconductor Stock Dashboard'),
    dcc.Tabs(id='tabs-container', value='tab-1-market', children=[
        dcc.Tab(label='Market Comparison', value='tab-1-market'),
        dcc.Tab(label='Risk & Correlation', value='tab-2-risk'),
        dcc.Tab(label='Price Forecast', value='tab-3-forecast'),
    ]),
    html.Div(id='tabs-content')
])

# --- 4. 콜백(Callback) 함수 정의 ---
@app.callback(Output('tabs-content', 'children'), Input('tabs-container', 'value'))
def render_content(tab):
    if tab == 'tab-1-market':
        return html.Div([dcc.Graph(figure=fig_normalized)])
    elif tab == 'tab-2-risk':
        return html.Div([dcc.Graph(figure=fig_volatility), dcc.Graph(figure=fig_heatmap)])
    elif tab == 'tab-3-forecast':
        return html.Div([dcc.Graph(figure=fig_arima), dcc.Graph(figure=fig_lstm)])

# --- 5. 앱 실행 ---
if __name__ == '__main__':
    app.run(debug=True)