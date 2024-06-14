from posixpath import expanduser
import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import requests as res
import io
import folium
import pytz
import geopy
import re
import numpy as np
import time
import plotly
from bs4 import BeautifulSoup
from plotly.subplots import make_subplots
from datetime import datetime,timedelta
from streamlit_folium import folium_static
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from ta.momentum import RSIIndicator

#美股區
def plot_index(period,time):
    time = time
    # Fetch historical data for S&P 500
    nasdaq_data = yf.download('^IXIC', period=period)
    nasdaq_100_data = yf.download('^NDX', period=period)
    sp500_data = yf.download('^GSPC', period=period)
    dji_data = yf.download('^DJI', period=period)
    sox_data = yf.download('^SOX', period=period)
    Russell_2000_data = yf.download('^RUT', period=period)    
    # Extract Close prices
    nasdaq_close = nasdaq_data['Close']
    nasdaq_100_close = nasdaq_100_data['Close']
    sp500_close = sp500_data['Close']  
    dji_close = dji_data['Close']
    sox_close = sox_data['Close']
    Russell_2000_close = Russell_2000_data['Close']   
    st.subheader(f'美股大盤＆中小企業{time}走勢')
    # Create Plotly subplot figure
    fig = make_subplots(rows=3, cols=2, subplot_titles=("NASDAQ", "NASDAQ-100", "S&P 500", "DJIA", "Berkshire Hathaway Inc.", "Russell-2000"))
    # Add traces for Log Close price
    fig.add_trace(go.Scatter(x=nasdaq_close.index, y=nasdaq_close.values, mode='lines', name='NASDAQ'), row=1, col=1)
    fig.add_trace(go.Scatter(x=nasdaq_100_close.index, y=nasdaq_100_close.values, mode='lines', name='NASDAQ-100'), row=1, col=2)
    fig.add_trace(go.Scatter(x=sp500_close.index, y=sp500_close.values, mode='lines', name='S&P 500'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dji_close.index, y=dji_close.values, mode='lines', name='DJIA'), row=2, col=2)
    fig.add_trace(go.Scatter(x=sox_close.index, y=sox_close.values, mode='lines', name='美國費城半導體指數'), row=3, col=1)
    fig.add_trace(go.Scatter(x=Russell_2000_close.index, y=Russell_2000_close.values, mode='lines', name='Russell-2000'), row=3, col=2)
    # Update layout
    fig.update_layout(height=800, width=1000,showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_pct(period, time):
    time = time
    # Fetch historical data for S&P 500
    nasdaq_data = yf.download('^IXIC', period=period)
    nasdaq_100_data = yf.download('^NDX', period=period)
    sp500_data = yf.download('^GSPC', period=period)
    dji_data = yf.download('^DJI', period=period)
    sox_data = yf.download('^SOX', period=period)
    Russell_2000_data = yf.download('^RUT', period=period)    
    # Extract Close prices
    nasdaq_close = nasdaq_data['Close']
    nasdaq_100_close = nasdaq_100_data['Close']
    sp500_close = sp500_data['Close']  
    dji_close = dji_data['Close']
    sox_close = sox_data['Close']
    Russell_2000_close = Russell_2000_data['Close']   
    
    # Check if any of the data frames are empty or do not have enough data
    if nasdaq_close.empty or len(nasdaq_close) < 2:
        st.warning(f"Nasdaq data is insufficient for the period: {period}")
        return
    if nasdaq_100_close.empty or len(nasdaq_100_close) < 2:
        st.warning(f"Nasdaq-100 data is insufficient for the period: {period}")
        return
    if sp500_close.empty or len(sp500_close) < 2:
        st.warning(f"S&P 500 data is insufficient for the period: {period}")
        return
    if dji_close.empty or len(dji_close) < 2:
        st.warning(f"DJI data is insufficient for the period: {period}")
        return
    if sox_close.empty or len(sox_close) < 2:
        st.warning(f"sox data is insufficient for the period: {period}")
        return
    if Russell_2000_close.empty or len(Russell_2000_close) < 2:
        st.warning(f"Russell-2000 data is insufficient for the period: {period}")
        return
    
    # Calculate total returns
    nasdaq_total_return = ((nasdaq_close.iloc[-1] - nasdaq_close.iloc[0]) / nasdaq_close.iloc[0]) * 100
    nasdaq_100_total_return = ((nasdaq_100_close.iloc[-1] - nasdaq_100_close.iloc[0]) / nasdaq_100_close.iloc[0]) * 100
    sp500_total_return = ((sp500_close.iloc[-1] - sp500_close.iloc[0]) / sp500_close.iloc[0]) * 100
    dji_total_return = ((dji_close.iloc[-1] - dji_close.iloc[0]) / dji_close.iloc[0]) * 100
    sox_total_return = ((sox_close.iloc[-1] - sox_close.iloc[0]) / sox_close.iloc[0]) * 100
    Russell_2000_total_return = ((Russell_2000_close.iloc[-1] - Russell_2000_close.iloc[0]) / Russell_2000_close.iloc[0]) * 100
    
    # Create Plotly figure
    fig = go.Figure()   
    # Create a dictionary to store the results
    returns_dict = {
        'NASDAQ': nasdaq_total_return,
        'NASDAQ-100': nasdaq_100_total_return,
        'S&P 500': sp500_total_return,
        'DJIA': dji_total_return,
        '美國費城半導體指數': sox_total_return,
        'Russell-2000': Russell_2000_total_return
    }
    colors = px.colors.qualitative.Plotly
    # Sort the dictionary by values in descending order
    sorted_returns = dict(sorted(returns_dict.items(), key=lambda item: item[1], reverse=True))
    # Add traces for Total Returns
    fig.add_trace(go.Bar(x=list(sorted_returns.keys()),
                         y=list(sorted_returns.values()),
                         marker_color=colors))
    # Update layout
    st.subheader(f'美股大盤＆中小企業市場{time}報酬率％')
    fig.update_layout(yaxis_title='Total Return (%)')
    st.plotly_chart(fig, use_container_width=True)

def plot_foreign(period,time):
    time = time
    # Fetch historical data for S&P 500
    sp500_data = yf.download('^GSPC', period=period)
    nasdaq_data = yf.download('^IXIC', period=period)
    hsi_data = yf.download('^HSI', period=period)
    shz_data = yf.download('399001.SZ', period=period)
    twse_data = yf.download('^TWII', period=period)
    jp_data = yf.download('^N225', period=period)   
    # Extract Close prices
    sp500_close = sp500_data['Close']
    nasdaq_close = nasdaq_data['Close']
    hsi_close = hsi_data['Close']*0.1382
    shz_close = shz_data['Close']*0.1382
    twse_close = twse_data['Close']*0.0308 
    jp_close = jp_data['Close']*0.0064  
    st.subheader(f'美股大盤＆海外大盤{time}走勢')
    # Create Plotly subplot figure
    fig = make_subplots(rows=3, cols=2, subplot_titles=("S&P 500", "NASDAQ", "恆生指數", "深證指數", "加權指數","日經指數"))
    # Add traces for Log Close price
    fig.add_trace(go.Scatter(x=sp500_close.index, y=sp500_close.values, mode='lines', name='S&P 500'), row=1, col=1)
    fig.add_trace(go.Scatter(x=nasdaq_close.index, y=nasdaq_close.values, mode='lines', name='NASDAQ'), row=1, col=2)
    fig.add_trace(go.Scatter(x=hsi_close.index, y=hsi_close.values, mode='lines', name='恆生指數'), row=2, col=1)
    fig.add_trace(go.Scatter(x=shz_close.index, y=shz_close.values, mode='lines', name='深證指數'), row=2, col=2)
    fig.add_trace(go.Scatter(x=twse_close.index, y=twse_close.values, mode='lines', name='加權指數'), row=3, col=1)
    fig.add_trace(go.Scatter(x=jp_close.index, y=jp_close.values, mode='lines', name='日經指數'), row=3, col=2)
    # Update layout
    fig.update_layout(height=800, width=1000,showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_pct_foreign(period,time):
    time = time
    # Fetch historical data for S&P 500
    sp500_data = yf.download('^GSPC', period=period)
    nasdaq_data = yf.download('^IXIC', period=period)
    hsi_data = yf.download('^HSI', period=period)
    shz_data = yf.download('399001.SZ', period=period)
    twse_data = yf.download('^TWII', period=period) 
    jp_data = yf.download('^N225', period=period)  
    # Extract Close prices
    sp500_close = sp500_data['Close'] 
    nasdaq_close = nasdaq_data['Close'] 
    hsi_close = hsi_data['Close'] * 0.1382
    shz_close = shz_data['Close'] * 0.1382
    twse_close = twse_data['Close'] * 0.0308
    jp_close = jp_data['Close']*0.0064
    # Calculate total returns
    sp500_total_return = ((sp500_close.iloc[-1] - sp500_close.iloc[0]) / sp500_close.iloc[0]) * 100
    nasdaq_total_return = ((nasdaq_close.iloc[-1] - nasdaq_close.iloc[0]) / nasdaq_close.iloc[0]) * 100
    hsi_total_return = ((hsi_close.iloc[-1] - hsi_close.iloc[0]) / hsi_close.iloc[0]) * 100
    shz_total_return = ((shz_close.iloc[-1] - shz_close.iloc[0]) / shz_close.iloc[0]) * 100
    twse_total_return = ((twse_close.iloc[-1] - twse_close.iloc[0]) / twse_close.iloc[0]) * 100
    jp_total_return = ((jp_close.iloc[-1] - jp_close.iloc[0]) / jp_close.iloc[0]) * 100
    # Create Plotly figure
    fig = go.Figure()   
    # Create a dictionary to store the results
    returns_dict = {
        'S&P 500': sp500_total_return,
        'NASDAQ': nasdaq_total_return, 
        '恆生指數': hsi_total_return,
        '深證指數': shz_total_return,
        '加權指數': twse_total_return,
        '日經指數': jp_total_return
    }
    colors = px.colors.qualitative.Plotly
    # Sort the dictionary by values in descending order
    sorted_returns = dict(sorted(returns_dict.items(), key=lambda item: item[1], reverse=True))
    # Add traces for Total Returns
    fig.add_trace(go.Bar(x=list(sorted_returns.keys()),
                         y=list(sorted_returns.values()),
                         marker_color=colors))
    # Update layout
    st.subheader(f'美股大盤＆海外大盤{time}報酬率％')
    fig.update_layout(yaxis_title='Total Return (%)')
    st.plotly_chart(fig, use_container_width=True)

# 定义将字符串中的百分号去除并转换为小数的函数
def clean_and_round(value):
    if isinstance(value, str):
        return float(value.strip('%')) / 100
    return value

# 定義將交易量字串轉換為數字的函數
def convert_volume_string_to_numeric(volume_str):
    if 'M' in volume_str:
        return float(volume_str.replace('M', '')) * 1000000
    elif 'B' in volume_str:
        return float(volume_str.replace('B', '')) * 1000000000
    else:
        return float(volume_str)

# 统一的图表布局设置
def get_chart_layout():
    return {
        "height": 600,  # 设置统一高度
        "margin": {"l": 40, "r": 40, "t": 40, "b": 40}  # 设置统一的边距
    }

# 今日上漲
def gainers_stock():
    try:
        url = "https://finance.yahoo.com/gainers/"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = res.get(url, headers=headers)
        response.raise_for_status()
        # 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        if table:
            table_html = str(table)
            f = io.StringIO(table_html)
            df = pd.read_html(f)[0]
            # 清理和保留小数点
            df['% Change'] = df['% Change'].map(clean_and_round)
            # 去除无法转换为数字的行
            df = df.dropna(subset=['% Change'])
            # 根据 % Change 列的值降序排列数据
            df_sorted = df.sort_values(by='% Change', ascending=False).head(25)
            # 定义所有长条的统一颜色为绿色
            color = 'rgba(0,255,0,0.6)'  # 绿色
            # 绘制长条图
            fig = go.Figure(data=[go.Bar(x=df_sorted['Symbol'], y=df_sorted['% Change'], marker=dict(color=color))])
            fig.update_layout(xaxis_title='Symbol', yaxis_title='% Change', **get_chart_layout())
            st.subheader('今日上漲前25名')
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("展開數據"):
                st.write(df_sorted)
            return df_sorted
        else:
            st.error("未找到表格")
            return None
    except Exception as e:
        st.error(f"獲取發生錯誤：{str(e)}")
        return None

# 今日下跌
def loser_stock():
    try:
        url = "https://finance.yahoo.com/losers/"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = res.get(url, headers=headers)
        response.raise_for_status()
        # 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        if table:
            table_html = str(table)
            f = io.StringIO(table_html)
            df = pd.read_html(f)[0]
            # 清理和保留小数点
            df['% Change'] = df['% Change'].map(clean_and_round)
            # 去除无法转换为数字的行
            df = df.dropna(subset=['% Change'])
            # 根据 % Change 列的值降序排列数据
            df_sorted = df.sort_values(by='% Change', ascending=True).head(25)
            # 定义所有长条的统一颜色为红色
            color = 'rgba(255,0,0,0.6)'  # 红色
            # 绘制长条图
            fig = go.Figure(data=[go.Bar(x=df_sorted['Symbol'], y=df_sorted['% Change'], marker=dict(color=color))])
            fig.update_layout(xaxis_title='Symbol', yaxis_title='% Change', **get_chart_layout())
            st.subheader('今日下跌前25名')
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("展開數據"):
                st.write(df_sorted)
            return df_sorted
        else:
            st.error("未找到表格")
            return None
    except Exception as e:
        st.error(f"獲取發生錯誤：{str(e)}")
        return None

# 今日熱門
def hot_stock():
    try:
        url = "https://finance.yahoo.com/most-active/"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = res.get(url, headers=headers)
        response.raise_for_status()
        # 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        if table:
            table_html = str(table)
            f = io.StringIO(table_html)
            df = pd.read_html(f)[0]
            # 提取 Volume 列的值並轉換為數字
            df['Numeric Volume'] = df['Volume'].apply(convert_volume_string_to_numeric)
            # 根据 Volume 列的值降序排列数据
            df_sorted = df.sort_values(by='Numeric Volume', ascending=False).head(25)
            # 定义所有长条的统一颜色为蓝色
            color = 'rgba(0,0,255,0.6)'  # 蓝色
            # 绘制长条图
            fig = go.Figure(data=[go.Bar(x=df_sorted['Symbol'], y=df_sorted['Numeric Volume'], marker=dict(color=color))])
            fig.update_layout(xaxis_title='Symbol', yaxis_title='Volume', **get_chart_layout())
            st.subheader('今日交易量前25名')
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("展開數據"):
                st.write(df_sorted)
            return df_sorted
        else:
            st.error("未找到表格")
            return None
    except Exception as e:
        st.error(f"獲取發生錯誤：{str(e)}")
        return None

# Function to get stock statistics
def get_stock_statistics(symbol):
    url = f"https://finviz.com/quote.ashx?t={symbol}&p=d#statements"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = res.get(url, headers=headers)
        response.raise_for_status()
    except res.exceptions.RequestException as e:
        st.error(f"獲取 {symbol} 數據時出錯: {e}")
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', class_='snapshot-table2')
    if not table:
        st.error("頁面上未找到表格")
        return None
    rows = table.find_all('tr')
    data = {}
    for row in rows:
        cells = row.find_all('td')
        for i in range(0, len(cells), 2):
            key = cells[i].get_text(strip=True)
            value = cells[i + 1].get_text(strip=True)
            data[key] = value
    return data

# Function to process values
def process_value(value):
    if isinstance(value, str):
        value = value.replace(',', '')  # Remove commas for thousands
        if value.endswith('%'):
            return float(value[:-1])  # Convert percentage to float
        elif value.endswith('B'):
            return float(value[:-1]) * 1e9  # Convert billions to float
        elif value.endswith('M'):
            return float(value[:-1]) * 1e6  # Convert millions to float
        elif value.endswith('K'):
            return float(value[:-1]) * 1e3  # Convert thousands to float
        elif value.replace('.', '', 1).isdigit():  # Check if it's a numeric string
            return float(value)  # Convert numeric string to float
    return value  # Return the original value if no conversion is needed


# Function to categorize and plot
def categorize_and_plot(df, symbol):
    categories = {
        '估值指標': ['P/E', 'Forward P/E', 'PEG', 'P/S', 'P/B', 'P/C', 'P/FCF'],
        '盈利能力': ['Gross Margin', 'Oper. Margin', 'Profit Margin', 'ROA', 'ROE', 'ROI'],
        '表現指標': ['Perf Week', 'Perf Month', 'Perf Quarter', 'Perf Half Y', 'Perf Year', 'Perf YTD'],
        '流動性': ['Quick Ratio', 'Current Ratio'],
        '所有權': ['Insider Own', 'Inst Own', 'Shs Outstanding'],
        '銷售與收入': ['Sales', 'Income'],
        '簡單移動平均':['SMA20','SMA50','SMA200'],
        '其他': ['EPS (ttm)', 'EPS next Y', 'EPS next Q', 'Book/sh', 'Cash/sh', 'Dividend', 'Dividend %', 'Beta']
    }
    specs = [
        [{'type': 'xy'}, {'type': 'xy'}],
        [{'type': 'xy'}, {'type': 'xy'}],
        [{'type': 'domain'}, {'type': 'domain'}],
        [{'type': 'xy'}, {'type': 'xy'}]
    ]
    fig = make_subplots(rows=4, cols=2, subplot_titles=list(categories.keys()), specs=specs)
    plot_idx = 0
    for category, metrics in categories.items():
        plot_idx += 1
        row = (plot_idx - 1) // 2 + 1
        col = (plot_idx - 1) % 2 + 1
        cat_data = df[df['Metric'].isin(metrics)].copy()
        cat_data['Value'] = cat_data['Value'].apply(process_value)
        cat_data['Value'] = pd.to_numeric(cat_data['Value'], errors='coerce')  # Convert non-numeric values to NaN
        cat_data = cat_data.dropna(subset=['Value'])  # Drop rows with NaN values in 'Value'
        cat_data = cat_data.sort_values(by='Value', ascending=False)
        if category in ['所有權', '銷售與收入']:
            chart = go.Pie(labels=cat_data['Metric'], values=cat_data['Value'], name=category, sort=False)
        else:
            chart = go.Bar(x=cat_data['Metric'], y=cat_data['Value'], name=category, marker=dict(color=cat_data['Value'], colorscale='Viridis'))
        fig.add_trace(chart, row=row, col=col)
    fig.update_layout(height=1200, showlegend=True)
    st.subheader(f'{symbol}-基本資訊')
    st.plotly_chart(fig, use_container_width=True)

# 定义函数以获取股票数据
def get_stock_data(symbol,time_range):
    stock_data = yf.download(symbol,period=time_range)
    return stock_data

# 计算价格差异的函数
def calculate_price_difference(stock_data, period_days):
    latest_price = stock_data.iloc[-1]["Adj Close"]  # 获取最新的收盘价
    previous_price = stock_data.iloc[-period_days]["Adj Close"] if len(stock_data) > period_days else stock_data.iloc[0]["Adj Close"]  # 获取特定天数前的收盘价
    price_difference = latest_price - previous_price  # 计算价格差异
    percentage_difference = (price_difference / previous_price) * 100  # 计算百分比变化
    return price_difference, percentage_difference  # 返回价格差异和百分比变化

#機構評級
def scrape_and_plot_finviz_data(symbol):
    # 爬虫部分
    url = f"https://finviz.com/quote.ashx?t={symbol}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = res.get(url, headers=headers)
    # 检查请求是否成功
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from {url}, status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    # 定位包含分析师评级的表格
    table = soup.find('table', class_='js-table-ratings styled-table-new is-rounded is-small')
    # 检查是否成功找到表格
    if table is None:
        raise Exception("Failed to find the ratings table on the page.")
    
    # 从表格中提取数据
    data = []
    for row in table.find_all('tr')[1:]:  # 跳过表头
        cols = row.find_all('td')
        data.append({
            "Date": cols[0].text.strip(),
            "Action": cols[1].text.strip(),
            "Analyst": cols[2].text.strip(),
            "Rating Change": cols[3].text.strip(),
            "Price Target Change": cols[4].text.strip() if len(cols) > 4 else None
        })
    
    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)
    # 移除空的目标价格变化
    df = df.dropna(subset=['Price Target Change'])
    # 清理数据，替换特殊字符
    df['Price Target Change'] = df['Price Target Change'].str.replace('→', '->').str.replace(' ', '')
    # 将目标价格变化转换为数值范围
    price_change_ranges = df['Price Target Change'].str.extract(r'\$(\d+)->\$(\d+)')
    price_change_ranges = price_change_ranges.apply(pd.to_numeric)
    df['Price Target Start'] = price_change_ranges[0]
    df['Price Target End'] = price_change_ranges[1]
    
    # 动态生成评级变化的顺序
    rating_order = df['Rating Change'].unique().tolist()
    
    df['Rating Change'] = pd.Categorical(df['Rating Change'], categories=rating_order, ordered=True)
    
    # 绘图部分
    # 可视化 1：分析师的目标价格变化
    fig1 = go.Figure()
    for i, row in df.iterrows():
        fig1.add_trace(go.Scatter(
            x=[row['Price Target Start'], row['Price Target End']],
            y=[row['Analyst'], row['Analyst']],
            mode='lines+markers+text',
            line=dict(color='blue' if row['Price Target End'] >= row['Price Target Start'] else 'red', width=2),
            marker=dict(size=10, color='blue' if row['Price Target End'] >= row['Price Target Start'] else 'red'),
            text=[f"${row['Price Target Start']}", f"${row['Price Target End']}"],
            textposition="top center"
        ))
    
    fig1.update_layout(
        title='機構目標價格變化',
        xaxis_title='目標價格',
        yaxis_title='機構',
        yaxis=dict(type='category'),
        showlegend=False,
        height=800,  # 增加图表高度
        width=1200   # 增加图表宽度
    )

    # 按指定顺序对评级变化进行排序
    df_sorted = df.sort_values(by='Rating Change', ascending=True)

    # 可视化 2：评级变化的分布，使用不同颜色
    fig2 = px.histogram(df_sorted, x='Rating Change', title='機構評級變化分佈', color='Rating Change')
    fig2.update_layout(
        height=800,  # 增加图表高度
        width=1200   # 增加图表宽度
    )
    # 显示图表
    st.subheader(f'機構買賣{symbol}資訊')
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander(f'展開{symbol}機構評級數據'):
        st.write(df)

#相關新聞
def get_stock_news(symbol):
    url = f"https://finviz.com/quote.ashx?t={symbol}&p=d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = res.get(url, headers=headers)
        response.raise_for_status()  # Ensure the request was successful
    except res.exceptions.RequestException as e:
        st.error(f"無法獲取{symbol}相關消息: {e}")
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find all news items
    news_table = soup.find('table', class_='fullview-news-outer')
    if news_table is None:
        st.error(f"無法獲取{symbol}相關新聞表格")
        return None
    news_items = news_table.find_all('tr')
    news_data = []
    for news_item in news_items:
        cells = news_item.find_all('td')
        if len(cells) < 2:
            continue
        date_info = cells[0].text.strip()
        news_link = cells[1].find('a', class_='tab-link-news')
        if news_link:
            news_title = news_link.text.strip()
            news_url = news_link['href']
            news_data.append({'Date': date_info, 'Title': news_title, 'URL': news_url})
    return news_data

#streamlit版面配置
def app():
    st.set_page_config(page_title="StockInfo", layout="wide", page_icon="📈")
    hide_menu_style = "<style> footer {visibility: hidden;} </style>"
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: rainbow;'>📈 StockInfo</h1>", unsafe_allow_html=True)
    st.header(' ',divider="rainbow")
    st.sidebar.title('📈 Menu')
    options = st.sidebar.selectbox('選擇功能', ['大盤指數','今日熱門','公司基本資訊','交易數據','機構買賣','近期相關消息'])
    st.sidebar.markdown('''
    免責聲明：        
    1. K 線圖觀看角度      
            - 美股: 綠漲、紅跌        
            - 台股: 綠跌、紅漲           
    2. 本平台僅適用於數據搜尋，不建議任何投資行為
    3. 有些數據僅限美股，台股尚未支援  
    4. 排版問題建議使用電腦查詢數據  
    ''')

    if  options == '大盤指數':
        period = st.selectbox('選擇時長',['年初至今','1年','2年','5年','10年','全部'])
        if period == '年初至今':
            period = 'ytd'
            time = '年初至今'
            plot_index(period,time)
            plot_pct(period,time)
            plot_foreign(period,time)
            plot_pct_foreign(period,time)
        elif period == '1年':
            period = '1y'
            time = '1年'
            plot_index(period,time)
            plot_pct(period,time)
            plot_foreign(period,time)
            plot_pct_foreign(period,time)
        elif period == '2年':
            period = '2y'
            time = '2年'
            plot_index(period,time)
            plot_pct(period,time)
            plot_foreign(period,time)
            plot_pct_foreign(period,time)
        elif period == '5年':
            period = '5y'
            time = '5年'
            plot_index(period,time)
            plot_pct(period,time)
            plot_foreign(period,time)
            plot_pct_foreign(period,time)
        elif period == '10年':
            period = '10y'
            time = '10年'
            plot_index(period,time)
            plot_pct(period,time)
            plot_foreign(period,time)
            plot_pct_foreign(period,time)
        elif period == '全部':
            period = 'max'
            time = '全部'
            plot_index(period,time)
            plot_pct(period,time)
            plot_foreign(period,time)
            plot_pct_foreign(period,time)

    elif options == '今日熱門':
        gainers_stock()
        loser_stock()
        hot_stock()
        st.markdown("[資料來源](https://finance.yahoo.com)")

    elif  options == '公司基本資訊':
        symbol = st.text_input('輸入美股代號').upper()
        if st.button('查詢'):
            ticker = get_stock_statistics(symbol)
            if ticker:
                df = pd.DataFrame(list(ticker.items()), columns=['Metric', 'Value'])
                categorize_and_plot(df,symbol)
                with st.expander(f'展開{symbol}-基本資訊數據'):
                    st.write(df)
                st.markdown("[資料來源](https://finviz.com)")
                
    elif  options == '交易數據':
        with st.expander("展開輸入參數"):
            range = st.selectbox('長期/短期', ['長期', '短期'])
            if range == '長期':
                symbol = st.text_input("輸入美股代碼").upper()
                time_range = st.selectbox('選擇時長', ['1年', '2年', '5年', '10年', '全部'])
                if time_range == '1年':
                    period = '1y'
                    period_days = 252
                elif time_range == '2年':
                    period = '2y'
                    period_days = 252 * 2
                elif time_range == '5年':
                    period = '5y'
                    period_days = 252 * 5
                elif time_range == '10年':
                    period = '10y'
                    period_days = 252 * 10
                elif time_range == '全部':
                    period = 'max'
                    period_days = None  # 使用全部数据的长度

            elif range == '短期':
                symbol = st.text_input("輸入美股代碼").upper()
                time_range = st.selectbox('選擇時長',['1個月','3個月','6個月'])
                if time_range == '1個月':
                    period = '1mo'
                    period_days = 21  # 一个月大约是21个交易日
                elif time_range == '2個月':
                    period = '2mo'
                    period_days = 42
                elif time_range == '3個月':
                    period = '3mo'
                    period_days = 63  # 三个月大约是63个交易日
                elif time_range == '6個月':
                    period = '6mo'
                    period_days = 126  # 六个月大约是126个交易日
        if st.button("查詢"):
            if symbol:
                # 获取股票数据
                stock_data = get_stock_data(symbol, period)
                st.header(f"{symbol}-{time_range}交易數據")
                if stock_data is not None and not stock_data.empty:
                    if period_days is None:
                        period_days = len(stock_data)  # 更新 period_days 为 stock_data 的长度
                    price_difference, percentage_difference = calculate_price_difference(stock_data, period_days)
                    latest_close_price = stock_data.iloc[-1]["Adj Close"]
                    highest_price = stock_data["High"].max()
                    lowest_price = stock_data["Low"].min()
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("最新收盤價", f"${latest_close_price:.2f}")
                    with col2:
                        st.metric(f"{time_range}增長率", f"${price_difference:.2f}", f"{percentage_difference:+.2f}%")
                    with col3:
                        st.metric(f"{time_range}最高價", f"${highest_price:.2f}")
                    with col4:
                        st.metric(f"{time_range}最低價", f"${lowest_price:.2f}")
                    st.subheader(f"{symbol}-{time_range}K線圖表")
                    fig = go.Figure()
                    fig = plotly.subplots.make_subplots(rows=4, cols=1,shared_xaxes=True,vertical_spacing=0.01,row_heights=[0.8,0.5,0.5,0.5])
                    mav5 = stock_data['Adj Close'].rolling(window=5).mean()  # 5日mav
                    mav20 = stock_data['Adj Close'].rolling(window=20).mean()  # 20日mav
                    mav60 = stock_data['Adj Close'].rolling(window=60).mean()  # 60日mav
                    rsi = RSIIndicator(close=stock_data['Adj Close'], window=14)
                    macd = MACD(close=stock_data['Adj Close'],window_slow=26,window_fast=12, window_sign=9)
                    fig.add_trace(go.Candlestick(x=stock_data.index,open=stock_data['Open'],high=stock_data['High'],low=stock_data['Low'],close=stock_data['Adj Close'],),row=1,col=1)
                    fig.update_layout(xaxis_rangeslider_visible=False)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=mav5, opacity=0.7, line=dict(color='blue', width=2), name='MAV-5'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=mav20, opacity=0.7,line=dict(color='orange', width=2), name='MAV-20'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=mav60,  opacity=0.7, line=dict(color='purple', width=2),name='MAV-60'), row=1, col=1)
                    # Plot volume trace on 2nd row
                    colors = ['green' if row['Open'] - row['Adj Close'] >= 0 else 'red' for index, row in stock_data.iterrows()]
                    fig.add_trace(go.Bar(x=stock_data.index,y=stock_data['Volume'],marker_color=colors,name='Volume'),row=2, col=1)
                    # Plot RSI trace on 5th row
                    fig.add_trace(go.Scatter(x=stock_data.index,y=rsi.rsi(),line=dict(color='purple',width=2)),row=3,col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index,y=[70]*len(stock_data.index),line=dict(color='red', width=1),name='Overbought'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index,y=[30]*len(stock_data.index),line=dict(color='green', width=1),name='Oversold'), row=3, col=1)
                     # Plot MACD trace on 3rd row
                    colorsM = ['green' if val >= 0 else 'red' for val in macd.macd_diff()]
                    fig.add_trace(go.Bar(x=stock_data.index,y=macd.macd_diff(),marker_color=colorsM),row=4,col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index,y=macd.macd(),line=dict(color='orange', width=2)),row=4,col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index,y=macd.macd_signal(),line=dict(color='blue', width=1)),row=4,col=1)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                    fig.update_yaxes(title_text="RSI", row=3, col=1)
                    fig.update_yaxes(title_text="MACD", row=4, col=1)
                    st.plotly_chart(fig,use_container_width=True)
                else:
                    st.error(f'查無{symbol}數據')
                with st.expander(f'展開{symbol}-{time_range}數據'):
                    st.dataframe(stock_data)
    
    elif  options == '機構買賣':
        symbol = st.text_input('輸入美股代號').upper()
        if st.button('查詢'):
            scrape_and_plot_finviz_data(symbol)
            st.markdown("[資料來源](https://finviz.com)")

    elif  options == '近期相關消息':
        st.subheader('近期相關新聞')
        symbol = st.text_input('輸入美股代號').upper()
        if st.button('查詢'):
            if symbol:
                news_data = get_stock_news(symbol)
                if news_data:
                    # 将新闻数据转换为DataFrame
                    df = pd.DataFrame(news_data)
                    st.subheader(f"{symbol}-近期相關消息")
                    st.write(df)  # 显示表格
                    # 打印所有新闻链接
                    with st.expander(f'展開{symbol}-近期相關消息連結'):
                        for news in news_data:
                            st.write(f'**[{news["Title"]}]({news["URL"]})**')
                    st.markdown("[資料來源](https://finviz.com)")
                else:
                    st.write(f"查無{symbol}近期相關消息")



if __name__ == "__main__":
    app()
