import yfinance as yf
import twstock
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
from geopy.geocoders import Nominatim
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from ta.momentum import RSIIndicator

#美股區

#大盤指數
@st.cache_data
def plot_index(period,time):
    time = time
    # Fetch historical data for S&P 500
    nasdaq_data = yf.download('^IXIC', period=period)
    nasdaq_100_data = yf.download('^NDX', period=period)
    sp500_data = yf.download('^GSPC', period=period)
    dji_data = yf.download('^DJI', period=period)
    brk_data = yf.download('BRK-A', period=period)
    Russell_2000_data = yf.download('^RUT', period=period)    
    # Extract Close prices
    nasdaq_close = nasdaq_data['Close']
    nasdaq_100_close = nasdaq_100_data['Close']
    sp500_close = sp500_data['Close']  
    dji_close = dji_data['Close']
    brk_close = brk_data['Close']
    Russell_2000_close = Russell_2000_data['Close']   
    st.subheader(f'美股大盤＆中小企業{time}走勢')
    # Create Plotly subplot figure
    fig = make_subplots(rows=3, cols=2, subplot_titles=("NASDAQ", "NASDAQ-100", "S&P 500", "DJIA", "Berkshire Hathaway Inc.", "Russell-2000"))
    # Add traces for Log Close price
    fig.add_trace(go.Scatter(x=nasdaq_close.index, y=nasdaq_close.values, mode='lines', name='NASDAQ'), row=1, col=1)
    fig.add_trace(go.Scatter(x=nasdaq_100_close.index, y=nasdaq_100_close.values, mode='lines', name='NASDAQ-100'), row=1, col=2)
    fig.add_trace(go.Scatter(x=sp500_close.index, y=sp500_close.values, mode='lines', name='S&P 500'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dji_close.index, y=dji_close.values, mode='lines', name='DJIA'), row=2, col=2)
    fig.add_trace(go.Scatter(x=brk_close.index, y=brk_close.values, mode='lines', name='Berkshire Hathaway Inc.'), row=3, col=1)
    fig.add_trace(go.Scatter(x=Russell_2000_close.index, y=Russell_2000_close.values, mode='lines', name='Russell-2000'), row=3, col=2)
    # Update layout
    fig.update_layout(height=800, width=1000,showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_pct(period,time):
    time = time
    # Fetch historical data for S&P 500
    nasdaq_data = yf.download('^IXIC',period=period)
    nasdaq_100_data = yf.download('^NDX',period=period)
    sp500_data = yf.download('^GSPC', period=period)
    dji_data = yf.download('^DJI', period=period)
    brk_data = yf.download('BRK-A', period=period)
    Russell_2000_data = yf.download('^RUT', period=period)    
    # Extract Close prices
    nasdaq_close = nasdaq_data['Close']
    nasdaq_100_close = nasdaq_100_data['Close']
    sp500_close = sp500_data['Close']  
    dji_close = dji_data['Close']
    brk_close = brk_data['Close']
    Russell_2000_close = Russell_2000_data['Close']   
    # Calculate total returns
    nasdaq_total_return = ((nasdaq_close.iloc[-1] - nasdaq_close.iloc[0]) / nasdaq_close.iloc[0]) * 100
    nasdaq_100_total_return = ((nasdaq_100_close.iloc[-1] - nasdaq_100_close.iloc[0]) / nasdaq_100_close.iloc[0]) * 100
    sp500_total_return = ((sp500_close.iloc[-1] - sp500_close.iloc[0]) / sp500_close.iloc[0]) * 100
    dji_total_return = ((dji_close.iloc[-1] - dji_close.iloc[0]) / dji_close.iloc[0]) * 100
    brk_total_return = ((brk_close.iloc[-1] - brk_close.iloc[0]) / brk_close.iloc[0]) * 100
    Russell_2000_total_return = ((Russell_2000_close.iloc[-1] - Russell_2000_close.iloc[0]) / Russell_2000_close.iloc[0]) * 100
    # Create Plotly figure
    fig = go.Figure()   
    # Create a dictionary to store the results
    returns_dict = {
        'NASDAQ': nasdaq_total_return,
        'NASDAQ-100': nasdaq_100_total_return,
        'S&P 500': sp500_total_return,
        'DJIA': dji_total_return,
        'Berkshire Hathaway Inc.': brk_total_return,
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

@st.cache_data
def plot_foreign(period,time):
    time = time
    # Fetch historical data for S&P 500
    sp500_data = yf.download('^GSPC', period=period)
    nasdaq_data = yf.download('^IXIC', period=period)
    sha_data = yf.download('000001.SS', period=period)
    shz_data = yf.download('399001.SZ', period=period)
    twse_data = yf.download('^TWII', period=period)
    jp_data = yf.download('^N225', period=period)   
    # Extract Close prices
    sp500_close = sp500_data['Close']
    nasdaq_close = nasdaq_data['Close']
    sha_close = sha_data['Close']*0.1382
    shz_close = shz_data['Close']*0.1382
    twse_close = twse_data['Close']*0.0308 
    jp_close = jp_data['Close']*0.0064  
    st.subheader(f'美股大盤＆海外大盤{time}走勢')
    # Create Plotly subplot figure
    fig = make_subplots(rows=3, cols=2, subplot_titles=("S&P 500", "NASDAQ", "上證指數", "深證指數", "加權指數","日經指數"))
    # Add traces for Log Close price
    fig.add_trace(go.Scatter(x=sp500_close.index, y=sp500_close.values, mode='lines', name='S&P 500'), row=1, col=1)
    fig.add_trace(go.Scatter(x=nasdaq_close.index, y=nasdaq_close.values, mode='lines', name='NASDAQ'), row=1, col=2)
    fig.add_trace(go.Scatter(x=sha_close.index, y=sha_close.values, mode='lines', name='上證指數'), row=2, col=1)
    fig.add_trace(go.Scatter(x=shz_close.index, y=shz_close.values, mode='lines', name='深證指數'), row=2, col=2)
    fig.add_trace(go.Scatter(x=twse_close.index, y=twse_close.values, mode='lines', name='加權指數'), row=3, col=1)
    fig.add_trace(go.Scatter(x=jp_close.index, y=jp_close.values, mode='lines', name='日經指數'), row=3, col=2)
    # Update layout
    fig.update_layout(height=800, width=1000,showlegend=False)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Close Price", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_pct_foreign(period,time):
    time = time
    # Fetch historical data for S&P 500
    sp500_data = yf.download('^GSPC', period=period)
    nasdaq_data = yf.download('^IXIC', period=period)
    sha_data = yf.download('000001.SS', period=period)
    shz_data = yf.download('399001.SZ', period=period)
    twse_data = yf.download('^TWII', period=period) 
    jp_data = yf.download('^N225', period=period)  
    # Extract Close prices
    sp500_close = sp500_data['Close'] 
    nasdaq_close = nasdaq_data['Close'] 
    sha_close = sha_data['Close'] * 0.1382
    shz_close = shz_data['Close'] * 0.1382
    twse_close = twse_data['Close'] * 0.0308
    jp_close = jp_data['Close']*0.0064
    # Calculate total returns
    sp500_total_return = ((sp500_close.iloc[-1] - sp500_close.iloc[0]) / sp500_close.iloc[0]) * 100
    nasdaq_total_return = ((nasdaq_close.iloc[-1] - nasdaq_close.iloc[0]) / nasdaq_close.iloc[0]) * 100
    sha_total_return = ((sha_close.iloc[-1] - sha_close.iloc[0]) / sha_close.iloc[0]) * 100
    shz_total_return = ((shz_close.iloc[-1] - shz_close.iloc[0]) / shz_close.iloc[0]) * 100
    twse_total_return = ((twse_close.iloc[-1] - twse_close.iloc[0]) / twse_close.iloc[0]) * 100
    jp_total_return = ((jp_close.iloc[-1] - jp_close.iloc[0]) / jp_close.iloc[0]) * 100
    # Create Plotly figure
    fig = go.Figure()   
    # Create a dictionary to store the results
    returns_dict = {
        'S&P 500': sp500_total_return,
        'NASDAQ': nasdaq_total_return, 
        '上證指數': sha_total_return,
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

#s&p 500 成分股
@st.cache_data
def sp500_dsymbol():
    # 從維基百科頁面獲取數據
    url = 'https://zh.wikipedia.org/wiki/S%26P_500成份股列表'
    response = res.get(url) 
    # 檢查請求是否成功
    if response.status_code == 200:
        # 解析 HTML 表格
        sp500 = pd.read_html(response.content, encoding='utf-8')    
        # 提取產業類別
        df = sp500[0]
        st.write(df)
        industries = df['全球行業分類標準部門'].value_counts()
        colors = px.colors.qualitative.Plotly
        # 繪製統計圖
        fig = go.Figure(data=[go.Bar(x=industries.index, y=industries.values, marker_color=colors)])
        fig.update_layout(xaxis_title='全球行業分類標準部門', yaxis_title='數量', width=600, height=300)   
        # 顯示統計圖
        st.write('S&P500產業統計')
        st.plotly_chart(fig) 
        # 顯示數據表
    else:
        st.error('無法獲取數據')

#nasdaq100成分股
@st.cache_data
def nasdaq_100symbol():
    # 從維基百科頁面獲取數據
    url = 'https://zh.wikipedia.org/wiki/納斯達克100指數'
    response = res.get(url) 
    # 檢查請求是否成功
    if response.status_code == 200:
        # 解析 HTML 表格
        nas100 = pd.read_html(response.content, encoding='utf-8')    
        # 提取產業類別
        df = nas100[2]
        st.write(df)
        industries = df['全球行業分類標準部門'].value_counts()
        colors = px.colors.qualitative.Plotly
        # 繪製統計圖
        fig = go.Figure(data=[go.Bar(x=industries.index, y=industries.values, marker_color=colors)])
        fig.update_layout(xaxis_title='全球行業分類標準部門', yaxis_title='數量', width=600, height=300)   
        # 顯示統計圖
        st.write('NASDAQ-100產業統計')
        st.plotly_chart(fig) 
        # 顯示數據表
    else:
        st.error('無法獲取數據')

#dji成分股
def dji_symbol():
    url = res.get('https://zh.wikipedia.org/zh-tw/道琼斯工业平均指数')
    dji = pd.read_html(url.content, encoding='utf-8')
    st.write(dji[2])

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

#相關新聞
@st.cache_data
def get_stock_news(symbol):
    url = f"https://finviz.com/quote.ashx?t={symbol}&p=d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = res.get(url, headers=headers)
        response.raise_for_status()  # 确保请求成功
    except res.exceptions.RequestException as e:
        st.error(f"無法獲取{symbol}相關消息:{e}")
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    # 查找所有新闻项
    news_table = soup.find('table', class_='fullview-news-outer')
    if news_table is None:
        st.error(f"無法獲取{symbol}相關新聞表格")
        return None
    news_items = news_table.find_all('tr')
    news_data = []
    for news_item in news_items:
        news_link = news_item.find('a', class_='tab-link-news')
        if news_link:
            news_title = news_link.text
            news_text = news_link.get_text(strip=True)
            news_url = news_link['href']
            news_data.append({'Title': news_title, 'URL': news_url})
    return news_data


# 台股區

@st.cache_data
def plot_index_tw(period,time):
    # Fetch historical data for S&P 500
    time = time
    twse_data = yf.download('^TWII', period=period)
    tpex_data = yf.download('^TWOII', period=period)
    tw50_data = yf.download('0050.TW', period=period)   
    # Extract Close prices
    twse_close = twse_data['Close']
    tpex_close = tpex_data['Close']
    tw50_close = tw50_data['Close']
    # Take the logarithm of the Close prices
    st.subheader(f'上市＆櫃檯&0050{time}走勢')
    # Create Plotly figure
    # Add trace for Log Close price
    fig = make_subplots(rows=3, cols=1, subplot_titles=('加權指數','櫃檯指數','0050'))
    fig.add_trace(go.Scatter(x=twse_close.index, y=twse_close.values, mode='lines', name='加權指數'),row=1,col=1)
    fig.add_trace(go.Scatter(x=tpex_close.index, y=tpex_close.values, mode='lines', name='櫃檯指數'),row=2,col=1)
    fig.add_trace(go.Scatter(x=tw50_close.index, y=tw50_close.values, mode='lines', name='0050'),row=3,col=1)
    # Update layout
    fig.update_layout(height=800, width=1000,showlegend=False)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Close Price", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_tw_asia(period,time):
    # Fetch historical data for S&P 500
    time = time
    sha_data = yf.download('000001.SS',period=period )
    shz_data = yf.download('399001.SZ', period=period)
    twse_data = yf.download('^TWII', period=period)
    jp_data = yf.download('^N225', period=period)
    hk_data = yf.download('^HSI', period=period)
    kr_data = yf.download('^KS11', period=period)
    sin_data = yf.download('^STI', period=period)
    th_data = yf.download('^SET.BK', period=period)
    # Extract Close prices
    sha_close = sha_data['Close'] * 4.4927  # 將上證指數轉換為新台幣
    shz_close = shz_data['Close'] * 4.4927  # 將上證指數轉換為新台幣
    twse_close = twse_data['Close']
    jp_close = jp_data['Close'] * 0.2084    # 將日經指數轉換為新台幣
    hk_close = hk_data['Close'] * 4.1549    # 將恒生指數轉換為新台幣
    kr_close = kr_data['Close'] * 0.0237    # 將韓國綜合股價指數轉換為新台幣
    sin_close = sin_data['Close'] * 23.9665 # 將新加坡海峽時報指數轉換為新台幣
    th_close = th_data['Close'] * 0.8842
    # Take the logarithm of the Close prices 
    st.subheader(f'台股大盤＆亞洲大盤{time}走勢')
    # Create Plotly figure
    # Add trace for Log Close price
    fig = make_subplots(rows=4, cols=2, subplot_titles=("上證指數", "深證指數", "加權指數", "日經指數", "恒生指數","韓國綜合股價指數","新加坡海峽時報指數","泰國SET指數"))
    fig.add_trace(go.Scatter(x=sha_close.index, y=sha_close.values, mode='lines', name='上證指數'),row=1,col=1)
    fig.add_trace(go.Scatter(x=shz_close.index, y=shz_close.values, mode='lines', name='深證指數'),row=1,col=2)
    fig.add_trace(go.Scatter(x=twse_close.index, y=twse_close.values, mode='lines', name='加權指數'),row=2,col=1)
    fig.add_trace(go.Scatter(x=jp_close.index, y=jp_close.values, mode='lines', name='日經指數'),row=2,col=2)
    fig.add_trace(go.Scatter(x=hk_close.index, y=hk_close.values, mode='lines', name='恒生指數'),row=3,col=1)
    fig.add_trace(go.Scatter(x=kr_close.index, y=kr_close.values, mode='lines', name='韓國綜合股價指數'),row=3,col=2)
    fig.add_trace(go.Scatter(x=sin_close.index, y=sin_close.values, mode='lines', name='新加坡海峽時報指數'),row=4,col=1)
    fig.add_trace(go.Scatter(x=th_close.index, y=th_close.values, mode='lines', name='泰國SET指數'),row=4,col=2)
    # Update layout
    fig.update_layout(height=800, width=1000,showlegend=False)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Close Price", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_pct_tw(period,time):
    # Fetch historical data for S&P 500
    time = time
    twse_data = yf.download('^TWII',period=period )
    sha_data = yf.download('000001.SS', period=period)
    shz_data = yf.download('399001.SZ', period=period)
    jp_data = yf.download('^N225', period=period)
    hk_data = yf.download('^HSI', period=period)
    kr_data = yf.download('^KS11',period=period)
    sin_data = yf.download('^STI', period=period)
    th_data = yf.download('^SET.BK', period=period)
    # Extract Close prices
    sha_close = sha_data['Close'] * 4.4927  # 將上證指數轉換為新台幣
    shz_close = shz_data['Close'] * 4.4927  # 將上證指數轉換為新台幣
    twse_close = twse_data['Close']
    jp_close = jp_data['Close'] * 0.2084    # 將日經指數轉換為新台幣
    hk_close = hk_data['Close'] * 4.1549    # 將恒生指數轉換為新台幣
    kr_close = kr_data['Close'] * 0.0237    # 將韓國綜合股價指數轉換為新台幣
    sin_close = sin_data['Close'] * 23.9665 # 將新加坡海峽時報指數轉換為新台幣
    th_close = th_data['Close'] * 0.8842
    # Calculate total returns
    twse_total_return = ((twse_close.iloc[-1] - twse_close.iloc[0]) / twse_close.iloc[0]) * 100
    shz_total_return = ((shz_close.iloc[-1] - shz_close.iloc[0]) / shz_close.iloc[0]) * 100
    sha_total_return = ((sha_close.iloc[-1] - sha_close.iloc[0]) / sha_close.iloc[0]) * 100
    jp_total_return = ((jp_close.iloc[-1] - jp_close.iloc[0]) / jp_close.iloc[0]) * 100
    hk_total_return = ((hk_close.iloc[-1] - hk_close.iloc[0]) / hk_close.iloc[0]) * 100
    kr_total_return = ((kr_close.iloc[-1] - kr_close.iloc[0]) / kr_close.iloc[0]) * 100
    sin_total_return = ((sin_close.iloc[-1] - sin_close.iloc[0]) / sin_close.iloc[0]) * 100
    th_total_return = ((th_close.iloc[-1] - th_close.iloc[0]) / th_close.iloc[0]) * 100
    # Create Plotly figure
    fig = go.Figure()   
    # Create a dictionary to store the results
    returns_dict = {
        '上證指數': sha_total_return,
        '深證指數': shz_total_return,
        '恒生指數': hk_total_return,
        '韓國綜合股價指數': kr_total_return,
        '新加坡海峽時報指數': sin_total_return,
        '日經指數': jp_total_return,
        '加權指數': twse_total_return,
        '泰國SET指數':th_total_return
    }
    # Sort the dictionary by values in descending order
    sorted_returns = dict(sorted(returns_dict.items(), key=lambda item: item[1], reverse=True))
    colors = px.colors.qualitative.Plotly
    # Add traces for Total Returns
    fig.add_trace(go.Bar(x=list(sorted_returns.keys()),
                         y=list(sorted_returns.values()),
                         marker_color=colors))
    # Update layout
    st.subheader(f'台股大盤＆亞洲大盤{time}報酬率％')
    fig.update_layout(yaxis_title='Total Return (%)')
    st.plotly_chart(fig, use_container_width=True)

#sti成分股
@st.cache_data
def sti_symbol():
    # 從維基百科頁面獲取數據
    url = 'https://tw.tradingview.com/symbols/TVC-STI/components/'
    response = res.get(url) 
    # 檢查請求是否成功
    if response.status_code == 200:
        # 解析 HTML 表格
        sti = pd.read_html(response.content, encoding='utf-8')    
        # 提取產業類別
        df = sti[0]
        st.write(df)
        industries = df['部門'].value_counts()
        colors = px.colors.qualitative.Plotly
        # 繪製統計圖
        fig = go.Figure(data=[go.Bar(x=industries.index, y=industries.values, marker_color=colors)])
        fig.update_layout(xaxis_title='部門', yaxis_title='數量', width=600, height=300)   
        # 顯示統計圖
        st.write('新加坡海峽指數產業統計')
        st.plotly_chart(fig) 
        # 顯示數據表
    else:
        st.error('無法獲取數據')

@st.cache_data
def hsi_symbol():
    # 從維基百科頁面獲取數據
    url = 'https://tw.tradingview.com/symbols/HSI-HSI/components/'
    response = res.get(url) 
    # 檢查請求是否成功
    if response.status_code == 200:
        # 解析 HTML 表格
        hsi = pd.read_html(response.content, encoding='utf-8')    
        # 提取產業類別
        df = hsi[0]
        st.write(df)
        industries = df['部門'].value_counts()
        colors = px.colors.qualitative.Plotly
        # 繪製統計圖
        fig = go.Figure(data=[go.Bar(x=industries.index, y=industries.values, marker_color=colors)])
        fig.update_layout(xaxis_title='部門', yaxis_title='數量', width=600, height=300)   
        # 顯示統計圖
        st.write('恒生指數產業統計')
        st.plotly_chart(fig) 
        # 顯示數據表
    else:
        st.error('無法獲取數據')

@st.cache_data
def n225_symbol():
    # 從維基百科頁面獲取數據
    url = 'https://zh.wikipedia.org/zh-tw/日经平均指数'
    response = res.get(url) 
    # 檢查請求是否成功
    if response.status_code == 200:
        # 解析 HTML 表格
        n225 = pd.read_html(response.content, encoding='utf-8')    
        # 提取產業類別
        df = n225[4]
        st.write(df)
        industries = df['行業'].value_counts()
        colors = px.colors.qualitative.Plotly
        # 繪製統計圖
        fig = go.Figure(data=[go.Bar(x=industries.index, y=industries.values, marker_color=colors)])
        fig.update_layout(xaxis_title='行業', yaxis_title='數量', width=600, height=300)   
        # 顯示統計圖
        st.write('日經指數產業統計')
        st.plotly_chart(fig) 
        # 顯示數據表
    else:
        st.error('無法獲取數據')

#深證指數成分股
def shz_symbol():
    # 從維基百科頁面獲取數據
    url = 'https://zh.wikipedia.org/zh-tw/深圳证券交易所成份股价指数'
    response = res.get(url) 
    # 檢查請求是否成功
    if response.status_code == 200:
        # 解析 HTML 表格
        shz = pd.read_html(response.content, encoding='utf-8')    
        # 提取產業類別
        df = shz[1]
        st.write(df)
        industries = df['所屬行業'].value_counts()
        colors = px.colors.qualitative.Plotly
        # 繪製統計圖
        fig = go.Figure(data=[go.Bar(x=industries.index, y=industries.values, marker_color=colors)])
        fig.update_layout(xaxis_title='所屬行業', yaxis_title='數量', width=600, height=300)   
        # 顯示統計圖
        st.write('深證指數產業統計')
        st.plotly_chart(fig) 
        # 顯示數據表
    else:
        st.error('無法獲取數據')

# 定义函数以获取股票数据
def get_twstock_data(symbol,time_range):
    today = datetime.today()
    years_ago = today - timedelta(days=time_range*365)  # 粗略計算，不考慮閏年
    year = years_ago.year
    month = years_ago.month
    stock = twstock.Stock(symbol)
    stock_data = stock.fetch_from(year,month)
    stock_data = pd.DataFrame(stock_data)
    return stock_data

# 定义函数以获取股票数据
def get_twstock_month(symbol,time_range):
    today = datetime.today()
    months_ago_date = today - timedelta(days=time_range*30)  # 粗略计算一个月为30天
    year = months_ago_date.year
    month = months_ago_date.month
    stock = twstock.Stock(symbol)
    stock_data = stock.fetch_from(year,month)
    stock_data = pd.DataFrame(stock_data)
    return stock_data

# 计算价格差异的函数
def calculate_twse_difference(stock_data,period_days):
    latest_price = stock_data.iloc[-1]["close"]  # 获取最新的收盘价
    previous_price = stock_data.iloc[-period_days]["close"] if len(stock_data) > period_days else stock_data.iloc[0]["close"]  # 获取特定天数前的收盘价
    price_difference = latest_price - previous_price  # 计算价格差异
    percentage_difference = (price_difference / previous_price) * 100  # 计算百分比变化
    return price_difference, percentage_difference  # 返回价格差异和百分比变化

#streamlit版面配置
def app():
    st.set_page_config(page_title="StockInfo", layout="wide", page_icon="📈")
    hide_menu_style = "<style> footer {visibility: hidden;} </style>"
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: rainbow;'>📈 StockInfo</h1>", unsafe_allow_html=True)
    st.header(' ',divider="rainbow")
    st.sidebar.title('📈 Menu')
    market = st.sidebar.selectbox('選擇市場', ['美國','台灣'])
    options = st.sidebar.selectbox('選擇功能', ['大盤指數','交易數據','近期相關消息'])
    st.sidebar.markdown('''
    免責聲明：        
    1. K 線圖觀看角度      
            - 美股: 綠漲、紅跌        
            - 台股: 綠跌、紅漲           
    2. 本平台僅適用於數據搜尋，不建議任何投資行為
    3. 有些數據僅限美股，台股尚未支援  
    4. 排版問題建議使用電腦查詢數據  
    ''')

    if market == '美國' and options == '大盤指數':
        period = st.selectbox('選擇時長',['年初至今','1年','3年','5年','10年','全部'])
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
        elif period == '3年':
            period = '3y'
            time = '3年'
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
        with st.expander("顯示成份股"):
            st.write('S&P500成份股')
            sp500_dsymbol()
            st.write('NASDAQ100成份股')
            nasdaq_100symbol()
            st.write('道瓊工業成份股')
            dji_symbol()
        st.markdown("[美股指數名詞解釋](https://www.oanda.com/bvi-ft/lab-education/indices/us-4index/)")
    elif market == '美國' and options == '交易數據':
        with st.expander("展開輸入參數"):
            range = st.selectbox('長期/短期', ['長期', '短期'])
            if range == '長期':
                symbol = st.text_input("輸入美股代碼").upper()
                time_range = st.selectbox('選擇時長', ['1年', '3年', '5年', '10年', '全部'])
                if time_range == '1年':
                    period = '1y'
                    period_days = 252
                elif time_range == '3年':
                    period = '3y'
                    period_days = 252 * 3
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
                time_range = st.selectbox('選擇時長',['1個月','2個月','3個月','6個月'])
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
                    st.download_button(f"下載{symbol}-{time_range}數據", stock_data.to_csv(index=True), file_name=f"{symbol}-{time_range}.csv", mime="text/csv")

    elif market == '美國' and options == '近期相關消息':
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
                    st.subheader(f"{symbol}-近期相關消息連結")
                    # 打印所有新闻链接
                    for news in news_data:
                        st.write(f'**[{news["Title"]}]({news["URL"]})**')
                else:
                    st.write(f"查無{symbol}近期相關消息")

    elif market == '台灣' and options == '大盤指數':
        period = st.selectbox('選擇時長',['年初至今','1年','3年','5年','10年','全部'])
        if period == '年初至今':
            time = '年初至今'
            period = 'ytd'
            plot_index_tw(period,time)
            plot_tw_asia(period,time)
            plot_pct_tw(period,time)
        elif period == '1年':
            time = '1年'
            period = '1y'
            plot_index_tw(period,time)
            plot_tw_asia(period,time)
            plot_pct_tw(period,time)
        elif period == '3年':
            time = '3年'
            period = '3y'
            plot_index_tw(period,time)
            plot_tw_asia(period,time)
            plot_pct_tw(period,time)
        elif period == '5年':
            time = '5年'
            period = '5y'
            plot_index_tw(period,time)
            plot_tw_asia(period,time)
            plot_pct_tw(period,time)
        elif period == '10年':
            time = '10年'
            period = '10y'
            plot_index_tw(period,time)
            plot_tw_asia(period,time)
            plot_pct_tw(period,time)
        elif period == '全部':
            time = '全部'
            period = 'max'
            plot_index_tw(period,time)
            plot_tw_asia(period,time)
            plot_pct_tw(period,time)
    elif market == '台灣' and options == '交易數據':
        with st.expander("展開輸入參數"):
            range = st.selectbox('長期/短期', ['長期', '短期'])
            symbol = st.text_input("輸入台股代碼")
            if range == '長期':
                time_range = st.selectbox('選擇時長', ['1年', '3年', '5年', '10年'])
                if time_range == '1年':
                    time_range = 1
                    time = '1年'
                    period_days = 252
                elif time_range == '3年':
                    time_range = 3
                    time = '3年'
                    period_days = 252 * 3
                elif time_range == '5年':
                    time_range = 5
                    time = '5年'
                    period_days = 252 * 5
                elif time_range == '10年':
                    time_range = 10
                    time = '10年'
                    period_days = 252 * 10
            elif range == '短期':
                time_range = st.selectbox('選擇時長',['1個月','2個月','3個月','6個月'])
                if time_range == '1個月':
                    time_range = 1
                    time = '1個月'
                    period_days = 21
                elif time_range == '2個月':
                    time_range = 2
                    time = '2個月'
                    period_days = 42
                elif time_range == '3個月':
                    time_range = 3
                    time = '3個月'
                    period_days = 63
                elif time_range == '6個月':
                    time_range = 6
                    time = '6個月'
                    period_days = 126
        if st.button("查詢"):
            stock_data = None  # 初始化变量
            if symbol:
                if range == '長期':
                    stock_data = get_twstock_data(symbol, time_range)
                elif range == '短期':
                    stock_data = get_twstock_month(symbol, time_range)
                st.header(f"{symbol}-{time}交易數據")
                if stock_data is not None and not stock_data.empty:
                    if period_days is None:
                        period_days = len(stock_data)
                    price_difference, percentage_difference = calculate_twse_difference(stock_data,period_days)
                    latest_close_price = stock_data.iloc[-1]["close"]
                    highest_price = stock_data["high"].max()
                    lowest_price = stock_data["low"].min()
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("最新收盤價", f"${latest_close_price:.2f}")
                    with col2:
                        st.metric(f"{time}增長率", f"${price_difference:.2f}", f"{percentage_difference:+.2f}%")
                    with col3:
                        st.metric(f"{time}最高價", f"${highest_price:.2f}")
                    with col4:
                        st.metric(f"{time}最低價", f"${lowest_price:.2f}")
                    st.subheader(f"{symbol}-{time}K線圖表")
                    fig = go.Figure()
                    fig = plotly.subplots.make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.8, 0.5, 0.5, 0.5])
                    mav5 = stock_data['close'].rolling(window=5).mean()
                    mav20 = stock_data['close'].rolling(window=20).mean()
                    mav60 = stock_data['close'].rolling(window=60).mean()
                    rsi = RSIIndicator(close=stock_data['close'], window=14)
                    macd = MACD(close=stock_data['close'], window_slow=26, window_fast=12, window_sign=9)
                    fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['open'], high=stock_data['high'],low=stock_data['low'], close=stock_data['close'],increasing_line_color= 'red', decreasing_line_color='green'), row=1, col=1)
                    fig.update_layout(xaxis_rangeslider_visible=False)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=mav5, opacity=0.7, line=dict(color='blue', width=2), name='MAV-5'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=mav20, opacity=0.7, line=dict(color='orange', width=2), name='MAV-20'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=mav60, opacity=0.7, line=dict(color='purple', width=2), name='MAV-60'), row=1, col=1)
                    colors = ['red' if row['open'] - row['close'] >= 0 else 'green' for index, row in stock_data.iterrows()]
                    fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['transaction'], marker_color=colors, name='Volume'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=rsi.rsi(), line=dict(color='purple', width=2)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=[70]*len(stock_data.index), line=dict(color='green', width=1), name='Overbought'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=[30]*len(stock_data.index), line=dict(color='red', width=1), name='Oversold'), row=3, col=1)
                    colorsM = ['red' if val >= 0 else 'green' for val in macd.macd_diff()]
                    fig.add_trace(go.Bar(x=stock_data.index, y=macd.macd_diff(), marker_color=colorsM), row=4, col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=macd.macd(), line=dict(color='orange', width=2)), row=4, col=1)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=macd.macd_signal(), line=dict(color='blue', width=1)), row=4, col=1)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                    fig.update_yaxes(title_text="RSI", row=3, col=1)
                    fig.update_yaxes(title_text="MACD", row=4, col=1)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f'查無{symbol}數據')
                with st.expander(f'展開{symbol}-{time}數據'):
                    st.dataframe(stock_data)
                    st.download_button(f"下載{symbol}-{time}數據", stock_data.to_csv(index=True), file_name=f"{symbol}-{time}.csv", mime="text/csv")

if __name__ == "__main__":
    app()
