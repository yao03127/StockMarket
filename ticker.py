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
from datetime import datetime
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

#美股區

#大盤指數
@st.cache_data
def plot_index(period):
    # Fetch historical data for S&P 500
    nasdaq_data = yf.download('^IXIC',period=period)
    nasdaq_100_data = yf.download('^NDX',period=period)
    sp500_data = yf.download('^GSPC',period=period)
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
    # Take the logarithm of the Close prices
    nasdaq_log_close = np.log(nasdaq_close)
    nasdaq_100_log_close = np.log(nasdaq_100_close)
    sp500_log_close = np.log(sp500_close)
    dji_log_close = np.log(dji_close)
    brk_log_close = np.log(brk_close)
    Russell_2000_log_close = np.log(Russell_2000_close)  
    st.subheader(f'美股大盤＆中小企業{period}走勢')
    # Create Plotly figure
    fig = go.Figure()   
    # Add trace for Log Close price
    fig.add_trace(go.Scatter(x=nasdaq_log_close.index, y=nasdaq_log_close.values, mode='lines', name='NASDAQ'))
    fig.add_trace(go.Scatter(x=nasdaq_100_log_close.index, y=nasdaq_100_log_close.values, mode='lines', name='NASDAQ-100'))
    fig.add_trace(go.Scatter(x=sp500_log_close.index, y=sp500_log_close.values, mode='lines', name='S&P 500'))
    fig.add_trace(go.Scatter(x=dji_log_close.index, y=dji_log_close.values, mode='lines', name='DJIA'))
    fig.add_trace(go.Scatter(x=brk_log_close.index, y=brk_log_close.values, mode='lines', name='Berkshire Hathaway Inc.'))
    fig.add_trace(go.Scatter(x=Russell_2000_log_close.index, y=Russell_2000_log_close.values, mode='lines', name='Russell-2000'))
    # Update layout
    fig.update_layout(xaxis_title='Date', yaxis_title='Log Close Price')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_pct(period):
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
    st.subheader(f'美股大盤＆中小企業市場{period}報酬率％')
    fig.update_layout(yaxis_title='Total Return (%)')
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_foreign(period):
    # Fetch historical data for S&P 500
    sp500_data = yf.download('^GSPC', period=period)
    nasdaq_data = yf.download('^IXIC', period=period)
    sha_data = yf.download('000001.SS', period=period)
    shz_data = yf.download('399001.SZ', period=period)
    twse_data = yf.download('^TWII', period=period)   
    # Extract Close prices
    sp500_close = sp500_data['Close']
    nasdaq_close = nasdaq_data['Close']
    sha_close = sha_data['Close']*0.1382
    shz_close = shz_data['Close']*0.1382
    twse_close = twse_data['Close']*0.0308  
    # Take the logarithm of the Close prices
    sp500_log_close = np.log(sp500_close)
    nasdaq_log_close = np.log(nasdaq_close)
    sha_log_close = np.log(sha_close)
    shz_log_close = np.log(shz_close)
    twse_log_close = np.log(twse_close)  
    st.subheader(f'美股大盤＆海外大盤{period}走勢')
    # Create Plotly figure
    fig = go.Figure()   
    # Add trace for Log Close price
    fig.add_trace(go.Scatter(x=sp500_log_close.index, y=sp500_log_close.values, mode='lines', name='S&P 500'))
    fig.add_trace(go.Scatter(x=nasdaq_log_close.index, y=nasdaq_log_close.values, mode='lines', name='NASDAQ'))
    fig.add_trace(go.Scatter(x=sha_log_close.index, y=sha_log_close.values, mode='lines', name='上證指數'))
    fig.add_trace(go.Scatter(x=shz_log_close.index, y=shz_log_close.values, mode='lines', name='深證指數'))
    fig.add_trace(go.Scatter(x=twse_log_close.index, y=twse_log_close.values, mode='lines', name='加權指數'))
    # Update layout
    fig.update_layout(xaxis_title='Date', yaxis_title='Log Close Price')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_pct_foreign(period):
    # Fetch historical data for S&P 500
    sp500_data = yf.download('^GSPC', period=period)
    nasdaq_data = yf.download('^IXIC', period=period)
    sha_data = yf.download('000001.SS', period=period)
    shz_data = yf.download('399001.SZ', period=period)
    twse_data = yf.download('^TWII', period=period)  
    # Extract Close prices
    sp500_close = sp500_data['Close'] 
    nasdaq_close = nasdaq_data['Close'] 
    sha_close = sha_data['Close'] * 0.1382
    shz_close = shz_data['Close'] * 0.1382
    twse_close = twse_data['Close'] * 0.0308  
    # Calculate total returns
    sp500_total_return = ((sp500_close.iloc[-1] - sp500_close.iloc[0]) / sp500_close.iloc[0]) * 100
    nasdaq_total_return = ((nasdaq_close.iloc[-1] - nasdaq_close.iloc[0]) / nasdaq_close.iloc[0]) * 100
    sha_total_return = ((sha_close.iloc[-1] - sha_close.iloc[0]) / sha_close.iloc[0]) * 100
    shz_total_return = ((shz_close.iloc[-1] - shz_close.iloc[0]) / shz_close.iloc[0]) * 100
    twse_total_return = ((twse_close.iloc[-1] - twse_close.iloc[0]) / twse_close.iloc[0]) * 100
    # Create Plotly figure
    fig = go.Figure()   
    # Create a dictionary to store the results
    returns_dict = {
        'S&P 500': sp500_total_return,
        'NASDAQ': nasdaq_total_return, 
        '上證指數': sha_total_return,
        '深證指數': shz_total_return,
        '加權指數': twse_total_return
    }
    colors = px.colors.qualitative.Plotly
    # Sort the dictionary by values in descending order
    sorted_returns = dict(sorted(returns_dict.items(), key=lambda item: item[1], reverse=True))
    # Add traces for Total Returns
    fig.add_trace(go.Bar(x=list(sorted_returns.keys()),
                         y=list(sorted_returns.values()),
                         marker_color=colors))
    # Update layout
    st.subheader(f'美股大盤＆海外大盤{period}報酬率％')
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

#今日熱門
@st.cache_data
def hot_stock():
    try:
        hot_stock_res = res.get("https://finance.yahoo.com/most-active/")
        f = io.StringIO(hot_stock_res.text)
        hot_stock_df_list = pd.read_html(f)
        hot_stock_df = hot_stock_df_list[0]
        hot_stock_df = hot_stock_df.drop(columns=['PE Ratio (TTM)', '52 Week Range'])
        st.subheader("今日交易量前25名")
        # 提取 Volume 列的值並轉換為數字
        hot_stock_df['Numeric Volume'] = hot_stock_df['Volume'].apply(convert_volume_string_to_numeric)
        symbols = hot_stock_df['Symbol']
        numeric_volumes = hot_stock_df['Numeric Volume']
        # 根据 Volume 列的值降序排列数据
        hot_stock_df_sorted = hot_stock_df.sort_values(by='Numeric Volume', ascending=False)
        symbols_sorted = hot_stock_df_sorted['Symbol']
        numeric_volumes_sorted = hot_stock_df_sorted['Numeric Volume']      
        # 定义所有长条的统一颜色为蓝色
        color = 'rgba(0,0,255,0.6)'  # 蓝色
        # 绘制长条图
        fig = go.Figure(data=[go.Bar(x=symbols_sorted, y=numeric_volumes_sorted, marker=dict(color=color))])
        fig.update_layout(xaxis_title='Symbol', yaxis_title='Volume')
        st.plotly_chart(fig)
        with st.expander("展開數據"):
            st.write(hot_stock_df_sorted)
        return hot_stock_df_sorted
    except Exception as e:
        st.error(f"獲取發生錯誤：{str(e)}")
        return None
      
#今日上漲
@st.cache_data
def gainers_stock():
    try:
        gainers_stock_res = res.get("https://finance.yahoo.com/gainers")
        f = io.StringIO(gainers_stock_res.text)
        gainers_stock_df_list = pd.read_html(f)
        gainers_stock_df = gainers_stock_df_list[0]
        gainers_stock_df = gainers_stock_df.drop(columns=['PE Ratio (TTM)', '52 Week Range'])       
        # 清理和保留小数点
        gainers_stock_df['% Change'] = gainers_stock_df['% Change'].map(clean_and_round)
        # 去除无法转换为数字的行
        gainers_stock_df = gainers_stock_df.dropna(subset=['% Change'])
        # 根据 % Change 列的值降序排列数据
        gainers_stock_df_sorted = gainers_stock_df.sort_values(by='% Change', ascending=False)
        st.subheader("今日上漲前25名")  
        # 定义所有长条的统一颜色为绿色
        color = 'rgba(0,255,0,0.6)'  # 绿色
        # 绘制长条图
        fig = go.Figure(data=[go.Bar(x=gainers_stock_df_sorted['Symbol'], y=gainers_stock_df_sorted['% Change'], marker=dict(color=color))])
        fig.update_layout(xaxis_title='Symbol', yaxis_title='% Change')
        st.plotly_chart(fig)
        with st.expander("展開數據"):
            st.write(gainers_stock_df_sorted)
        return gainers_stock_df_list
    except Exception as e:
        print(f"獲取發生錯誤：{str(e)}")
        return None
#今日下跌
@st.cache_data
def loser_stock():
    try:
        loser_stock_res = res.get("https://finance.yahoo.com/losers")
        f = io.StringIO(loser_stock_res.text)
        loser_stock_df_list = pd.read_html(f)
        loser_stock_df = loser_stock_df_list[0]
        loser_stock_df = loser_stock_df.drop(columns=['PE Ratio (TTM)', '52 Week Range'])       
        # 清理和保留小数点
        loser_stock_df['% Change'] = loser_stock_df['% Change'].map(clean_and_round)
        # 去除无法转换为数字的行
        loser_stock_df = loser_stock_df.dropna(subset=['% Change'])
        # 根据 % Change 列的值降序排列数据
        loser_stock_df_sorted = loser_stock_df.sort_values(by='% Change', ascending=True)
        st.subheader("今日下跌前25名")  
        # 定义所有长条的统一颜色为绿色
        color = 'rgba(255,0,0,0.6)'  # 深红色
        # 绘制长条图
        fig = go.Figure(data=[go.Bar(x=loser_stock_df_sorted['Symbol'], y=loser_stock_df_sorted['% Change'], marker=dict(color=color))])
        fig.update_layout(xaxis_title='Symbol', yaxis_title='% Change')
        st.plotly_chart(fig)
        with st.expander("展開數據"):
            st.write(loser_stock_df_sorted)
        return loser_stock_df_list
    except Exception as e:
        print(f"獲取發生錯誤：{str(e)}")
        return None

# 獲取公司基本資訊
@st.cache_data
def company_info(symbol):
    try:
        stock_info = yf.Ticker(symbol)
        com_info = stock_info.info
        return com_info
    except Exception as e:
        st.error(f"無法獲取{symbol}基本資訊：{str(e)}")
        return None

@st.cache_data    
def display_location(com_info):
    if 'city' in com_info and 'country' in com_info:
        city = com_info['city']
        country = com_info['country']

        # 使用 Nominatim 服务进行地理编码
        geolocator = Nominatim(user_agent="streamlit_app")
        location = geolocator.geocode(f"{city}, {country}")
        if location:
            # 使用 folium 创建地图，并将其定位到公司位置
            map = folium.Map(location=[location.latitude, location.longitude], zoom_start=10)
            # 添加标记
            folium.Marker([location.latitude, location.longitude], popup=f"{city}, {country}").add_to(map)
            # 使用 streamlit-folium 显示地图
            folium_static(map)
        else:
            st.error(f"無法找到{symbol}位置")

@st.cache_data
def display_info(com_info):
    if com_info:        
        selected_indicators = ['longName', 'country', 'city', 'marketCap', 'totalRevenue', 'grossMargins', 'operatingMargins',
                               'profitMargins', 'trailingEps', 'pegRatio', 'dividendRate', 'payoutRatio', 'bookValue',
                               'operatingCashflow', 'freeCashflow', 'returnOnEquity']

        selected_info = {indicator: com_info.get(indicator, '') for indicator in selected_indicators}
        #建立字典翻譯
        translation = {
            'longName': '公司名稱',
            'country': '國家',
            'city': '城市',
            'marketCap': '市值',
            'totalRevenue': '總收入',
            'grossMargins': '毛利率',
            'operatingMargins': '營業利潤率', 
            'profitMargins': '净利率',
            'trailingEps': '每股收益',
            'pegRatio': 'PEG 比率',
            'dividendRate': '股息率',
            'payoutRatio': '股息支付比例',
            'bookValue': '每股淨資產',
            'operatingCashflow': '營運現金流',
            'freeCashflow': '自由現金流',
            'returnOnEquity': '股東權益報酬率'
        }
        #Pandas DataFrame
        company_info = pd.DataFrame.from_dict(selected_info,orient='index',columns=['Value'])
        company_info.rename(index=translation,inplace=True)
        #轉換成百分比
        percent_columns = ['毛利率', '營業利潤率', '净利率', '股息率', '股息支付比例', '股東權益報酬率']
        for col in percent_columns:
            if col in company_info.index:
                company_info.at[col, 'Value'] = pd.to_numeric(company_info.at[col, 'Value'], errors='coerce')  # 将非数字转换为 NaN
                company_info.at[col, 'Value'] = f"{company_info.at[col, 'Value']:.2%}" if pd.notna(company_info.at[col, 'Value']) else None
        #千分位表示
        company_info['Value'] = company_info['Value'].apply(lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) and x >= 1000 else x)
        st.subheader(f"{symbol}-基本資訊")
        st.table(company_info)
        st.subheader(f"{symbol}-位置資訊")
        display_location(com_info)
    else:
        st.error(f"無法獲取{symbol}-基本訊息")

#財報-年度
@st.cache_data
def financial_statements(symbol):
    try:
        stock_info = yf.Ticker(symbol)
        balance_sheet = stock_info.balance_sheet
        income_statement = stock_info.income_stmt
        cash_flow = stock_info.cashflow
        return balance_sheet, income_statement, cash_flow
    except Exception as e:
        st.error(f"獲取{symbol}-財報發生錯誤：{str(e)}")
        return None, None, None

@st.cache_data
def balance(balance_sheet):
    if balance_sheet is not None:
        st.subheader(f"{symbol}-資產負債表(年度)")
        st.write(balance_sheet)

@st.cache_data
def income(income_statement):
    if income_statement is not None:
        st.subheader(f"{symbol}-綜合損益表(年度)")
        st.write(income_statement)

@st.cache_data
def cashflow(cash_flow):
    if cash_flow is not None:
        st.subheader(f"{symbol}-現金流量表(年度)")
        st.write(cash_flow)

#財報-季度
@st.cache_data
def financial_statements_quarterly(symbol):
    try:
        stock_info = yf.Ticker(symbol)
        balance_sheet_quarterly = stock_info.quarterly_balance_sheet
        income_statement_quarterly = stock_info.quarterly_income_stmt
        cash_flow_quarterly = stock_info.quarterly_cashflow  # 這裡修正了錯誤
        return balance_sheet_quarterly, income_statement_quarterly, cash_flow_quarterly
    except Exception as e:
        st.error(f"獲取{symbol}-財報發生錯誤：{str(e)}")
        return None, None, None

@st.cache_data
def balance_quarterly(balance_sheet_quarterly):
    if balance_sheet_quarterly is not None:
        st.subheader(f"{symbol}-資產負債表(季度)")
        st.write(balance_sheet_quarterly)

@st.cache_data
def income_quarterly(income_statement_quarterly):
    if income_statement_quarterly is not None:
        st.subheader(f"{symbol}-綜合損益表(季度)")
        st.write(income_statement_quarterly)

@st.cache_data
def cashflow_quarterly(cash_flow_quarterly):
    if cash_flow_quarterly is not None:
        st.subheader(f"{symbol}-現金流量表(季度)")
        st.write(cash_flow_quarterly)

#獲取歷史交易數據
@st.cache_data
def stock_data(symbol,start_date,end_date):
    try:
        stock_data = yf.download(symbol,start=start_date,end=end_date)
        st.subheader('交易數據')
        with st.expander("展開數據"):
            st.write(stock_data)
        return stock_data
    except Exception as e:
        st.error(f"無法獲取{symbol}-交易數據：{str(e)}")
        return None

#繪製k線圖
@st.cache_data
def plot_candle(stock_data, mav_days):
    fig = go.Figure()
    # K线图
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Adj Close']))
    # 移動平均線
    mav5 = stock_data['Close'].rolling(window=5).mean()  # 5日mav
    mav10 = stock_data['Close'].rolling(window=10).mean()  # 10日mav
    mav15 = stock_data['Close'].rolling(window=15).mean()  # 15日mav
    mav = stock_data['Close'].rolling(window=mav_days).mean()  # mav_days日mav    
    fig.add_trace(go.Scatter(x=stock_data.index, y=mav5, mode='lines', name='MAV-5'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=mav10, mode='lines', name='MAV-10'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=mav15, mode='lines', name='MAV-15'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=mav, mode='lines', name=f'MAV-{mav_days}'))
    # 更新圖表佈局
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis_title='日期', yaxis_title='價格')
    st.subheader(f'{symbol}-K線圖')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#繪製交易量柱狀圖
@st.cache_data
def plot_volume(stock_data):
    fig = go.Figure(data=[go.Bar(x=stock_data.index, y=stock_data['Volume'])])
    fig.update_layout(xaxis_title='日期', yaxis_title='交易量')
    st.subheader(f'{symbol}-交易量')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#繪製趨勢圖
@st.cache_data
def plot_trend(stock_data):
    fig = go.Figure()
    # 收盤價線
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], mode='lines', name='調整後收盤價'))
    # 開盤價線
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Open'], mode='lines', name='開盤價'))
    # 更新佈局
    fig.update_layout(xaxis_title='日期', yaxis_title='')
    # 顯示圖表
    st.subheader(f'{symbol}-趨勢圖')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#股票比較
@st.cache_data
def stock_data_vs(symbols,start_date,end_date):
    try:
        stock_data = yf.download(symbols, start=start_date, end=end_date)
        stock_data = stock_data.drop(['Open','High','Low','Close'], axis=1)
        st.subheader('交易數據')
        with st.expander("展開數據"):
            st.write(stock_data)
        return stock_data
    except Exception as e:
        st.error(f"無法獲取交易數據: {str(e)}")
        return None

@st.cache_data
def plot_trend_vs(stock_data, symbols):
    fig = go.Figure()
    for symbol in symbols:
        if symbol in stock_data.columns.get_level_values(1):
            df = stock_data.xs(symbol, level=1, axis=1)
            df['Adj Close Log'] = np.log(df['Adj Close'])  # 對 Adj Close 進行對數轉換
            fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close Log'], mode='lines', name=symbol))
    fig.update_layout(xaxis_title='日期', yaxis_title='價格')
    st.subheader('趨勢比較圖')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_volume_chart(stock_data,symbols):
    fig = go.Figure()
    for symbol in symbols:
        if symbol in stock_data.columns.get_level_values(1):
            df = stock_data.xs(symbol, level=1, axis=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name=symbol))
    fig.update_layout(xaxis_title='日期', yaxis_title='交易量')
    st.subheader('交易量比較圖')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

# 內部交易
@st.cache_data
def stock_insider_transactions(symbol, head):
    translation = {
        'Shares':'股份',
        'Value':'價值',
        'Text':'事件',
        'Insider':'內部人員',
        'Position':'職位',
        'Start Date':'開始日期',
        'Ownership':'持有權'
        }
    try:
        ticker = yf.Ticker(symbol)
        insider_transactions = ticker.insider_transactions
        insider_transactions = insider_transactions.drop(columns=['URL','Transaction'])
        insider_transactions = insider_transactions.rename(columns=translation)
        if insider_transactions is not None and not insider_transactions.empty:
            if head > 0:
                insider_transactions = insider_transactions.head(head)               
                # 將數字轉換為千位數格式
                insider_transactions['股份'] = insider_transactions['股份'].apply(lambda x: "{:,.0f}".format(x) if isinstance(x, int) else x)
                insider_transactions['價值'] = insider_transactions['價值'].apply(lambda x: "${:,.0f}".format(x) if isinstance(x, int) else x)
                
                st.subheader(f'{symbol}-最近{head}筆內部動作')
                st.write(insider_transactions)
            else:
                st.warning("請輸入大於 0 的數字")
        else:
            st.error(f"無法獲取{symbol}內部交易數據或數據為空")
    except Exception as e:
        st.error(f"獲取{symbol}內部交易數據時出錯：{str(e)}")
        
#內部持股
@st.cache_data
def stock_insider_roster_holders(symbol):
    symbol = symbol
    symbol = yf.Ticker(symbol)
    insider_roster_holders = symbol.insider_roster_holders
    insider_roster_holders = insider_roster_holders.drop(columns='URL')
    st.subheader('內部持股')
    insider_roster_holders = st.write(insider_roster_holders)
    return insider_roster_holders

#機構持股
@st.cache_data
def stock_institutional_holders(symbol):
    translation = {
        'Date Reported': '日期',
        'Holder': '機構名稱',
        'pctHeld': '持股百分比',
        'Shares': '股份',
        'Value': '價值'
    }
    ticker = yf.Ticker(symbol)  # 使用參數中的 symbol 創建 Ticker 物件
    institutional_holders = ticker.institutional_holders
    institutional_holders = institutional_holders.rename(columns=translation) 
    # 將百分比轉換為百分數形式
    institutional_holders['持股百分比'] = institutional_holders['持股百分比'].apply(lambda x: f"{x:.2f}%" if isinstance(x, float) else x)   
    # 將數字轉換為千位數格式
    institutional_holders['股份'] = institutional_holders['股份'].apply(lambda x: "{:,.0f}".format(x) if isinstance(x, int) else x)
    institutional_holders['價值'] = institutional_holders['價值'].apply(lambda x: "${:,.0f}".format(x) if isinstance(x, int) else x)
    st.subheader(f'持有{symbol}的機構')
    st.write(institutional_holders)
    return institutional_holders

#機構買賣
@st.cache_data
def stock_upgrades_downgrades(symbol, head):
    translation = {
        'GradeDate':'日期',
        'Firm':'機構',
        'ToGrade':'最新動作',
        'FromGrade':'之前動作',
        'Action':'立場',
        }
    try:
        ticker = yf.Ticker(symbol)
        upgrades_downgrade = ticker.upgrades_downgrades
        upgrades_downgrade = upgrades_downgrade.rename(columns=translation)
        if upgrades_downgrade is not None and not upgrades_downgrade.empty:
            if head > 0:
                upgrades_downgrade = upgrades_downgrade.head(head)               
                st.subheader(f'機構買賣{symbol}最近{head}筆數據')
                st.write(upgrades_downgrade)
            else:
                st.warning("請輸入大於 0 的數字")
        else:
            st.error(f"無法獲取持有{symbol}的機構買賣數據或數據為空")
    except Exception as e:
        st.error(f"獲取機構買賣{symbol}數據時出錯：{str(e)}")

#相關新聞
@st.cache_data
def display_news_table(symbol):
    translation_columns = {
        'title':'標題',
        'publisher':'出版商',
        'link':'網址',
        'relatedTickers':'相關股票代碼'
    }
    ticker = yf.Ticker(symbol)
    news = ticker.news
    news_df = pd.DataFrame(news).drop(columns=['uuid', 'providerPublishTime', 'type', 'thumbnail'])
    news_df = news_df.rename(columns=translation_columns)
    st.subheader(f'{symbol}-相關新聞')
    st.write(news_df)

@st.cache_data   
def display_news_links(symbol):
    ticker = yf.Ticker(symbol)
    news = ticker.news
    news_df = pd.DataFrame(news).drop(columns=['uuid', 'providerPublishTime', 'type', 'thumbnail'])
    st.subheader(f'{symbol}-相關新聞連結')
    # 存储链接的列表
    urls = []
    for url in news_df['link']:
        urls.append(url)
        st.markdown(f"<a href='{url}' target='_blank'>{url}</a>", unsafe_allow_html=True)

# 台股區

@st.cache_data
def plot_index_tw(period):
    # Fetch historical data for S&P 500
    twse_data = yf.download('^TWII', period=period)
    tpex_data = yf.download('^TWOII', period=period)
    tw50_data = yf.download('0050.TW', period=period)   
    # Extract Close prices
    twse_close = twse_data['Close']
    tpex_close = tpex_data['Close']
    tw50_close = tw50_data['Close']
    # Take the logarithm of the Close prices
    twse_log_close = np.log(twse_close)
    tpex_log_close = np.log(tpex_close)
    tw50_log_close = np.log(tw50_close)
    st.subheader(f'上市＆櫃檯&0050{period}走勢')
    # Create Plotly figure
    fig = go.Figure()   
    # Add trace for Log Close price
    fig.add_trace(go.Scatter(x=twse_log_close.index, y=twse_log_close.values, mode='lines', name='加權指數'))
    fig.add_trace(go.Scatter(x=tpex_log_close.index, y=tpex_log_close.values, mode='lines', name='櫃檯指數'))
    fig.add_trace(go.Scatter(x=tw50_log_close.index, y=tw50_log_close.values, mode='lines', name='0050'))
    # Update layout
    fig.update_layout(xaxis_title='Date', yaxis_title='Log Close Price')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_tw_asia(period):
    # Fetch historical data for S&P 500
    sha_data = yf.download('000001.SS',period=period )
    shz_data = yf.download('399001.SZ', period=period)
    twse_data = yf.download('^TWII', period=period)
    jp_data = yf.download('^N225', period=period)
    hk_data = yf.download('^HSI', period=period)
    kr_data = yf.download('^KS11', period=period)
    sin_data = yf.download('^STI', period=period)
    # Extract Close prices
    sha_close = sha_data['Close'] * 4.4927  # 將上證指數轉換為新台幣
    shz_close = shz_data['Close'] * 4.4927  # 將上證指數轉換為新台幣
    twse_close = twse_data['Close']
    jp_close = jp_data['Close'] * 0.2084    # 將日經指數轉換為新台幣
    hk_close = hk_data['Close'] * 4.1549    # 將恒生指數轉換為新台幣
    kr_close = kr_data['Close'] * 0.0237    # 將韓國綜合股價指數轉換為新台幣
    sin_close = sin_data['Close'] * 23.9665 # 將新加坡海峽時報指數轉換為新台幣
    # Take the logarithm of the Close prices
    sha_log_close = np.log(sha_close)
    shz_log_close = np.log(shz_close)
    twse_log_close = np.log(twse_close)
    jp_log_close = np.log(jp_close)
    hk_log_close = np.log(hk_close)
    kr_log_close = np.log(kr_close)
    sin_log_close = np.log(sin_close) 
    st.subheader(f'台股大盤＆亞洲大盤{period}走勢')
    # Create Plotly figure
    fig = go.Figure()   
    # Add trace for Log Close price
    fig.add_trace(go.Scatter(x=sha_log_close.index, y=sha_log_close.values, mode='lines', name='上證指數'))
    fig.add_trace(go.Scatter(x=shz_log_close.index, y=shz_log_close.values, mode='lines', name='深證指數'))
    fig.add_trace(go.Scatter(x=twse_log_close.index, y=twse_log_close.values, mode='lines', name='加權指數'))
    fig.add_trace(go.Scatter(x=jp_log_close.index, y=jp_log_close.values, mode='lines', name='日經指數'))
    fig.add_trace(go.Scatter(x=hk_log_close.index, y=hk_log_close.values, mode='lines', name='恒生指數'))
    fig.add_trace(go.Scatter(x=kr_log_close.index, y=kr_log_close.values, mode='lines', name='韓國綜合股價指數'))
    fig.add_trace(go.Scatter(x=sin_log_close.index, y=sin_log_close.values, mode='lines', name='新加坡海峽時報指數'))
    # Update layout
    fig.update_layout(xaxis_title='Date', yaxis_title='Log Close Price')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_pct_tw(period):
    # Fetch historical data for S&P 500
    twse_data = yf.download('^TWII',period=period )
    sha_data = yf.download('000001.SS', period=period)
    shz_data = yf.download('399001.SZ', period=period)
    jp_data = yf.download('^N225', period=period)
    hk_data = yf.download('^HSI', period=period)
    kr_data = yf.download('^KS11',period=period)
    sin_data = yf.download('^STI', period=period) 
    # Extract Close prices
    sha_close = sha_data['Close'] * 4.4927  # 將上證指數轉換為新台幣
    shz_close = shz_data['Close'] * 4.4927  # 將上證指數轉換為新台幣
    twse_close = twse_data['Close']
    jp_close = jp_data['Close'] * 0.2084    # 將日經指數轉換為新台幣
    hk_close = hk_data['Close'] * 4.1549    # 將恒生指數轉換為新台幣
    kr_close = kr_data['Close'] * 0.0237    # 將韓國綜合股價指數轉換為新台幣
    sin_close = sin_data['Close'] * 23.9665 # 將新加坡海峽時報指數轉換為新台幣
    # Calculate total returns
    twse_total_return = ((twse_close.iloc[-1] - twse_close.iloc[0]) / twse_close.iloc[0]) * 100
    shz_total_return = ((shz_close.iloc[-1] - shz_close.iloc[0]) / shz_close.iloc[0]) * 100
    sha_total_return = ((sha_close.iloc[-1] - sha_close.iloc[0]) / sha_close.iloc[0]) * 100
    jp_total_return = ((jp_close.iloc[-1] - jp_close.iloc[0]) / jp_close.iloc[0]) * 100
    hk_total_return = ((hk_close.iloc[-1] - hk_close.iloc[0]) / hk_close.iloc[0]) * 100
    kr_total_return = ((kr_close.iloc[-1] - kr_close.iloc[0]) / kr_close.iloc[0]) * 100
    sin_total_return = ((sin_close.iloc[-1] - sin_close.iloc[0]) / sin_close.iloc[0]) * 100
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
    }
    # Sort the dictionary by values in descending order
    sorted_returns = dict(sorted(returns_dict.items(), key=lambda item: item[1], reverse=True))
    colors = px.colors.qualitative.Plotly
    # Add traces for Total Returns
    fig.add_trace(go.Bar(x=list(sorted_returns.keys()),
                         y=list(sorted_returns.values()),
                         marker_color=colors))
    # Update layout
    st.subheader(f'台股大盤＆亞洲大盤{period}報酬率％')
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

#成交量前二十名證券
@st.cache_data
def twse_20():
    # Get data from the API
    response = res.get('https://openapi.twse.com.tw/v1/exchangeReport/MI_INDEX20')
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()      
        # Create DataFrame from the retrieved data
        df = pd.DataFrame(data) 
        # Convert TradeVolume column to numeric for sorting
        df['TradeVolume'] = pd.to_numeric(df['TradeVolume'])
        # Sort DataFrame by TradeVolume in descending order
        df_sorted = df.sort_values(by='TradeVolume', ascending=False)
        # Plot the bar chart
        fig = go.Figure(data=[go.Bar(x=df_sorted['Name'], y=df_sorted['TradeVolume'], marker_color='rgba(0,0,255,0.6)')])
        fig.update_layout(xaxis_title='Name', yaxis_title='TradeVolume')
        # Display the bar chart
        st.subheader('今日上市交易量前20名')
        st.plotly_chart(fig)
        # Display the data table
        with st.expander("展開數據"):
            st.write(df_sorted)
    else:
        st.error('Failed to fetch data from the API')

@st.cache_data
def tpex_20():
    # Get data from the API
    response = res.get('https://www.tpex.org.tw/openapi/v1/tpex_volume_rank')
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()      
        # Create DataFrame from the retrieved data
        df = pd.DataFrame(data) 
        # Convert TradeVolume column to numeric for sorting
        df['TradingVolume'] = pd.to_numeric(df['TradingVolume'])
        # Sort DataFrame by TradeVolume in descending order
        df_sorted = df.sort_values(by='TradingVolume', ascending=False).head(20)
        # Plot the bar chart
        fig = go.Figure(data=[go.Bar(x=df_sorted['CompanyName'], y=df_sorted['TradingVolume'], marker_color='rgba(255,0,0,0.6)')])
        fig.update_layout(xaxis_title='CompanyName', yaxis_title='TradingVolume')
        # Display the bar chart
        st.subheader('今日櫃檯交易量前20名')
        st.plotly_chart(fig)
        # Display the data table
        with st.expander("展開數據"):
            st.write(df_sorted)
    else:
        st.error('Failed to fetch data from the API')

#公司基本資訊
@st.cache_data
def twse_info(symbol):
    try:
        symbol_tw = symbol + ".tw"  # 在股票代碼後加上.tw
        stock_info = yf.Ticker(symbol_tw)
        com_info = stock_info.info
        return com_info
    except Exception as e:
        st.error(f"無法獲取{symbol}基本資訊or{symbol}為上櫃興櫃公司：{str(e)}")
        return None

@st.cache_data
def tpex_info(symbol):
    try:
        symbol_tw = symbol + ".two"  # 在股票代碼後加上.two
        stock_info = yf.Ticker(symbol_tw)
        com_info = stock_info.info
        return com_info
    except Exception as e:
        st.error(f"無法獲取{symbol}基本資訊or{symbol}為上市公司：{str(e)}")
        return None

@st.cache_data    
def display_location(twse_info):
    if 'city' in twse_info and 'country' in twse_info:
        city = twse_info['city']
        country = twse_info['country']

        # 使用 Nominatim 服务进行地理编码
        geolocator = Nominatim(user_agent="streamlit_app")
        location = geolocator.geocode(f"{city}, {country}")

        if location:
            # 使用 folium 创建地图，并将其定位到公司位置
            map = folium.Map(location=[location.latitude, location.longitude], zoom_start=10)
            # 添加标记
            folium.Marker([location.latitude, location.longitude], popup=f"{city}, {country}").add_to(map)
            # 使用 streamlit-folium 显示地图
            folium_static(map)
        else:
            st.error(f"無法找到{symbol}位置or{symbol}為上櫃興櫃公司")

@st.cache_data    
def display_location(tpex_info):
    if 'city' in tpex_info and 'country' in tpex_info:
        city = tpex_info['city']
        country = tpex_info['country']
        # 使用 Nominatim 服务进行地理编码
        geolocator = Nominatim(user_agent="streamlit_app")
        location = geolocator.geocode(f"{city}, {country}")
        if location:
            # 使用 folium 创建地图，并将其定位到公司位置
            map = folium.Map(location=[location.latitude, location.longitude], zoom_start=10)
            # 添加标记
            folium.Marker([location.latitude, location.longitude], popup=f"{city}, {country}").add_to(map)
            # 使用 streamlit-folium 显示地图
            folium_static(map)
        else:
            st.error(f"無法找到{symbol}位置or{symbol}為上市公司")

@st.cache_data
def display_info(twse_info):
    if twse_info:
        
        selected_indicators = ['longName', 'country', 'city', 'marketCap', 'totalRevenue', 'grossMargins', 'operatingMargins',
                               'profitMargins', 'trailingEps', 'pegRatio', 'dividendRate', 'payoutRatio', 'bookValue',
                               'operatingCashflow', 'freeCashflow', 'returnOnEquity']
        selected_info = {indicator: twse_info.get(indicator, '') for indicator in selected_indicators}

        #建立字典翻譯
        translation = {
            'longName': '公司名稱',
            'country': '國家',
            'city': '城市',
            'marketCap': '市值',
            'totalRevenue': '總收入',
            'grossMargins': '毛利率',
            'operatingMargins': '營業利潤率', 
            'profitMargins': '净利率',
            'trailingEps': '每股收益',
            'pegRatio': 'PEG 比率',
            'dividendRate': '股息率',
            'payoutRatio': '股息支付比例',
            'bookValue': '每股淨資產',
            'operatingCashflow': '營運現金流',
            'freeCashflow': '自由現金流',
            'returnOnEquity': '股東權益報酬率'
        }
        #Pandas DataFrame
        company_info = pd.DataFrame.from_dict(selected_info,orient='index',columns=['Value'])
        company_info.rename(index=translation,inplace=True)
        #轉換成百分比
        percent_columns = ['毛利率', '營業利潤率', '净利率', '股息率', '股息支付比例', '股東權益報酬率']
        for col in percent_columns:
            if col in company_info.index:
                company_info.at[col, 'Value'] = pd.to_numeric(company_info.at[col, 'Value'], errors='coerce')  # 将非数字转换为 NaN
                company_info.at[col, 'Value'] = f"{company_info.at[col, 'Value']:.2%}" if pd.notna(company_info.at[col, 'Value']) else None
        #千分位表示
        company_info['Value'] = company_info['Value'].apply(lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) and x >= 1000 else x)
        st.subheader(f"{symbol}-基本資訊")
        st.table(company_info)
        st.subheader(f"{symbol}-位置資訊")
        display_location(twse_info)
    else:
        st.error(f"無法獲取{symbol}-基本訊息or{symbol}為上櫃興櫃公司")

@st.cache_data
def display_info(tpex_info):
    if tpex_info:
        
        selected_indicators = ['longName', 'country', 'city', 'marketCap', 'totalRevenue', 'grossMargins', 'operatingMargins',
                               'profitMargins', 'trailingEps', 'pegRatio', 'dividendRate', 'payoutRatio', 'bookValue',
                               'operatingCashflow', 'freeCashflow', 'returnOnEquity']
        selected_info = {indicator: tpex_info.get(indicator, '') for indicator in selected_indicators}
        #建立字典翻譯
        translation = {
            'longName': '公司名稱',
            'country': '國家',
            'city': '城市',
            'marketCap': '市值',
            'totalRevenue': '總收入',
            'grossMargins': '毛利率',
            'operatingMargins': '營業利潤率', 
            'profitMargins': '净利率',
            'trailingEps': '每股收益',
            'pegRatio': 'PEG 比率',
            'dividendRate': '股息率',
            'payoutRatio': '股息支付比例',
            'bookValue': '每股淨資產',
            'operatingCashflow': '營運現金流',
            'freeCashflow': '自由現金流',
            'returnOnEquity': '股東權益報酬率'
        }
        #Pandas DataFrame
        company_info = pd.DataFrame.from_dict(selected_info,orient='index',columns=['Value'])
        company_info.rename(index=translation,inplace=True)
        #轉換成百分比
        percent_columns = ['毛利率', '營業利潤率', '净利率', '股息率', '股息支付比例', '股東權益報酬率']
        for col in percent_columns:
            if col in company_info.index:
                company_info.at[col, 'Value'] = pd.to_numeric(company_info.at[col, 'Value'], errors='coerce')  # 将非数字转换为 NaN
                company_info.at[col, 'Value'] = f"{company_info.at[col, 'Value']:.2%}" if pd.notna(company_info.at[col, 'Value']) else None
        #千分位表示
        company_info['Value'] = company_info['Value'].apply(lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) and x >= 1000 else x)
        st.subheader(f"{symbol}-基本資訊")
        st.table(company_info)
        st.subheader(f"{symbol}-位置資訊")
        display_location(tpex_info)
    else:
        st.error(f"無法獲取{symbol}-基本訊息or{symbol}為上市公司")

#財報-年度
@st.cache_data
def financial_statements_twse(symbol):
    try:
        symbol = symbol + ".tw"  # 在股票代碼後加上.tw
        stock_info = yf.Ticker(symbol)
        balance_sheet_twse = stock_info.balance_sheet
        income_statement_twse = stock_info.income_stmt
        cash_flow_twse = stock_info.cashflow
        return balance_sheet_twse, income_statement_twse, cash_flow_twse
    except Exception as e:
        st.error(f"獲取{symbol}-財報發生錯誤or{symbol}為上櫃興櫃公司：{str(e)}")
        return None, None, None
    
@st.cache_data
def financial_statements_tpex(symbol):
    try:
        symbol = symbol + ".two"  # 在股票代碼後加上.tw
        stock_info = yf.Ticker(symbol)
        balance_sheet_tpex = stock_info.balance_sheet
        income_statement_tpex = stock_info.income_stmt
        cash_flow_tpex = stock_info.cashflow
        return balance_sheet_tpex, income_statement_tpex, cash_flow_tpex
    except Exception as e:
        st.error(f"獲取{symbol}-財報發生錯誤or{symbol}為上市公司：{str(e)}")
        return None, None, None

@st.cache_data
def balance_twse(balance_sheet_twse):
    if balance_sheet_twse is not None:
        st.subheader(f"{symbol}-資產負債表(年度)")
        st.write(balance_sheet_twse)

@st.cache_data
def balance_tpex(balance_sheet_tpex):
    if balance_sheet_tpex is not None:
        st.subheader(f"{symbol}-資產負債表(年度)")
        st.write(balance_sheet_tpex)

@st.cache_data
def income_twse(income_statement_twse):
    if income_statement_twse is not None:
        st.subheader(f"{symbol}-綜合損益表(年度)")
        st.write(income_statement_twse)

@st.cache_data
def income_tpex(income_statement_tpex):
    if income_statement_tpex is not None:
        st.subheader(f"{symbol}-綜合損益表(年度)")
        st.write(income_statement_tpex)

@st.cache_data
def cashflow_twse(cash_flow_twse):
    if cash_flow_twse is not None:
        st.subheader(f"{symbol}-現金流量表(年度)")
        st.write(cash_flow_twse)

@st.cache_data
def cashflow_tpex(cash_flow_tpex):
    if cash_flow_tpex is not None:
        st.subheader(f"{symbol}-現金流量表(年度)")
        st.write(cash_flow_tpex)

#財報-季度
@st.cache_data
def financial_statements_quarterly_twse(symbol):
    try:
        symbol = symbol + ".tw"  # 在股票代碼後加上.tw
        stock_info = yf.Ticker(symbol)
        balance_sheet_quarterly_twse = stock_info.quarterly_balance_sheet
        income_statement_quarterly_twse = stock_info.quarterly_income_stmt
        cash_flow_quarterly_twse = stock_info.quarterly_cashflow  # 這裡修正了錯誤
        return balance_sheet_quarterly_twse, income_statement_quarterly_twse, cash_flow_quarterly_twse
    except Exception as e:
        st.error(f"獲取{symbol}-財報發生錯誤or{symbol}為上櫃興櫃公司：{str(e)}")
        return None, None, None
    
@st.cache_data
def financial_statements_quarterly_tpex(symbol):
    try:
        symbol = symbol + ".two"  # 在股票代碼後加上.tw
        stock_info = yf.Ticker(symbol)
        balance_sheet_quarterly_tpex = stock_info.quarterly_balance_sheet
        income_statement_quarterly_tpex = stock_info.quarterly_income_stmt
        cash_flow_quarterly_tpex = stock_info.quarterly_cashflow  # 這裡修正了錯誤
        return balance_sheet_quarterly_tpex, income_statement_quarterly_tpex, cash_flow_quarterly_tpex
    except Exception as e:
        st.error(f"獲取{symbol}-財報發生錯誤or{symbol}為上市公司：{str(e)}")
        return None, None, None

@st.cache_data
def balance_quarterly_twse(balance_sheet_quarterly_twse):
    if balance_sheet_quarterly_twse is not None:
        st.subheader(f"{symbol}-資產負債表(季度)")
        st.write(balance_sheet_quarterly_twse)

@st.cache_data
def balance_quarterly_tpex(balance_sheet_quarterly_tpex):
    if balance_sheet_quarterly_tpex is not None:
        st.subheader(f"{symbol}-資產負債表(季度)")
        st.write(balance_sheet_quarterly_tpex)

@st.cache_data
def income_quarterly_twse(income_statement_quarterly_twse):
    if income_statement_quarterly_twse is not None:
        st.subheader(f"{symbol}-綜合損益表(季度)")
        st.write(income_statement_quarterly_twse)

@st.cache_data
def income_quarterly_tpex(income_statement_quarterly_tpex):
    if income_statement_quarterly_tpex is not None:
        st.subheader(f"{symbol}-綜合損益表(季度)")
        st.write(income_statement_quarterly_tpex)

@st.cache_data
def cashflow_quarterly_twse(cash_flow_quarterly_twse):
    if cash_flow_quarterly_twse is not None:
        st.subheader(f"{symbol}-現金流量表(季度)")
        st.write(cash_flow_quarterly_twse)

@st.cache_data
def cashflow_quarterly_tpex(cash_flow_quarterly_tpex):
    if cash_flow_quarterly_tpex is not None:
        st.subheader(f"{symbol}-現金流量表(季度)")
        st.write(cash_flow_quarterly_tpex)

#月營收表
@st.cache_data
def twse_month(symbol):
    try:
        # 發送請求並讀取 JSON 資料
        url = res.get('https://openapi.twse.com.tw/v1/opendata/t187ap05_L')
        data = url.json()
        # 將 JSON 資料轉換成 DataFrame
        df = pd.DataFrame(data)
        # 使用者輸入股票代號
        symbol = symbol
        # 尋找符合股票代號的資料
        twse = df.loc[df['公司代號'] == symbol]
        # 列印結果
        if not twse.empty:
            st.subheader(f'{symbol}月營收表')
            st.write(twse)            
            # 提取所需數據
            columns = ['營業收入-當月營收', '營業收入-上月營收', '營業收入-去年當月營收', '累計營業收入-當月累計營收', '累計營業收入-去年累計營收']
            data = twse[columns].squeeze()  # 將 DataFrame 轉換為 Series            
            # 繪製長條圖
            fig = go.Figure(data=[go.Bar(x=data.index, y=data.values)])
            fig.update_layout(title=f'{symbol} 營業收入', xaxis_title='項目', yaxis_title='金額')
            st.subheader(f'{symbol}月營收圖表')
            st.plotly_chart(fig)           
        else:
            st.error(f"找不到{symbol}月營收表or{symbol}為上櫃興櫃公司")
    except Exception as e:
        st.error(f"獲取{symbol}月營收表時發生錯誤：{str(e)}")

@st.cache_data
def tpex_month(symbol):
    try:
        # 發送請求並讀取 JSON 資料
        url = res.get('https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O')
        data = url.json()
        # 將 JSON 資料轉換成 DataFrame
        df = pd.DataFrame(data)
        # 使用者輸入股票代號
        symbol = symbol
        # 尋找符合股票代號的資料
        twse = df.loc[df['公司代號'] == symbol]
        # 列印結果
        if not twse.empty:
            st.subheader(f'{symbol}月營收表')
            st.write(twse)            
            # 提取所需數據
            columns = ['營業收入-當月營收', '營業收入-上月營收', '營業收入-去年當月營收', '累計營業收入-當月累計營收', '累計營業收入-去年累計營收']
            data = twse[columns].squeeze()  # 將 DataFrame 轉換為 Series            
            # 繪製長條圖
            fig = go.Figure(data=[go.Bar(x=data.index, y=data.values)])
            fig.update_layout(title=f'{symbol} 營業收入', xaxis_title='項目', yaxis_title='金額')
            st.subheader(f'{symbol}月營收圖表')
            st.plotly_chart(fig)           
        else:
            st.error(f"找不到{symbol}月營收表or{symbol}為上市公司")
    except Exception as e:
        st.error(f"獲取{symbol}月營收表時發生錯誤：{str(e)}")

#獲取歷史交易數據
@st.cache_data
def twse_data(symbol,start_date,end_date):
    try:
        symbol = symbol + ".tw"
        twse_data = yf.download(symbol,start=start_date,end=end_date)
        st.subheader('交易數據')
        with st.expander("展開數據"):
            st.write(twse_data)
        return twse_data
    except Exception as e:
        st.error(f"無法獲取{symbol}-交易數據：{str(e)}")
        return None

#繪製k線圖
@st.cache_data
def twse_candle(twse_data, mav_days):
    fig = go.Figure()
    # K线图
    fig.add_trace(go.Candlestick(x=twse_data.index,
                                 open=twse_data['Open'],
                                 high=twse_data['High'],
                                 low=twse_data['Low'],
                                 close=twse_data['Adj Close']))
    # 移動平均線
    mav5 = twse_data['Close'].rolling(window=5).mean()  # 5日mav
    mav10 = twse_data['Close'].rolling(window=10).mean()  # 10日mav
    mav15 = twse_data['Close'].rolling(window=15).mean()  # 15日mav
    mav = twse_data['Close'].rolling(window=mav_days).mean()  # mav_days日mav    
    fig.add_trace(go.Scatter(x=twse_data.index, y=mav5, mode='lines', name='MAV-5'))
    fig.add_trace(go.Scatter(x=twse_data.index, y=mav10, mode='lines', name='MAV-10'))
    fig.add_trace(go.Scatter(x=twse_data.index, y=mav15, mode='lines', name='MAV-15'))
    fig.add_trace(go.Scatter(x=twse_data.index, y=mav, mode='lines', name=f'MAV-{mav_days}'))
    # 更新圖表佈局
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis_title='日期', yaxis_title='價格')
    st.subheader(f'{symbol}-K線圖')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#繪製交易量柱狀圖
@st.cache_data
def twse_volume(twse_data):
    fig = go.Figure(data=[go.Bar(x=twse_data.index, y=twse_data['Volume'])])
    fig.update_layout(xaxis_title='日期', yaxis_title='交易量')
    st.subheader(f'{symbol}-交易量')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#繪製趨勢圖
@st.cache_data
def twse_trend(twse_data):
    fig = go.Figure()
    # 收盤價線
    fig.add_trace(go.Scatter(x=twse_data.index, y=twse_data['Adj Close'], mode='lines', name='調整後收盤價'))
    # 開盤價線
    fig.add_trace(go.Scatter(x=twse_data.index, y=twse_data['Open'], mode='lines', name='開盤價'))
    # 更新佈局
    fig.update_layout(xaxis_title='日期', yaxis_title='')
    # 顯示圖表
    st.subheader(f'{symbol}-趨勢圖')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#股票比較
@st.cache_data
def twse_data_vs(symbols,start_date,end_date):
    try:
        twse_data = yf.download(symbols, start=start_date, end=end_date)
        twse_data = twse_data.drop(['Open','High','Low','Close'], axis=1)
        st.subheader('交易數據')
        with st.expander("展開數據"):
            st.write(twse_data)
        return twse_data
    except Exception as e:
        st.error(f"無法獲取交易數據: {str(e)}")
        return None

@st.cache_data
def twse_trend_vs(twse_data,symbols):
    fig = go.Figure()
    for symbol in symbols:
        if symbol in twse_data.columns.get_level_values(1):
            df = twse_data.xs(symbol, level=1, axis=1)
            df['Adj Close Log'] = np.log(df['Adj Close'])  # 對 Adj Close 進行對數轉換
            fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close Log'], mode='lines', name=symbol))
    fig.update_layout(xaxis_title='日期', yaxis_title='價格')
    st.subheader('趨勢比較圖')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def twse_volume_chart(twse_data,symbols):
    fig = go.Figure()
    for symbol in symbols:
        if symbol in twse_data.columns.get_level_values(1):
            df = twse_data.xs(symbol, level=1, axis=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name=symbol))
    fig.update_layout(xaxis_title='日期', yaxis_title='交易量')
    st.subheader('交易量比較圖')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#獲取歷史交易數據
@st.cache_data
def tpex_data(symbol,start_date,end_date):
    try:
        symbol = symbol + ".two"
        tpex_data = yf.download(symbol,start=start_date,end=end_date)
        st.subheader('交易數據')
        with st.expander("展開數據"):
            st.write(tpex_data)
        return tpex_data
    except Exception as e:
        st.error(f"無法獲取{symbol}-交易數據：{str(e)}")
        return None

#繪製k線圖
@st.cache_data
def tpex_candle(tpex_data, mav_days):
    fig = go.Figure()
    # K线图
    fig.add_trace(go.Candlestick(x=tpex_data.index,
                                 open=tpex_data['Open'],
                                 high=tpex_data['High'],
                                 low=tpex_data['Low'],
                                 close=tpex_data['Adj Close']))
    # 移動平均線
    mav5 = tpex_data['Close'].rolling(window=5).mean()  # 5日mav
    mav10 = tpex_data['Close'].rolling(window=10).mean()  # 10日mav
    mav15 = tpex_data['Close'].rolling(window=15).mean()  # 15日mav
    mav = tpex_data['Close'].rolling(window=mav_days).mean()  # mav_days日mav    
    fig.add_trace(go.Scatter(x=tpex_data.index, y=mav5, mode='lines', name='MAV-5'))
    fig.add_trace(go.Scatter(x=tpex_data.index, y=mav10, mode='lines', name='MAV-10'))
    fig.add_trace(go.Scatter(x=tpex_data.index, y=mav15, mode='lines', name='MAV-15'))
    fig.add_trace(go.Scatter(x=tpex_data.index, y=mav, mode='lines', name=f'MAV-{mav_days}'))
    # 更新圖表佈局
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis_title='日期', yaxis_title='價格')
    st.subheader(f'{symbol}-K線圖')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#繪製交易量柱狀圖
@st.cache_data
def tpex_volume(tpex_data):
    fig = go.Figure(data=[go.Bar(x=tpex_data.index, y=tpex_data['Volume'])])
    fig.update_layout(xaxis_title='日期', yaxis_title='交易量')
    st.subheader(f'{symbol}-交易量')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#繪製趨勢圖
@st.cache_data
def tpex_trend(tpex_data):
    fig = go.Figure()
    # 收盤價線
    fig.add_trace(go.Scatter(x=tpex_data.index, y=tpex_data['Adj Close'], mode='lines', name='調整後收盤價'))
    # 開盤價線
    fig.add_trace(go.Scatter(x=tpex_data.index, y=tpex_data['Open'], mode='lines', name='開盤價'))
    # 更新佈局
    fig.update_layout(xaxis_title='日期', yaxis_title='')
    # 顯示圖表
    st.subheader(f'{symbol}-趨勢圖')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#股票比較
@st.cache_data
def tpex_data_vs(symbols, start_date, end_date):
    try:
        tpex_data = yf.download(symbols, start=start_date, end=end_date)
        tpex_data = tpex_data.drop(['Open','High','Low','Close'], axis=1)
        st.subheader('交易數據')
        with st.expander("展開數據"):
            st.write(tpex_data)
        return tpex_data
    except Exception as e:
        st.error(f"無法獲取交易數據: {str(e)}")
        return None

@st.cache_data
def tpex_trend_vs(tpex_data,symbols):
    fig = go.Figure()
    for symbol in symbols:
        if symbol in tpex_data.columns.get_level_values(1):
            df = tpex_data.xs(symbol, level=1, axis=1)
            df['Adj Close Log'] = np.log(df['Adj Close'])  # 對 Adj Close 進行對數轉換
            fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close Log'], mode='lines', name=symbol))
    fig.update_layout(xaxis_title='日期', yaxis_title='價格')
    st.subheader('趨勢比較圖')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def tpex_volume_chart(tpex_data,symbols):
    fig = go.Figure()
    for symbol in symbols:
        if symbol in tpex_data.columns.get_level_values(1):
            df = tpex_data.xs(symbol, level=1, axis=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name=symbol))
    fig.update_layout(xaxis_title='日期', yaxis_title='交易量')
    st.subheader('交易量比較圖')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#streamlit版面配置
st.markdown("<h1 style='text-align: center; color: rainbow;'>StockInfo</h1>", unsafe_allow_html=True)
st.header(' ',divider="rainbow")

st.sidebar.title('Menu')
market = st.sidebar.selectbox('選擇市場', ['美國','台灣'])
options = st.sidebar.selectbox('選擇功能', ['大盤指數','今日熱門','公司基本資訊','公司財報查詢','交易數據','內部資訊','機構買賣','近期相關消息'])
st.sidebar.markdown('''
免責聲明：        
1. K 線圖請以美股角度來觀看      
        - 美股: 綠漲、紅跌        
        - 台股: 綠跌、紅漲           
2. 本平台僅適用於數據搜尋，不建議任何投資行為
3. 有些數據僅限美股，台股尚未支援  
4. 建議使用電腦或平板查詢數據  
''')

if market == '美國' and options == '大盤指數':
    period = st.selectbox('選擇時長',['年初至今','1年','3年','5年','10年','全部'])
    if period == '年初至今':
        period = 'ytd'
        plot_index(period)
        plot_pct(period)
        plot_foreign(period)
        plot_pct_foreign(period)
    elif period == '1年':
        period = '1y'
        plot_index(period)
        plot_pct(period)
        plot_foreign(period)
        plot_pct_foreign(period)
    elif period == '3年':
        period = '3y'
        plot_index(period)
        plot_pct(period)
        plot_foreign(period)
        plot_pct_foreign(period)
    elif period == '5年':
        period = '5y'
        plot_index(period)
        plot_pct(period)
        plot_foreign(period)
        plot_pct_foreign(period)
    elif period == '10年':
        period = '10y'
        plot_index(period)
        plot_pct(period)
        plot_foreign(period)
        plot_pct_foreign(period)
    elif period == '全部':
        period = 'max'
        plot_index(period)
        plot_pct(period)
        plot_foreign(period)
        plot_pct_foreign(period)
    with st.expander("顯示成份股"):
        st.write('S&P500成份股')
        sp500_dsymbol()
        st.write('NASDAQ100成份股')
        nasdaq_100symbol()
        st.write('道瓊工業成份股')
        dji_symbol()
    st.markdown("[美股指數名詞解釋](https://www.oanda.com/bvi-ft/lab-education/indices/us-4index/)")

elif market == '美國' and options == '今日熱門':
    hot_stock()
    gainers_stock()
    loser_stock()

elif market == '美國' and options == '公司基本資訊':
    symbol = st.text_input('輸入美股代號').upper()
    if st.button('查詢'):
        com_info = company_info(symbol)
        display_info(com_info)

elif market == '美國' and options == '公司財報查詢':
    select = st.selectbox('選擇查詢資訊',['年報','季報'])
    if select == '年報':
        symbol = st.text_input('輸入美股代號').upper()
        if st.button('查詢'):
            balance_sheet, income_statement, cash_flow = financial_statements(symbol)
            if balance_sheet is not None:
                balance(balance_sheet)
            if income_statement is not None:
                income(income_statement)
            if cash_flow is not None:
                cashflow(cash_flow)
            else:
                st.error(f"無法獲取{symbol}-年報")
    elif select == '季報':
        symbol = st.text_input('輸入美股代號').upper()
        if st.button('查詢'):
            balance_sheet_quarterly, income_statement_quarterly, cash_flow_quarterly = financial_statements_quarterly(symbol)
            if balance_sheet_quarterly is not None:
                balance_quarterly(balance_sheet_quarterly)
            if income_statement_quarterly is not None:
                income_quarterly(income_statement_quarterly)
            if cash_flow_quarterly is not None:
                cashflow_quarterly(cash_flow_quarterly)
            else:
                st.error(f"無法獲取{symbol}-季報")

elif market == '美國' and options == '交易數據':
    select = st.selectbox('選擇查詢資訊',['個股','多股'])
    if select == '個股':
        with st.expander("展開輸入參數"):
            symbol = st.text_input('輸入美股代號', key='single_stock').upper()
            start_date = st.date_input('開始日期', key='start_date')
            end_date = st.date_input('结束日期', key='end_date')
            mav_days = st.number_input('輸入MAV天數', min_value=15, max_value=360, value=15, step=1)  # 添加MAV天數的輸入
        if st.button('查詢'):
            stock_data = stock_data(symbol, start_date, end_date)
            if stock_data is not None:
                plot_candle(stock_data, mav_days)  # 將MAV天數傳遞給 plot_candle 函式
                plot_volume(stock_data)
                plot_trend(stock_data)
            else:
                st.error(f"無法獲取{symbol}交易數據")
    elif select == '多股':
        with st.expander("展開輸入參數"):
            symbol1 = st.text_input('輸入美股代號 1', key='stock1').upper()
            symbol2 = st.text_input('輸入美股代號 2', key='stock2').upper()
            symbol3 = st.text_input('輸入美股代號 3', key='stock3').upper()
            start_date_multi = st.date_input('開始日期', key='start_date_multi')
            end_date_multi = st.date_input('結束日期', key='end_date_multi')
            # 在 expander 之外執行相關程式碼
        if st.button('比較'):
            symbols = [s.upper() for s in [symbol1, symbol2, symbol3] if s]
            if symbols:
                stock_data = stock_data_vs(symbols, start_date_multi, end_date_multi)
                if stock_data is not None:
                    plot_trend_vs(stock_data, symbols)
                    plot_volume_chart(stock_data, symbols)
                else:
                    st.error('請輸入至少一隻美股')

elif market == '美國' and options == '內部資訊':
    symbol = st.text_input('輸入美股代號')
    head = int(st.number_input('輸入欲查詢資料筆數'))
    if st.button('查詢'):
        stock_insider_transactions(symbol,head)
        stock_insider_roster_holders(symbol)
        
elif market == '美國' and options == '機構買賣':
    symbol = st.text_input('輸入美股代號').upper()
    head = int(st.number_input('輸入查詢資料筆數'))
    if st.button('查詢'):
        stock_institutional_holders(symbol)
        stock_upgrades_downgrades(symbol, head)

elif market == '美國' and options == '近期相關消息' :
    st.subheader('近期相關新聞')
    symbol = st.text_input('輸入美股代號')
    if st.button('查詢'):
        display_news_table(symbol)
        display_news_links(symbol)

elif market == '台灣' and options == '大盤指數':
    period = st.selectbox('選擇時長',['年初至今','1年','3年','5年','10年','全部'])
    if period == '年初至今':
        period = 'ytd'
        plot_index_tw(period)
        plot_tw_asia(period)
        plot_pct_tw(period)
    elif period == '1年':
        period = '1y'
        plot_index_tw(period)
        plot_tw_asia(period)
        plot_pct_tw(period)
    elif period == '3年':
        period = '3y'
        plot_index_tw(period)
        plot_tw_asia(period)
        plot_pct_tw(period)
    elif period == '5年':
        period = '5y'
        plot_index_tw(period)
        plot_tw_asia(period)
        plot_pct_tw(period)
    elif period == '10年':
        period = '10y'
        plot_index_tw(period)
        plot_tw_asia(period)
        plot_pct_tw(period)
    elif period == '全部':
        period = 'max'
        plot_index_tw(period)
        plot_tw_asia(period)
        plot_pct_tw(period)
    with st.expander("顯示成份股"):
        st.write('新加坡海峽時報指數成份股')
        sti_symbol()
        st.write('恒生指數成份股')
        hsi_symbol()
        st.write('日經指數成份股')
        n225_symbol()
        st.write('深證指數成份股')
        shz_symbol()

elif market == '台灣' and options == '今日熱門' :
    twse_20()
    tpex_20()

elif market == '台灣' and options == '公司基本資訊' :
    select = st.selectbox('選擇市場',['上市','櫃檯'])
    if select == '上市':
        symbol = st.text_input('輸入台股上市代號')
        if st.button('查詢'):
            twse_info = twse_info(symbol)
            display_info(twse_info)
    elif select == '櫃檯':
        symbol = st.text_input('輸入台股櫃檯代號')
        if st.button('查詢'):
            tpex_info = tpex_info(symbol)
            display_info(tpex_info)

elif market == '台灣' and options == '公司財報查詢':
    select = st.selectbox('選擇市場',['上市','櫃檯'])
    select2 = st.selectbox('選擇查詢資訊',['年報','季報','月營收'])
    if select == '上市' and select2 == '年報':
        symbol = st.text_input('輸入台股上市代號')
        if st.button('查詢'):
            balance_sheet_twse, income_statement_twse, cash_flow_twse = financial_statements_twse(symbol)
            if balance_sheet_twse is not None:
                balance_twse(balance_sheet_twse)
            if income_statement_twse is not None:
                income_twse(income_statement_twse)
            if cash_flow_twse is not None:
                cashflow_twse(cash_flow_twse)
            else:
                st.error(f"無法獲取{symbol}-年報")
    elif select == '上市' and select2 == '季報':
        symbol = st.text_input('輸入台股上市代號')
        if st.button('查詢'):
            balance_sheet_quarterly_twse, income_statement_quarterly_twse, cash_flow_quarterly_twse = financial_statements_quarterly_twse(symbol)
            if balance_sheet_quarterly_twse is not None:
                balance_quarterly_twse(balance_sheet_quarterly_twse)
            if income_statement_quarterly_twse is not None:
                income_quarterly_twse(income_statement_quarterly_twse)
            if cash_flow_quarterly_twse is not None:
                cashflow_quarterly_twse(cash_flow_quarterly_twse)
            else:
                st.error(f"無法獲取{symbol}-季報")
    elif select == '上市' and select2 == '月營收':
        symbol = st.text_input('輸入台股上市代號')
        if st.button('查詢'):
            twse_month(symbol)
    elif select == '櫃檯' and select2 == '年報':
        symbol = st.text_input('輸入台股櫃檯代號')
        if st.button('查詢'):
            balance_sheet_tpex, income_statement_tpex, cash_flow_tpex = financial_statements_tpex(symbol)
            if balance_sheet_tpex is not None:
                balance_tpex(balance_sheet_tpex)
            if income_statement_tpex is not None:
                income_tpex(income_statement_tpex)
            if cash_flow_tpex is not None:
                cashflow_tpex(cash_flow_tpex)
            else:
                st.error(f"無法獲取{symbol}-年報")
    elif select == '櫃檯' and select2 == '季報':
        symbol = st.text_input('輸入台股櫃買代號')
        if st.button('查詢'):
            balance_sheet_quarterly_tpex, income_statement_quarterly_tpex, cash_flow_quarterly_tpex = financial_statements_quarterly_tpex(symbol)
            if balance_sheet_quarterly_tpex is not None:
                balance_quarterly_tpex(balance_sheet_quarterly_tpex)
            if income_statement_quarterly_tpex is not None:
                income_quarterly_tpex(income_statement_quarterly_tpex)
            if cash_flow_quarterly_tpex is not None :
                cashflow_quarterly_tpex(cash_flow_quarterly_tpex)
            else:
                st.error(f"無法獲取{symbol}-季報")
    elif select == '櫃檯' and select2 == '月營收':
        symbol = st.text_input('輸入台股櫃檯代號')
        if st.button('查詢'):
            tpex_month(symbol)

elif market == '台灣' and options == '交易數據':
    select = st.selectbox('選擇查市場',['上市','櫃檯'])
    select2 = st.selectbox('選擇查詢資訊',['個股','多股'])
    if select == '上市' and select2 == '個股':
        with st.expander("展開輸入參數"):
            symbol = st.text_input('輸台股上市代號', key='single_stock')
            start_date = st.date_input('開始日期', key='start_date')
            end_date = st.date_input('结束日期', key='end_date')
            mav_days = st.number_input('輸入MAV天數', min_value=15, max_value=360, value=15, step=1)  # 添加MAV天數的輸入
        if st.button('查詢'):
            twse_data = twse_data(symbol, start_date, end_date)
            if twse_data is not None:
                twse_candle(twse_data, mav_days)  # 將MAV天數傳遞給 plot_candle 函式
                twse_volume(twse_data)
                twse_trend(twse_data)
            else:
                st.error(f"無法獲取{symbol}交易數據or{symbol}為上櫃興櫃公司")
    elif select == '上市' and select2 == '多股':
        with st.expander("展開輸入參數"):
            symbol1 = st.text_input('輸台股上市代號 1', key='stock1')+ ".tw"
            symbol2 = st.text_input('輸台股上市代號 2', key='stock2')+ ".tw"
            symbol3 = st.text_input('輸台股上市代號 3', key='stock3')+ ".tw"
            start_date_multi = st.date_input('開始日期', key='start_date_multi')
            end_date_multi = st.date_input('結束日期', key='end_date_multi')
        if st.button('比較'):
            symbols = [s.upper() for s in [symbol1, symbol2, symbol3] if s]
            if symbols:
                twse_data = twse_data_vs(symbols, start_date_multi, end_date_multi)
                if twse_data is not None:
                    twse_trend_vs(twse_data, symbols)
                    twse_volume_chart(twse_data, symbols)
                else:
                    st.error('請輸入至少一隻台股上市')
    if select == '櫃檯' and select2 == '個股':
        with st.expander("展開輸入參數"):
            symbol = st.text_input('輸台股櫃檯代號', key='single_stock')
            start_date = st.date_input('開始日期', key='start_date')
            end_date = st.date_input('结束日期', key='end_date')
            mav_days = st.number_input('輸入MAV天數', min_value=15, max_value=360, value=15, step=1)  # 添加MAV天數的輸入
        if st.button('查詢'):
            tpex_data = tpex_data(symbol, start_date, end_date)
            if tpex_data is not None:
                tpex_candle(tpex_data, mav_days)  # 將MAV天數傳遞給 plot_candle 函式
                tpex_volume(tpex_data)
                tpex_trend(tpex_data)
            else:
                st.error(f"無法獲取{symbol}交易數據or{symbol}為上市公司")
    elif select == '櫃檯' and select2 == '多股':
        with st.expander("展開輸入參數"):
            symbol1 = st.text_input('輸台股櫃檯代號 1', key='stock1')+".two"
            symbol2 = st.text_input('輸台股櫃檯代號 2', key='stock2')+".two"
            symbol3 = st.text_input('輸台股櫃檯代號 3', key='stock3')+".two"
            start_date_multi = st.date_input('開始日期', key='start_date_multi')
            end_date_multi = st.date_input('結束日期', key='end_date_multi')
        if st.button('比較'):
            symbols = [s.upper() for s in [symbol1, symbol2, symbol3] if s]
            if symbols:
                tpex_data = tpex_data_vs(symbols, start_date_multi, end_date_multi)
                if tpex_data is not None:
                    tpex_trend_vs(tpex_data, symbols)
                    tpex_volume_chart(tpex_data, symbols)
                else:
                    st.error('請輸入至少一隻台股櫃檯')
