import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import requests as res
import io
import folium
import pytz
import geopy
import re
from googletrans import Translator
from datetime import datetime
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

#美股區
  
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

#大盤指數
def plot_index(start_date='2014-01-01'):
    # Fetch historical data for S&P 500
    nasdaq_data = yf.download('^IXIC', start=start_date)
    nasdaq_100_data = yf.download('^NDX', start=start_date)
    sp500_data = yf.download('^GSPC', start=start_date)
    dji_data = yf.download('^DJI', start=start_date)
    Russell_2000_data = yf.download('^RUT', start=start_date)
    # Extract Close prices
    nasdaq_close = nasdaq_data['Close']
    nasdaq_100_close = nasdaq_100_data['Close']
    sp500_close = sp500_data['Close']  
    dji_close = dji_data['Close']
    Russell_2000_close = Russell_2000_data['Close']
    st.subheader('美股大盤＆中小企業市場走勢')
    # Create Plotly figure
    fig = go.Figure()   
    # Add trace for Close price
    fig.add_trace(go.Scatter(x=nasdaq_close.index, y=nasdaq_close.values, mode='lines', name='NASDAQ'))
    fig.add_trace(go.Scatter(x=nasdaq_100_close.index, y=nasdaq_100_close.values, mode='lines', name='NASDAQ-100'))
    fig.add_trace(go.Scatter(x=sp500_close.index, y=sp500_close.values, mode='lines', name='S&P 500'))
    fig.add_trace(go.Scatter(x=dji_close.index, y=dji_close.values, mode='lines', name='DJIA'))
    fig.add_trace(go.Scatter(x=Russell_2000_close.index, y=Russell_2000_close.values, mode='lines', name='Russell-2000'))
    # Update layout
    fig.update_layout(xaxis_title='Date', yaxis_title='Close Price')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_foreign(start_date='2014-01-01'):
    # Fetch historical data for S&P 500
    sp500_data = yf.download('^GSPC', start=start_date)
    sha_data = yf.download('000001.SS', start=start_date)
    twse_data = yf.download('^TWII', start=start_date)
    # Extract Close prices
    sp500_close = sp500_data['Close']
    sha_close = sha_data['Close']*0.1384
    twse_close = twse_data['Close']*0.0308
    st.subheader('美股大盤＆海外大盤走勢(換算美金2024/5/12)')
    # Create Plotly figure
    fig = go.Figure()   
    # Add trace for Close price
    fig.add_trace(go.Scatter(x=sp500_close.index, y=sp500_close.values, mode='lines', name='S&P 500'))
    fig.add_trace(go.Scatter(x=sha_close.index, y=sha_close.values, mode='lines', name='上證指數'))
    fig.add_trace(go.Scatter(x=twse_close.index, y=twse_close.values, mode='lines', name='加權指數'))
    # Update layout
    fig.update_layout(xaxis_title='Date', yaxis_title='Close Price')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#s&p 500 成分股
def sp500_dsymbol():
    url = res.get('https://zh.wikipedia.org/wiki/S%26P_500成份股列表')
    sp500 = pd.read_html(url.content, encoding='utf-8')
    st.write(sp500[0])

#nasdaq100成分股
def nasdaq_100symbol():
    url = res.get('https://zh.wikipedia.org/wiki/納斯達克100指數')
    nasdaq_100 = pd.read_html(url.content, encoding='utf-8')
    st.write(nasdaq_100[2])

#dji成分股
def dji_symbol():
    url = res.get('https://zh.wikipedia.org/zh-tw/道琼斯工业平均指数')
    dji = pd.read_html(url.content, encoding='utf-8')
    st.write(dji[2])

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
def plot_trend_vs(stock_data,symbols):
    fig = go.Figure()
    for symbol in symbols:
        if symbol in stock_data.columns.get_level_values(1):
            df = stock_data.xs(symbol, level=1, axis=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name=symbol))
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

#公司盈利
@st.cache_data
def stock_earnings_date(symbol):
    translation = {
        #'Earnings Date':'日期',
        'EPS Estimate':'每股盈利預估',
        'Reported EPS':'實際每股盈利'
        }
    ticker = yf.Ticker(symbol)  # 使用參數中的 symbol 創建 Ticker 物件
    earnings_dates = ticker.earnings_dates
    earnings_dates = earnings_dates.rename(columns=translation)
    st.subheader(f'{symbol}-盈利資訊')  # 使用函數參數中的 symbol
    earnings_dates = st.write(earnings_dates)
    return earnings_dates

#股息/股票分割
@st.cache_data
def stock_actions(symbol, start_date, end_date):
    translation = {
        'Date':'日期',
        'Dividends':'股息',
        'Stock Splits':'股票拆分'
        }
    try:
        ticker = yf.Ticker(symbol)  # 使用參數中的 symbol 創建 Ticker 物件
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.min.time())
        start_date = pytz.utc.localize(start_date)
        end_date = pytz.utc.localize(end_date)
        actions = ticker.actions[start_date:end_date]  # 指定日期範圍
        actions = actions.rename(columns=translation)
        st.subheader(f'{symbol}-股息/股票分割')
        st.write(actions)
        return actions
    except Exception as e:
        st.error(f"無法獲取{symbol}-股息/股票分割：{str(e)}")
        return None
   
#股權架構
@st.cache_data
def stock_major_holder(symbol):
    translation_columns = {'Breakdown':'持股情況','Value':'數值'}
    translation_index = {
        'insidersPercentHeld': '內部持股百分比',
        'institutionsPercentHeld': '機構持股百分比',
        'institutionsFloatPercentHeld': '流通股機構持股百分比',
        'institutionsCount': '內部總持有股份',
        'nstitutionsCount': '機構持股數量'
        }
    ticker = yf.Ticker(symbol)  # 使用參數中的 symbol 創建 Ticker 物件
    major_holders = ticker.major_holders
    major_holders = major_holders.rename(columns=translation_columns, index=translation_index)  
    #轉換成百分比
    columns = ['內部持股百分比', '機構持股百分比', '流通股機構持股百分比']
    for col in columns:
        if col in major_holders.index:
            major_holders.at[col, '數值'] = pd.to_numeric(major_holders.at[col, '數值'], errors='coerce')  # 将非数字转换为 NaN
            major_holders.at[col, '數值'] = f"{major_holders.at[col, '數值']:.2%}" if pd.notna(major_holders.at[col, '數值']) else None
        #千分位表示
        major_holders['數值'] = major_holders['數值'].apply(lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) and x >= 1000 else x)
    st.subheader(f'{symbol}-股權架構')
    st.write(major_holders)
    return major_holders

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
                
                st.subheader(f'{symbol}-內部交易')
                st.write(insider_transactions)
            else:
                st.warning("請輸入大於 0 的數字")
        else:
            st.error(f"無法獲取{symbol}內部交易數據或數據為空")
    except Exception as e:
        st.error(f"獲取{symbol}內部交易數據時出錯：{str(e)}")

# 內部購買
@st.cache_data
def stock_insider_purchases(symbol):
    ticker = yf.Ticker(symbol)
    insider_purchases = ticker.insider_purchases
    #轉換成百分比
    columns = ['% Net Shares Purchased (Sold)', '% Buy Shares', '% Sell Shares']
    for col in columns:
        if col in insider_purchases.index:
            insider_purchases.at[col, 'Shares'] = pd.to_numeric(insider_purchases.at[col, 'Shares'], errors='coerce')  # 将非数字转换为 NaN
            insider_purchases.at[col, 'Shares'] = f"{insider_purchases.at[col, 'Shares']:.2%}" if pd.notna(insider_purchases.at[col, 'Shares']) else None
        #千分位表示
        insider_purchases['Shares'] = insider_purchases['Shares'].apply(lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) and x >= 1000 else x)
    trans_column = insider_purchases['Insider Purchases Last 6m'] 
    translator = Translator()  # 初始化 Translator
    translated_texts = []
    for text in trans_column:
        translation = translator.translate(text, dest='zh-tw').text
        translated_texts.append(translation)
    insider_purchases['Insider Purchases Last 6m'] = translated_texts
    st.subheader(f'{symbol}-內部購買')
    st.write(insider_purchases)
    return insider_purchases

#內部持股
@st.cache_data
def stock_insider_roster_holders(symbol):
    translation = {
        'Name':' ',
        'Position':'職位',
        'Most Recent Transaction':'最近的交易',
        'Latest Transaction Date':'最新交易日期',
        'Shares Owned Indirectly':'間接擁有股份',
        'Position Indirect Date':'間接持股日期',
        'Shares Owned Directly':'直接擁有股份',
        'Position Direct Date':'直接持股日期'
        }
    symbol = symbol
    symbol = yf.Ticker(symbol)
    insider_roster_holders = symbol.insider_roster_holders
    insider_roster_holders = insider_roster_holders.rename(columns=translation)
    insider_roster_holders = insider_roster_holders.drop(columns='URL')
    insider_roster_holders = st.write(insider_roster_holders)
    return insider_roster_holders

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
                st.subheader(f'機構買賣{symbol}數據')
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

def plot_index_tw(start_date='2014-01-01'):
    # Fetch historical data for S&P 500
    twse_data = yf.download('^TWII', start=start_date)
    tpex_data = yf.download('^TWOII', start=start_date)
    tw50_data = yf.download('^TSE50', start=start_date)
    # Extract Close prices
    twse_close = twse_data['Close']
    tpex_close = tpex_data['Close']
    st.subheader('上市＆櫃檯走勢')
    # Create Plotly figure
    fig = go.Figure()   
    # Add trace for Close price
    fig.add_trace(go.Scatter(x=twse_close.index, y=twse_close.values, mode='lines', name='加權指數'))
    fig.add_trace(go.Scatter(x=tpex_close.index, y=tpex_close.values, mode='lines', name='櫃檯指數'))
    # Update layout
    fig.update_layout(xaxis_title='Date', yaxis_title='Close Price')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_foreign_asia(start_date='2014-01-01'):
    # Fetch historical data for S&P 500
    sha_data = yf.download('000001.SS', start=start_date)
    twse_data = yf.download('^TWII', start=start_date)
    jp_data = yf.download('^N225', start=start_date)
    hk_data = yf.download('^HSI', start=start_date)
    kr_data = yf.download('^KS11', start=start_date)
    sin_data = yf.download('^STI', start=start_date)
    # Extract Close prices
    sha_close = sha_data['Close']*4.4927
    twse_close = twse_data['Close']
    jp_close = jp_data['Close']*0.2084
    hk_close = hk_data['Close']*4.1549
    kr_close = kr_data['Close']*0.0237
    sin_close = sin_data['Close']*23.9665
    st.subheader('台股大盤＆亞洲大盤走勢(換算新台幣2024/5/12)')
    # Create Plotly figure
    fig = go.Figure()   
    # Add trace for Close price
    fig.add_trace(go.Scatter(x=sha_close.index, y=sha_close.values, mode='lines', name='上證指數'))
    fig.add_trace(go.Scatter(x=twse_close.index, y=twse_close.values, mode='lines', name='加權指數'))
    fig.add_trace(go.Scatter(x=jp_close.index, y=jp_close.values, mode='lines', name='日經指數'))
    fig.add_trace(go.Scatter(x=hk_close.index, y=hk_close.values, mode='lines', name='恒生指數'))
    fig.add_trace(go.Scatter(x=kr_close.index, y=kr_close.values, mode='lines', name='韓國綜合股價指數'))
    fig.add_trace(go.Scatter(x=sin_close.index, y=sin_close.values, mode='lines', name='新加坡海峽時報指數'))
    # Update layout
    fig.update_layout(xaxis_title='Date', yaxis_title='Close Price')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

#sti成分股
def sti_symbol():
    url = res.get('https://zh.wikipedia.org/wiki/新加坡海峽時報指數')
    sti = pd.read_html(url.content, encoding='utf-8')
    st.write(sti[1])

#成交量前二十名證券
@st.cache_data
def twse_20():
    url = res.get('https://openapi.twse.com.tw/v1/exchangeReport/MI_INDEX20')
    url = pd.read_json(url.text)
    st.subheader('今日上市交易量前20名')
    st.write(url)

@st.cache_data
def tpex_20():
    url = res.get('https://www.tpex.org.tw/openapi/v1/tpex_volume_rank')
    url = pd.read_json(url.text).head(20)
    st.subheader('今日櫃買交易量前20名')
    st.write(url)

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
            fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name=symbol))
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
            fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name=symbol))
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
options = st.sidebar.selectbox('選擇功能', ['大盤指數','今日熱門','公司基本資訊','公司財報查詢','交易數據','股票資訊','內部資訊','機構買賣','近期相關消息'])
st.sidebar.markdown('''
    免責聲明：        
    1. K 線圖請以美股角度來觀看（        
        - 美股: 綠漲、紅跌）        
        - 台股: 綠跌、紅漲           
    2. 本平台僅適用於數據搜尋，不建議任何投資行為
    3. 有些數據僅限美股，台股尚未支援  
''')

if market == '美國' and options == '大盤指數':
    plot_index(start_date='2014-01-01')
    with st.expander("顯示成分股"):
        st.write('S&P500成份股')
        sp500_dsymbol()
        st.write('NASDAQ100成份股')
        nasdaq_100symbol()
        st.write('道瓊工業成份股')
        dji_symbol()
    plot_foreign(start_date='2014-01-01')
    st.markdown("[詳細名詞解釋](https://www.oanda.com/bvi-ft/lab-education/indices/us-4index/)")

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
            symbol4 = st.text_input('輸入美股代號 4', key='stock4').upper()
            symbol5 = st.text_input('輸入美股代號 5', key='stock5').upper()
            symbol6 = st.text_input('輸入美股代號 6', key='stock6').upper()
            start_date_multi = st.date_input('開始日期', key='start_date_multi')
            end_date_multi = st.date_input('結束日期', key='end_date_multi')
            # 在 expander 之外執行相關程式碼
        if st.button('比較'):
            symbols = [s.upper() for s in [symbol1, symbol2, symbol3, symbol4, symbol5, symbol6] if s]
            if symbols:
                stock_data = stock_data_vs(symbols, start_date_multi, end_date_multi)
                if stock_data is not None:
                    plot_trend_vs(stock_data, symbols)
                    plot_volume_chart(stock_data, symbols)
                else:
                    st.error('請輸入至少一隻美股')

elif market == '美國' and options == '股票資訊':
    with st.expander("展開輸入參數"):
        symbol = st.text_input('輸入美股代號').upper()
        start_date = st.date_input('開始日期')
        end_date = st.date_input('結束日期' ,key='end_date')
    if st.button('查詢'):
        stock_earnings_date(symbol)
        stock_actions(symbol,start_date,end_date)
        stock_major_holder(symbol)
        stock_institutional_holders(symbol)

elif market == '美國' and options == '內部資訊':
    select = st.selectbox('選擇查詢資訊',['內部交易','內部購買','內部持股'])
    if select == '內部交易':
        with st.expander("展開輸入參數"):
            symbol = st.text_input('輸入美股代號')
            head = int(st.number_input('輸入欲查詢資料筆數'))
        if st.button('查詢'):
            stock_insider_transactions(symbol,head) 
    elif select == '內部購買':
        symbol = st.text_input('輸入美股代號')
        if st.button('查詢'):
            stock_insider_purchases(symbol)
    elif select == '內部持股':
        symbol = st.text_input('輸入美股代號')
        if st.button('查詢'):
            stock_insider_roster_holders(symbol)

elif market == '美國' and options == '機構買賣':
    with st.expander("展開輸入參數"):
        symbol = st.text_input('輸入美股代號').upper()
        head = int(st.number_input('輸入查詢資料筆數'))
    if st.button('查詢'):
        stock_upgrades_downgrades(symbol, head)

elif market == '美國' and options == '近期相關消息' :
    st.subheader('近期相關新聞')
    symbol = st.text_input('輸入美股代號')
    if st.button('查詢'):
        display_news_table(symbol)
        display_news_links(symbol)

elif market == '台灣' and options == '大盤指數':
     plot_index_tw(start_date='2014-01-01')
     plot_foreign_asia(start_date='2014-01-01')
     with st.expander("顯示成分股"):
         st.write('新加坡海峽時報指數成份股')
         sti_symbol()

elif market == '台灣' and options == '今日熱門' :
    twse_20()
    tpex_20()

elif market == '台灣' and options == '公司基本資訊' :
    select = st.selectbox('選擇市場',['上市','櫃買'])
    if select == '上市':
        symbol = st.text_input('輸入台股上市代號')
        if st.button('查詢'):
            twse_info = twse_info(symbol)
            display_info(twse_info)
    elif select == '櫃買':
        symbol = st.text_input('輸入台股櫃買代號')
        if st.button('查詢'):
            tpex_info = tpex_info(symbol)
            display_info(tpex_info)

elif market == '台灣' and options == '公司財報查詢':
    select = st.selectbox('選擇市場',['上市','櫃買'])
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
    elif select == '櫃買' and select2 == '年報':
        symbol = st.text_input('輸入台股櫃買代號')
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
    elif select == '櫃買' and select2 == '季報':
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
    elif select == '櫃買' and select2 == '月營收':
        symbol = st.text_input('輸入台股櫃買代號')
        if st.button('查詢'):
            tpex_month(symbol)

elif market == '台灣' and options == '交易數據':
    select = st.selectbox('選擇查市場',['上市','櫃買'])
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
            symbol4 = st.text_input('輸台股上市代號 4', key='stock4')+ ".tw"
            symbol5 = st.text_input('輸台股上市代號 5', key='stock5')+ ".tw"
            symbol6 = st.text_input('輸台股上市代號 6', key='stock6')+ ".tw"
            start_date_multi = st.date_input('開始日期', key='start_date_multi')
            end_date_multi = st.date_input('結束日期', key='end_date_multi')
        if st.button('比較'):
            symbols = [s.upper() for s in [symbol1, symbol2, symbol3, symbol4, symbol5, symbol6] if s]
            if symbols:
                twse_data = twse_data_vs(symbols, start_date_multi, end_date_multi)
                if twse_data is not None:
                    twse_trend_vs(twse_data, symbols)
                    twse_volume_chart(twse_data, symbols)
                else:
                    st.error('請輸入至少一隻台股上市')
    if select == '櫃買' and select2 == '個股':
        with st.expander("展開輸入參數"):
            symbol = st.text_input('輸台股櫃買代號', key='single_stock')
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
    elif select == '櫃買' and select2 == '多股':
        with st.expander("展開輸入參數"):
            symbol1 = st.text_input('輸台股櫃買代號 1', key='stock1')+".two"
            symbol2 = st.text_input('輸台股櫃買代號 2', key='stock2')+".two"
            symbol3 = st.text_input('輸台股櫃買代號 3', key='stock3')+".two"
            symbol4 = st.text_input('輸台股櫃買代號 4', key='stock4')+".two"
            symbol5 = st.text_input('輸台股櫃買代號 5', key='stock5')+".two"
            symbol6 = st.text_input('輸台股櫃買代號 6', key='stock6')+".two"
            start_date_multi = st.date_input('開始日期', key='start_date_multi')
            end_date_multi = st.date_input('結束日期', key='end_date_multi')
        if st.button('比較'):
            symbols = [s.upper() for s in [symbol1, symbol2, symbol3, symbol4, symbol5, symbol6] if s]
            if symbols:
                tpex_data = tpex_data_vs(symbols, start_date_multi, end_date_multi)
                if tpex_data is not None:
                    tpex_trend_vs(tpex_data, symbols)
                    tpex_volume_chart(tpex_data, symbols)
                else:
                    st.error('請輸入至少一隻台股櫃買')
