import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import requests as res
import io
import folium
import pytz
from googletrans import Translator
from datetime import datetime
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

#美股區

#今日熱門
@st.cache_data
def hot_stock():
    url = "https://finance.yahoo.com/most-active/"
    hot_stock = res.get(url)
    f = io.StringIO(hot_stock.text)
    hot_stock_df = pd.read_html(f)
    hot_stock_df = hot_stock_df[0]
    hot_stock_df = hot_stock_df.drop(columns=['PE Ratio (TTM)', '52 Week Range'])
    st.subheader("今日交易量前25名")
    st.table(hot_stock_df)
    return hot_stock_df

#今日上漲
@st.cache_data
def gainers_stock():
    url = "https://finance.yahoo.com/gainers"
    gainers_stock = res.get(url)
    f = io.StringIO(gainers_stock.text)
    gainers_stock_df = pd.read_html(f)
    gainers_stock_df = gainers_stock_df[0]
    gainers_stock_df = gainers_stock_df.drop(columns=['PE Ratio (TTM)', '52 Week Range'])
    st.subheader("今日上漲前25名")
    st.table(gainers_stock_df)
    return gainers_stock_df

#今日下跌
@st.cache_data
def loser_stock():
    url = "https://finance.yahoo.com/losers/"
    loser_stock = res.get(url)
    f = io.StringIO(loser_stock.text)
    loser_stock_df = pd.read_html(f)
    loser_stock_df = loser_stock_df[0]
    loser_stock_df = loser_stock_df.drop(columns=['PE Ratio (TTM)', '52 Week Range'])
    st.subheader("今日下跌前25名")
    st.table(loser_stock_df)
    return loser_stock_df

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
        st.table(balance_sheet)

@st.cache_data
def income(income_statement):
    if income_statement is not None:
        st.subheader(f"{symbol}-綜合損益表(年度)")
        st.table(income_statement)

@st.cache_data
def cashflow(cash_flow):
    if cash_flow is not None:
        st.subheader(f"{symbol}-現金流量表(年度)")
        st.table(cash_flow)

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
        st.table(balance_sheet_quarterly)

@st.cache_data
def income_quarterly(income_statement_quarterly):
    if income_statement_quarterly is not None:
        st.subheader(f"{symbol}-綜合損益表(季度)")
        st.table(income_statement_quarterly)

@st.cache_data
def cashflow_quarterly(cash_flow_quarterly):
    if cash_flow_quarterly is not None:
        st.subheader(f"{symbol}-現金流量表(季度)")
        st.table(cash_flow_quarterly)

#獲取歷史交易數據
@st.cache_data
def stock_data(symbol,start_date,end_date):
    try:
        stock_data = yf.download(symbol,start=start_date,end=end_date)
        st.subheader('交易數據')
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
#@st.cache_data
def stock_earnings_dates(symbol):
    translation = {
        'Earnings Date':'日期',
        'EPS Estimate':'每股盈利預估',
        'Reported EPS':'實際每股盈利'
        }
    ticker = yf.Ticker(symbol)  # 使用參數中的 symbol 創建 Ticker 物件
    earnings_dates = ticker.earnings_dates
    earnings_dates = earnings_dates.rename(columns=translation)
    st.subheader(f'{symbol}-盈利資訊')  # 使用函數參數中的 symbol
    earnings_dates = st.table(earnings_dates)
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
        st.table(actions)
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
    st.table(major_holders)
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
    st.table(institutional_holders)
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
                st.table(insider_transactions)
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
    st.table(insider_purchases)
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
        #'Position Indirect Date':'間接持股日期',
        'Shares Owned Directly':'直接擁有股份',
        'Position Direct Date':'直接持股日期'
        }
    symbol = symbol
    symbol = yf.Ticker(symbol)
    insider_roster_holders = symbol.insider_roster_holders
    insider_roster_holders = insider_roster_holders.rename(columns=translation)
    insider_roster_holders = insider_roster_holders.drop(columns='URL')
    insider_roster_holders = st.table(insider_roster_holders)
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
                st.table(upgrades_downgrade)
            else:
                st.warning("請輸入大於 0 的數字")
        else:
            st.error(f"無法獲取持有{symbol}的機構買賣數據或數據為空")
    except Exception as e:
        st.error(f"獲取持有{symbol}的機構買賣數據時出錯：{str(e)}")

#相關新聞
#@st.cache_data
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
    st.table(news_df)

#@st.cache_data   
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


#streamlit版面配置
st.markdown("<h1 style='text-align: center; color: rainbow;'>StockMarket</h1>", unsafe_allow_html=True)
st.header(' ',divider="rainbow")
st.sidebar.title('StockMarket')

market = st.sidebar.selectbox('選擇市場', ['美國','台灣'])
options = st.sidebar.selectbox('選擇功能', ['今日熱門','公司基本資訊','公司財報查詢','交易數據','股票資訊','內部資訊','機構買賣','近期相關消息'])
text = st.sidebar.markdown('''
    免責聲明：        
    1. K 線圖請以美股角度來觀看（        
        - 美股: 綠漲、紅跌）        
        - 台股: 綠跌、紅漲           
    2. 本平台僅適用於數據搜尋，不建議任何投資行為
    3. 有些數據僅限美股，台股尚未支援 
''')

if market == '美國' and options == '今日熱門':
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
        symbol = st.text_input('輸入美股代號', key='single_stock').upper()
        start_date = st.date_input('開始日期', key='start_date')
        end_date = st.date_input('结束日期', key='end_date')
        mav_days = st.number_input('輸入MAV天數', min_value=15, max_value=60, value=15, step=1)  # 添加MAV天數的輸入
        if st.button('查詢'):
            stock_data = stock_data(symbol, start_date, end_date)
            if stock_data is not None:
                plot_candle(stock_data, mav_days)  # 將MAV天數傳遞給 plot_candle 函式
                plot_volume(stock_data)
                plot_trend(stock_data)
            else:
                st.error(f"無法獲取{symbol}交易數據")
    elif select == '多股':
        symbol1 = st.text_input('輸入美股代號 1', key='stock1').upper()
        symbol2 = st.text_input('輸入美股代號 2', key='stock2').upper()
        symbol3 = st.text_input('輸入美股代號 3', key='stock3').upper()
        symbol4 = st.text_input('輸入美股代號 4', key='stock4').upper()
        symbol5 = st.text_input('輸入美股代號 5', key='stock5').upper()
        symbol6 = st.text_input('輸入美股代號 6', key='stock6').upper()
        start_date_multi = st.date_input('開始日期', key='start_date_multi')
        end_date_multi = st.date_input('結束日期', key='end_date_multi')
        if st.button('比較'):
            symbols = [s.upper() for s in [symbol1, symbol2, symbol3, symbol4, symbol5, symbol6] if s]
            if symbols:
                stock_data = stock_data_vs(symbols, start_date_multi, end_date_multi)
                if stock_data is not None:
                    plot_trend_vs(stock_data, symbols)
                    plot_volume_chart(stock_data, symbols)
                else:
                    st.error('請輸入至少一個股票')

elif market == '美國' and options == '股票資訊':
    symbol = st.text_input('輸入股票').upper()
    start_date = st.date_input('開始日期')
    end_date = st.date_input('結束日期' ,key='end_date')
    if st.button('查詢'):
        stock_earnings_dates(symbol)
        stock_actions(symbol,start_date,end_date)
        stock_major_holder(symbol)
        stock_institutional_holders(symbol)

elif market == '美國' and options == '內部資訊':
    select = st.selectbox('選擇欲查詢資訊',['內部交易','內部購買','內部持股'])
    if select == '內部交易':
        symbol = st.text_input('輸入股票')
        head = int(st.number_input('輸入欲查詢資料筆數'))
        if st.button('查詢'):
            stock_insider_transactions(symbol,head) 
    elif select == '內部購買':
        symbol = st.text_input('輸入股票')
        if st.button('查詢'):
            stock_insider_purchases(symbol)
    elif select == '內部持股':
        symbol = st.text_input('輸入股票')
        if st.button('查詢'):
            stock_insider_roster_holders(symbol)

elif market == '美國' and options == '機構買賣':
    symbol = st.text_input('輸入股票').upper()
    head = int(st.number_input('輸入欲查詢資料筆數'))
    if st.button('查詢'):
        stock_upgrades_downgrades(symbol, head)

elif market == '美國' and options == '近期相關消息' :
    st.subheader('近期相關新聞')
    symbol = st.text_input('輸入股票')
    if st.button('查詢'):
        display_news_table(symbol)
        display_news_links(symbol)
