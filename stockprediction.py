import streamlit as st 
from datetime import date 
import yfinance 
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

START= "2015-01-01"
TODAY= date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction APP")

stocks=("GOOG","AAPL","MSFT","GME")
selected_stocks=st.selectbox("select stock",stocks)

n_years=st.slider("Years of prediction:", 1, 4)
period=n_years*365

@st.cache_data
def load_data(stock):
    data=yfinance.download(stock,START,TODAY)
    data.reset_index(inplace=True)
    return data
data=load_data(selected_stocks)



st.subheader("Stock data")
st.write(data.tail())

#plot stock data
fig1=go.Figure()
fig1.add_trace(go.Scatter(x=data["Date"],y=data["Open"],name="stock_open"))
fig1.add_trace(go.Scatter(x=data["Date"],y=data["Close"],name="stock_open"))
fig1.layout.update(title_text="Time Series data",   xaxis_rangeslider_visible=True)
st.plotly_chart(fig1) 

#Predict future prices
data=data.reset_index()
df_train=data[["Date","Close"]]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})


m=Prophet(daily_seasonality=True)
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)

#Plot forecast
st.subheader("Forecast data")
fig2=plot_plotly(m,forecast)
st.plotly_chart(fig2)

