import streamlit as st 
from datetime import date 
import yfinance 
import pandas as pd
from plotly import graph_objs as go
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

START= "2015-01-01"
TODAY= date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="Stocks app",page_icon="bar")

st.title("Stock Prediction APP 📈 ")
 
stocks=("GOOG","AAPL","MSFT","GME","TSLA")
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

#split data into train and test

data_train=pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_test=pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

#scaler 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
train_arr=scaler.fit_transform(data_train)



#Load model
model=load_model('keras_model.h5')

#Testing data

input_data=scaler.fit_transform(data_test)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)

scales=scaler.scale_
scalef=1/scales[0]
y_predicted=y_predicted*scalef
y_test=y_test*scalef

#Final prediction graph
st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original price')
plt.plot(y_predicted,'r',label='Predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)

