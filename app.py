import streamlit as st
import pandas as pd
import plotly.express as px
from gridpulse.core import demo,validate,fit,forecast,anomalies
st.set_page_config(page_title='GridPulse AI | ZubairLabs',page_icon='⚡',layout='wide')
st.title('⚡ GridPulse AI')
st.caption('Energy Demand Forecasting & Consumption Intelligence · ZubairLabs')
st.info('Portfolio prototype: synthetic data by default. Predictions are not validated for real-world energy operations. Forecast assumes unchanged temperature.')
with st.sidebar:
 st.header('Settings');file=st.file_uploader('Upload hourly CSV',type='csv');days=st.slider('Demo history (days)',30,365,120,15);rate=st.number_input('Illustrative price per kWh',min_value=0.,value=.18,step=.01);sensitivity=st.slider('Anomaly fraction',.01,.15,.03,.01)
try:
 d=validate(pd.read_csv(file) if file else demo(days));model,metrics,holdout=fit(d);future=forecast(d);flags=anomalies(d,sensitivity)
except Exception as e:st.error(str(e));st.stop()
t1,t2,t3,t4=st.tabs(['📊 Overview','🔮 Forecast','🚨 Anomalies','🧪 Evaluation & Data'])
with t1:
 a,b,c,e=st.columns(4);a.metric('Total consumption',f'{d.demand_kwh.sum():,.0f} kWh');b.metric('Average hourly',f'{d.demand_kwh.mean():.1f} kWh');c.metric('Peak hourly',f'{d.demand_kwh.max():.1f} kWh');e.metric('Illustrative cost',f'{d.demand_kwh.sum()*rate:,.0f}')
 st.plotly_chart(px.line(d.tail(336),x='timestamp',y='demand_kwh',title='Last 14 days of hourly consumption'),use_container_width=True)
 hourly=d.assign(hour=d.timestamp.dt.hour).groupby('hour',as_index=False).demand_kwh.mean();st.plotly_chart(px.bar(hourly,x='hour',y='demand_kwh',title='Average consumption by hour'),use_container_width=True)
with t2:
 st.subheader('Experimental 24-hour recursive forecast');st.caption('Last measured temperature held constant. Multi-step errors may accumulate.');st.plotly_chart(px.line(future,x='timestamp',y='forecast_kwh',markers=True),use_container_width=True);st.metric('Forecast total',f'{future.forecast_kwh.sum():,.0f} kWh');st.download_button('Download forecast',future.to_csv(index=False),'forecast.csv','text/csv')
with t3:
 st.metric('Flagged observations',int(flags.anomaly.sum()));st.caption('Isolation Forest flags statistical outliers, not confirmed faults.');st.plotly_chart(px.scatter(flags.tail(720),x='timestamp',y='demand_kwh',color='anomaly'),use_container_width=True);st.dataframe(flags[flags.anomaly].tail(100));st.download_button('Download flags',flags.to_csv(index=False),'anomalies.csv','text/csv')
with t4:
 st.subheader('Chronological 80/20 holdout');st.caption('Metrics evaluate one-step predictions with observed historical lags, not the recursive 24-hour forecast.');a,b=st.columns(2);a.metric('MAE',f"{metrics['MAE']:.2f} kWh");b.metric('RMSE',f"{metrics['RMSE']:.2f} kWh");st.plotly_chart(px.line(holdout.tail(240),x='timestamp',y=['demand_kwh','predicted_kwh']),use_container_width=True);st.dataframe(d.tail(500));st.download_button('Download readings',d.to_csv(index=False),'readings.csv','text/csv')
st.caption('ZubairLabs · No paid APIs · Synthetic demonstration')
