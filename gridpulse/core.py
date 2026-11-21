import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

def demo(days=120,seed=42):
 r=np.random.default_rng(seed); t=pd.date_range('2025-01-01',periods=days*24,freq='h');h=t.hour.to_numpy();w=t.dayofweek.to_numpy();temp=20+6*np.sin(2*np.pi*h/24)+r.normal(0,2,len(t)); y=np.maximum(10,140+25*np.sin(2*np.pi*(h-8)/24)+15*(w<5)+2*np.maximum(temp-23,0)+r.normal(0,8,len(t)));return pd.DataFrame({'timestamp':t,'demand_kwh':y,'temperature_c':temp})
def validate(df):
 if not {'timestamp','demand_kwh','temperature_c'}.issubset(df):raise ValueError('Required columns: timestamp, demand_kwh, temperature_c')
 d=df[['timestamp','demand_kwh','temperature_c']].copy();d['timestamp']=pd.to_datetime(d.timestamp);d[['demand_kwh','temperature_c']]=d[['demand_kwh','temperature_c']].apply(pd.to_numeric);d=d.sort_values('timestamp').reset_index(drop=True)
 if len(d)<360 or d.isna().any().any() or (d.demand_kwh<0).any() or not d.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all():raise ValueError('Need 15+ days of continuous, nonnegative hourly readings without gaps or missing values')
 return d
FEATURES=['hour','weekday','temperature_c','lag24','lag168','mean24']
def features(d):
 d=d.copy();d['hour']=d.timestamp.dt.hour;d['weekday']=d.timestamp.dt.dayofweek;d['lag24']=d.demand_kwh.shift(24);d['lag168']=d.demand_kwh.shift(168);d['mean24']=d.demand_kwh.shift(1).rolling(24).mean();return d.dropna().reset_index(drop=True)
def fit(d):
 f=features(validate(d));cut=int(len(f)*.8);train,test=f.iloc[:cut],f.iloc[cut:];m=RandomForestRegressor(n_estimators=60,max_depth=12,min_samples_leaf=3,n_jobs=-1,random_state=42);m.fit(train[FEATURES],train.demand_kwh);p=m.predict(test[FEATURES]);out=test[['timestamp','demand_kwh']].copy();out['predicted_kwh']=p;return m,{'MAE':float(mean_absolute_error(test.demand_kwh,p)),'RMSE':float(np.sqrt(mean_squared_error(test.demand_kwh,p)))},out
def forecast(d):
 d=validate(d);f=features(d);m=RandomForestRegressor(n_estimators=60,max_depth=12,min_samples_leaf=3,n_jobs=-1,random_state=42).fit(f[FEATURES],f.demand_kwh);v=d.demand_kwh.tolist();last=d.iloc[-1];rows=[]
 for k in range(1,25):
  t=last.timestamp+pd.Timedelta(hours=k);x=pd.DataFrame([{'hour':t.hour,'weekday':t.dayofweek,'temperature_c':last.temperature_c,'lag24':v[-24],'lag168':v[-168],'mean24':np.mean(v[-24:])}]);y=max(0,float(m.predict(x[FEATURES])[0]));v.append(y);rows.append({'timestamp':t,'forecast_kwh':y})
 return pd.DataFrame(rows)
def anomalies(d,rate=.03):
 d=validate(d).copy();d['anomaly']=IsolationForest(n_estimators=100,contamination=rate,random_state=42).fit_predict(d[['demand_kwh','temperature_c']])==-1;return d
