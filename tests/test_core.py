import pytest
from gridpulse.core import demo,validate,features,fit,forecast,anomalies
def test_demo():assert len(demo(20))==480
def test_determinism():assert demo(20).equals(demo(20))
def test_validation():assert len(validate(demo(20)))==480
def test_negative():
 d=demo(20);d.loc[0,'demand_kwh']=-1
 with pytest.raises(ValueError):validate(d)
def test_lag():
 d=demo(20);assert features(d).iloc[0].lag24==d.iloc[144].demand_kwh
def test_fit():
 _,m,p=fit(demo(20));assert m['MAE']>=0 and len(p)>0
def test_forecast():assert len(forecast(demo(20)))==24
def test_anomalies():assert anomalies(demo(20)).anomaly.any()
