from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, HiveContext
import os
from statistics import median
import pandas as pd
from pyspark.sql.types import *

from pyspark.sql.functions import pandas_udf, PandasUDFType
from prophet import Prophet

## DATA PREPROCESSING ##
os.environ['PYSPARK_PYTHON'] = './environment/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = os.popen('which python').read()
sparkSession = (SparkSession.builder.appName('ClusterModelApp')
                .master('yarn')
                .config('spark.yarn.dist.archives', "bigdata.tar.gz#environment")
                .config("spark.executor.instances", "4")
                .enableHiveSupport()
                .getOrCreate())

df_train = sparkSession.sql('SELECT * FROM cluster_grouped')
df_oil = sparkSession.sql('SELECT * FROM oil')

df_oil = df_oil.withColumnRenamed("dcoilwtico", "oil")
df_oil = df_oil.withColumn("oil",df_oil.oil.cast('float'))

values = df_oil.select('oil').to_pandas_on_spark()['oil'].to_numpy()
median = median(values)

df_train = df_train.join(df_oil, df_oil["date_oil"] ==  df_train["date_"], "left")
df_train = df_train.drop("date_oil")

df_train = df_train.fillna(value=median,subset=["oil"])
df_train = df_train.withColumnRenamed("date_", "ds").withColumnRenamed("sales","y")

df_holidays_events = sparkSession.sql('SELECT * FROM holidays_events')
df_holidays_events = df_holidays_events.filter(df_holidays_events['locale']=="National").filter(df_holidays_events['transferred']=="False").select('date_',"description")
df_holidays_events = df_holidays_events.withColumnRenamed("date_","ds")
df_holidays_events = df_holidays_events.withColumnRenamed("description","holiday")

df = df_train.withColumnRenamed('ds','date').groupby("date").sum("y").sort('date')
df_holidays_events = df_holidays_events.sort('ds')
joined = df_holidays_events.join(df, df.date == df_holidays_events.ds, 'left')

df_holidays_events=df_holidays_events.toPandas()
df_holidays_events['lower_window']  = 0
df_holidays_events['upper_window']  = 0

payments = pd.date_range(start='2012-01-01', end='2017-12-31', freq='SM')
payments = pd.DataFrame({'ds': payments, "holiday": "Salary", 'lower_window': 0, 'upper_window': 2})
df_holidays_events = df_holidays_events.append(payments)

df_oil = df_oil.toPandas()
df_oil = df_oil.fillna(median)
df_oil['date_oil'] = pd.to_datetime(df_oil['date_oil'])
df_oil = df_oil.rename(columns={"date_oil":'ds'})

df_train.show()
## MODEL ##

def get_oil():
  return df_oil

def get_holidays():
  return df_holidays_events

def getMedian():
    return median

result_schema = StructType([
                  StructField('ds', TimestampType()),
                  StructField('cluster', IntegerType()),
                  StructField('y', DoubleType()),
                  StructField('yhat', DoubleType()),
                  StructField('yhat_upper', DoubleType()),
                  StructField('yhat_lower', DoubleType())
])



@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def forecast_sales(store_pd):
  #holidays generation
  h = get_holidays()
  model = Prophet(
      interval_width=0.95,
      seasonality_mode= 'multiplicative',
      daily_seasonality=False,
      weekly_seasonality=True,
      yearly_seasonality=True,
      holidays=h,
      changepoint_prior_scale=0.5,
      seasonality_prior_scale=5.0,
  )
  model.add_regressor('oil',mode='multiplicative')
  model.add_regressor('promotion',mode='multiplicative')

  store_pd['ds'] = pd.to_datetime(store_pd['ds'])
  date_split = pd.to_datetime('2016-12-31')
  train_set = store_pd[store_pd['ds'] <= date_split]

  model.fit(train_set)
    
  future_pd = model.make_future_dataframe(periods=365)
  df_oil = get_oil()
  future_pd=pd.merge(left=future_pd,right=store_pd,on='ds',how='left')
  future_pd=future_pd[['ds','promotion']]
  future_pd = pd.merge(left=future_pd, right=df_oil, on='ds', how='left')
  future_pd = future_pd.fillna(getMedian()) 
    
    
  forecast_pd = model.predict(future_pd)
  f_pd = forecast_pd[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].set_index('ds')
  st_pd = train_set[['ds', 'cluster', 'y']].set_index('ds')
  
  result_pd = f_pd.join(st_pd, how='left')
  result_pd.reset_index(level=0, inplace=True)
  result_pd['cluster'] = train_set['cluster'].iloc[0]
  result_pd = result_pd[['ds', 'cluster', 'y', 'yhat', 'yhat_upper', 'yhat_lower']]
  return result_pd

## apply the model on the various clusters and save the results on hdfs##
df_train.groupby('cluster').apply(forecast_sales).write.csv(path='models/clusters_models.csv', header=True, sep=',')

