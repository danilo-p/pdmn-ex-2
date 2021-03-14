import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType

spark = SparkSession\
    .builder\
    .appName("TP2 - 3")\
    .getOrCreate()


# path = './datasets_sample/covid19/{}_Coronavirus_Tweets.CSV'
path = 'hdfs://compute1:9000/datasets/covid19/{}_Coronavirus_Tweets.CSV'


def read_dataset_for_date(date):
    return spark.read\
        .option("header", True)\
        .csv(path.format(date))


# dates = ['2020-03-31', '2020-04-01', '2020-04-02', '2020-04-05', '2020-04-06', '2020-04-07', '2020-04-10',
#          '2020-04-12', '2020-04-14', '2020-04-16', '2020-04-20', '2020-04-26', '2020-04-27', '2020-04-28', '2020-04-30']

dates = ['2020-03-29', '2020-04-07', '2020-04-16', '2020-04-25', '2020-03-30', '2020-04-08', '2020-04-17', '2020-04-26', '2020-03-31', '2020-04-09', '2020-04-18', '2020-04-27', '2020-04-01', '2020-04-10', '2020-04-19', '2020-04-28',
         '2020-04-02', '2020-04-11', '2020-04-20', '2020-04-29', '2020-04-03', '2020-04-12', '2020-04-21', '2020-04-30', '2020-04-04', '2020-04-13', '2020-04-22', '2020-04-05', '2020-04-14', '2020-04-23', '2020-04-06', '2020-04-15', '2020-04-24']

tweets = None
for date in dates:
    tweets_for_date = read_dataset_for_date(date)
    if not tweets:
        tweets = tweets_for_date
    else:
        tweets = tweets.union(tweets_for_date)


tweets_q3_df = tweets.filter(
    "created_at like '%-%-%T%:%:%Z' and account_created_at like '%-%-%T%:%:%Z'")

tweets_q3_df = tweets_q3_df.withColumn(
    "followers_count_int", f.col("followers_count").cast("int"))
tweets_q3_users_df = tweets_q3_df\
    .groupBy("user_id")\
    .agg(f.min("followers_count_int").alias('min'), f.max("followers_count_int").alias('max'), f.count("*").alias("count"))

tweets_q3_users_df = tweets_q3_users_df.withColumn(
    'growth', tweets_q3_users_df['max'] - tweets_q3_users_df['min'])

top_100_active_df = tweets_q3_users_df.sort(
    "count", ascending=False).limit(100)

# top_100_active_df.write.csv(
#     './local_tp2_3_top_100_active.csv', header=True)
top_100_active_df.write.csv(
    'hdfs://compute1:9000/user/danilo-p/tp2_3_top_100_active.csv', header=True)

top_100_growth_df = tweets_q3_users_df.sort(
    "growth", ascending=False).limit(100)

# top_100_growth_df.write.csv(
#     './local_tp2_3_top_100_growth.csv', header=True)
top_100_growth_df.write.csv(
    'hdfs://compute1:9000/user/danilo-p/tp2_3_top_100_growth.csv', header=True)

top_growth_active_df = top_100_active_df.alias('ta').join(
    top_100_growth_df.alias('tg'), f.col('ta.user_id') == f.col('tg.user_id'))
top_growth_active_df = top_growth_active_df.select("ta.user_id")

# top_growth_active_df.write.csv(
#     './local_tp2_3_top_growth_active.csv', header=True)
top_growth_active_df.write.csv(
    'hdfs://compute1:9000/user/danilo-p/tp2_3_top_growth_active.csv', header=True)
