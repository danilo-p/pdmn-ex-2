from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType

spark = SparkSession\
    .builder\
    .appName("TP2 - 1 (BR)")\
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


def transform_text(text):
    text = text.replace(".", " ")
    text = text.replace(",", " ")
    text = text.replace('"', " ")
    return text.lower()


transform_text_udf = spark.udf.register(
    "transform_text", transform_text, StringType())

tweets_br = tweets.filter("country_code = 'BR'")

tweets_br = tweets_br.select('text', transform_text_udf('text'))

parts_df = tweets_br.withColumn("parts", f.explode(
    f.split(f.col('transform_text(text)'), ' '))).select("parts")

hashtags_df = parts_df.filter("parts like '#%'")

top_15_hashtags_br_df = hashtags_df.groupBy("parts")\
    .count()\
    .sort('count', ascending=False)\
    .limit(15)

# top_15_hashtags_br_df.write.csv('./top_15_hashtags_br.csv', header=True)
top_15_hashtags_br_df.write.csv(
    'hdfs://compute1:9000/user/danilo-p/top_15_hashtags_br.csv', header=True)
