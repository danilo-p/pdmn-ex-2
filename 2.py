import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType

spark = SparkSession\
    .builder\
    .appName("TP2 - 2")\
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


def transform_created_at(created_at):
    return created_at.split("T")[0]


def extract_mavg_data_for_country(country_code):
    transform_created_at_udf = spark.udf.register(
        "transform_created_at", transform_created_at, StringType())

    tweets_q2 = tweets.filter(
        "created_at like '2020-%' and country_code = '{}'".format(country_code))

    tweets_q2 = tweets_q2.select(
        'created_at', transform_created_at_udf('created_at'))

    tweets_q2_dates = tweets_q2.groupBy("transform_created_at(created_at)")\
        .count()\
        .sort('transform_created_at(created_at)')

    def produce_days(row):
        date_time_str = row['transform_created_at(created_at)']
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')

        result = []
        for i in range(0, 5):
            d = date_time_obj + datetime.timedelta(days=i)
            result.append((d.strftime("%Y-%m-%d"), (1, row['count'])))

        return result

    mavg_data_rdd = tweets_q2_dates.rdd\
        .flatMap(produce_days)\
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))\
        .sortByKey()\
        .map(lambda a: (a[0], a[1][1] / a[1][0]))\
        .map(lambda a: "{},{}".format(a[0], a[1]))

    # mavg_data_rdd.saveAsTextFile(
    #     "./local_tp2_2_mavg_data_{}.csv".format(country_code))
    mavg_data_rdd.saveAsTextFile(
        "hdfs://compute1:9000/user/danilo-p/tp2_2_mavg_data_{}.csv".format(country_code))


extract_mavg_data_for_country("US")
extract_mavg_data_for_country("BR")
extract_mavg_data_for_country("MX")
