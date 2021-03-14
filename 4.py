import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA

spark = SparkSession\
    .builder\
    .appName("TP2 - 4")\
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

# Based on
# - https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3741049972324885/3783546674231782/4413065072037724/latest.html
# - https://databricks.com/blog/2015/09/22/large-scale-topic-modeling-improvements-to-lda-on-apache-spark.html

tweets_q4_text = tweets.filter(
    "created_at like '2020-%' and country_code = 'US'").select("text")

tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens",
                           pattern="([#@]\\S*|\\W|https)", minTokenLength=4)

tweets_q4_tokens_df = tokenizer.transform(tweets_q4_text).select("tokens")

locale = spark.sparkContext._jvm.java.util.Locale
locale.setDefault(locale.forLanguageTag("en-US"))

remover = StopWordsRemover(inputCol="tokens", outputCol="tokens_filtered")
tokens_filtered_df = remover.transform(
    tweets_q4_tokens_df).select("tokens_filtered")

vocab_df = tokens_filtered_df.withColumn("vocab", f.explode(
    f.col('tokens_filtered'))).select("vocab").distinct()

tokens_filtered_df = tokens_filtered_df.withColumn(
    "id", f.monotonically_increasing_id())

cv = CountVectorizer(inputCol="tokens_filtered",
                     outputCol="features", vocabSize=vocab_df.count())
cv_model = cv.fit(tokens_filtered_df)

count_vectors = cv_model.transform(tokens_filtered_df).select("id", "features")


def create_topics_df(lda_model):
    topic_indices = lda_model.describeTopics(5)
    vocab_list = cv_model.vocabulary
    topics = topic_indices.rdd.map(lambda t: zip(
        [vocab_list[ti] for ti in t.termIndices], t.termWeights))

    i = 0
    data = []
    for topic in topics.collect():
        for term, weight in topic:
            data.append((i, term, weight))
        i += 1

    return spark.createDataFrame(data, ["topic_id", "term", "term_weight"])


topics_max_iter_3_df = create_topics_df(LDA(k=3, maxIter=3).fit(
    count_vectors))

# topics_max_iter_3_df.write.csv(
#     './local_tp2_4_topics_max_iter_3.csv', header=True)
topics_max_iter_3_df.write.csv(
    'hdfs://compute1:9000/user/danilo-p/tp2_4_topics_max_iter_3.csv', header=True)

topics_max_iter_10_df = create_topics_df(LDA(k=3, maxIter=10).fit(
    count_vectors))

# topics_max_iter_10_df.write.csv(
#     './local_tp2_4_topics_max_iter_10.csv', header=True)
topics_max_iter_10_df.write.csv(
    'hdfs://compute1:9000/user/danilo-p/tp2_4_topics_max_iter_10.csv', header=True)

topics_max_iter_30_df = create_topics_df(LDA(k=3, maxIter=30).fit(
    count_vectors))

# topics_max_iter_30_df.write.csv(
#     './local_tp2_4_topics_max_iter_30.csv', header=True)
topics_max_iter_30_df.write.csv(
    'hdfs://compute1:9000/user/danilo-p/tp2_4_topics_max_iter_30.csv', header=True)

topics_max_iter_100_df = create_topics_df(LDA(k=3, maxIter=100).fit(
    count_vectors))

# topics_max_iter_100_df.write.csv(
#     './local_tp2_4_topics_max_iter_100.csv', header=True)
topics_max_iter_100_df.write.csv(
    'hdfs://compute1:9000/user/danilo-p/tp2_4_topics_max_iter_100.csv', header=True)
