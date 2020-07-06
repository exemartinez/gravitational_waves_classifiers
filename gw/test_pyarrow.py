# Now, we load up all the images and we cross it with the csv file, in order to obtain a "more suitable" dataset to train an algorithm.
import findspark
findspark.init()

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("pyspark-gw").getOrCreate()

print(spark.version)