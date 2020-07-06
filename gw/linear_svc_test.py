#Performing the due classification of the testing dataset.
import findspark
import gc

findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LinearSVCModel
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.linalg import Vectors
import pyspark.sql.functions as F
import numpy as np

# initialise sparkContext
spark = SparkSession.builder \
    .master('local') \
    .appName('myAppName') \
    .config('spark.executor.memory', '12gb') \
    .config("spark.cores.max", "2") \
    .getOrCreate()

sc = spark.sparkContext

# using SQLContext to read parquet file
sqlContext = SQLContext(sc)

# to read parquet file (full) # TODO - Research a way to read ONLY a given partition.
#parquet_df = spark.read.parquet('/dataset/gw_gravity_spy_dataframe')
parquet_df = spark.read.parquet('/dataset/gw_gravity_spy_dataframe')

#filter the test set
test_set= parquet_df.where(parquet_df['sample_type']=='test')
test_set.count()

import pyspark.sql.functions as F

test = test_set.select("label","png")

print(test.columns)

#Reshaping the labels as "Chrip" = "True" and all the others as "False"; Chrips are Gravitational Waves.
result = test.where(test.label == "Chirp")
print("Gravitational Waves: {0}".format(result.count()))

test = test.withColumn('gw', (test.label == "Chirp"))
test = test.drop("label")

test = test.withColumn('features', test.png)
test = test.withColumn('label', test.gw)

test = test.drop("png")
test = test.drop("gw")
test = test.withColumn('label', (F.col('label') == True).cast('integer'))

print(test.columns)

result = test.where(test.label == 1)
print("Gravitational Waves (after): {0}".format(result.count()))

#droping the categories
test_dumb = test.drop('label')

#predicting...
print("Loading the model...")
lsvcModel =  LinearSVCModel.load("./lsvc.model")
result = lsvcModel.transform(test_dumb)

#We present the prediction for the first element from TEST.
#ASSUMING that the order is the very same...

print("Main procedures finished. Saving block...")

# We use the spark dataset to write its contents (all the gravity spy's dataset) to a partquet file for easy classification. 

result.write.format("parquet") \
.partitionBy("prediction") \
.option("path", "/dataset/gw_gravity_spy_dataframe_prediction") \
.mode("overwrite") \
.saveAsTable("gw_gravity_spy_prediction")    

spark.stop()

gwdf = None
sdf = None

gc.collect()

