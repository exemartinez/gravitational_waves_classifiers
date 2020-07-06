#Initialize the spark context and tools for processing the stored rows.
import findspark

findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVC
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

#filter the training set
train_set= parquet_df.where(parquet_df['sample_type']=='train')

# Extract and prepare the df to be processed with just the due columns. In this case, we want just the image.
# (and a way to associate it with the rest of the columns after classification)

train = train_set.select("label","png")

print("Records to be used for training: {0}".format(str(train_set.count())))

print(train.columns)

#Reshaping the labels as "Chrip" = "True" and all the others as "False"; Chrips are Gravitational Waves.
result = train.where(train.label == "Chirp")
print("Gravitational Waves: {0}".format(result.count()))

train = train.withColumn('gw', (train.label == "Chirp"))
train = train.drop("label")

train = train.withColumn('features', train.png)
train = train.withColumn('label', train.gw)

train = train.drop("png")
train = train.drop("gw")
train = train.withColumn('label', (F.col('label') == True).cast('integer')) # we cast it as integer to give the classifier algorithm what it likes.

print(train.columns)

result = train.where(train.label == 1)
print("Gravitational Waves (after): {0}".format(result.count()))


# Linear Support Vector Machine Classifier
lsvc = LinearSVC(maxIter=10, regParam=0.1)

print("Initiating training...")
# Fit the model
lsvcModel = lsvc.fit(train)

print("Finished!")

# Print the coefficients and intercept for linearsSVC
print("Coefficients: " + str(lsvcModel.coefficients))
print("Intercept: " + str(lsvcModel.intercept))

lsvcModel.save("./lsvc.model")

spark.stop()