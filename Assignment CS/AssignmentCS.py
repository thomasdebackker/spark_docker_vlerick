from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

BUCKET = "dmacademy-course-assets"
KEY1 = "vlerick/pre_release.csv"
KEY2 = "vlerick/after_release.csv"

config = {
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
    "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
}
conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

#Read the CSV file from the S3 bucket
pre_release = spark.read.csv(f"s3://{BUCKET}/{KEY1}",header=True)
after_release = spark.read.csv(f"s3a://{BUCKET}/{KEY2}",header=True)