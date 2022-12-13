import sys

sys.path.append("/path/to/pyspark")

import pyspark

sys.path.append("/path/to/pyspark")


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

df1 = spark.read.csv(f"s3a://{BUCKET}/{KEY1}", header=True)
df2 = spark.read.csv(f"s3a://{BUCKET}/{KEY2}", header=True)

pre_release = df1.toPandas()
post_release = df2.toPandas()

# IMDB CODE

movie_data = pre_release.merge(post_release, on='movie_title', how='inner')
movie_data = movie_data.drop(columns = ["num_critic_for_reviews","gross","movie_title","num_voted_users","num_user_for_reviews","movie_facebook_likes"])

movie_data.drop_duplicates(inplace = True)

movie_data = movie_data.drop(columns = ["director_name","actor_1_name","actor_2_name","actor_3_name"])

movie_data['content_rating'].fillna('R', inplace=True)

movie_data['language'].fillna('English', inplace=True)

movie_data["actor_1_facebook_likes"].fillna(653.813397, inplace=True)
movie_data["actor_2_facebook_likes"].fillna(438.397510, inplace=True)
movie_data["actor_3_facebook_likes"].fillna(299.271593, inplace=True)
movie_data.dropna()

for i in ["UK","France","Germany","Spain","Italy","Norway","Denmark","Ireland","Netherlands","Romania","Sweden","Iceland","West Germany","Slovenia","Poland","Czech Republic","Bulgaria","Belgium"]:
    c = movie_data["country"] != i
    movie_data["country"].where(c, "Europe", inplace = True)
cat = ["USA","Europe"]
movie_data['country'] = movie_data.country.where(movie_data.country.isin(cat), 'other')

l = movie_data["language"] == "English"
movie_data["language"].where(l, "Foreign", inplace = True)

x = movie_data.drop(columns = ["imdb_score","content_rating"])
y = movie_data["imdb_score"] 

x = pd.concat([x, x['genres'].str.get_dummies(sep='|')], axis=1)
x = x.drop("genres",axis = 1)

x_head_cat = ["language","country"]
x_head_num = x.columns[~x.columns.isin(x_head_cat)]

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    del dummies[dummies.columns[0]] 
    res = pd.concat([original_dataframe, dummies], axis=1)
    res.drop(feature_to_encode, axis=1, inplace=True) 
    return(res)
for categorical_feature in x_head_cat:
    x[categorical_feature] = x[categorical_feature].astype('category')
    x = encode_and_bind(x,categorical_feature)

seed = 123 
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.3, random_state = seed)

rfreg = RandomForestRegressor(max_depth=10, min_samples_leaf =1, random_state=0).fit(x_train, y_train)

array_pred = np.round(rfreg.predict(x_val),0)
y_pred = pd.DataFrame({"y_pred": array_pred},index=x_val.index) #index must be same as original database
val_pred = pd.concat([y_val,y_pred,x_val],axis=1)

def equivalent_type(f):
    if f == 'datetime64[ns]': return TimestampType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return DoubleType()
    elif f == 'float32': return FloatType()
    else: return StringType()

def define_structure(string, format_type):
    try: typo = equivalent_type(format_type)
    except: typo = StringType()
    return StructField(string, typo)


def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types): 
      struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return spark.createDataFrame(pandas_df, p_schema)

print(val_pred)

val_pred = pandas_to_spark(val_pred)
val_pred.write.json(f"s3a://{BUCKET}/vlerick/thomas_debackker/")