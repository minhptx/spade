from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, udf
from pyspark.sql.types import ArrayType, StringType


def eval_with_catch(x):
    try:
        return eval(x)
    except Exception:
        return []


def clean_str(x):
    return x.strip().encode("ascii", "ignore").decode("ascii")


if __name__ == "__main__":
    sc = SparkContext("local[*]")
    spark = SparkSession(sc)
    # spark.sparkContext.setLogLevel("WARN")
    df = spark.read.format("csv").option("header", "true").load("data/raw/sherlock/1*.csv")
    udf_eval = udf(lambda x: eval_with_catch(x), ArrayType(StringType()))
    udf_clean = udf(lambda x: clean_str(x), StringType())

    df = df.withColumn("data", udf_eval("data"))
    df = df.withColumn("data", explode("data"))
    df = df.withColumn("data", udf_clean("data"))
    df.select("data").limit(10000000).coalesce(1).write.option("header", "true").mode(
        "overwrite"
    ).csv("data/train/webtables.csv")
