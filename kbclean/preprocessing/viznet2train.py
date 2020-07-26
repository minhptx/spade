from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, udf
from pyspark.sql.types import ArrayType, StringType
import regex as re


def eval_with_catch(x):
    try:
        return eval(x)
    except Exception:
        return []


def clean_str(x):
    x = x.strip().encode("ascii", "ignore").decode("ascii")
    return str2regex(x)


def str2regex(x):
    try:
        if x is None:
            return ""
        x = re.sub(r"[A-Z]+", "A", x)
        x = re.sub(r"[0-9]+", "0", x)
        x = re.sub(r"[a-z]+", "a", x)
        return x
    except Exception as e:
        return x


if __name__ == "__main__":
    sc = SparkContext("local[*]")
    spark = SparkSession(sc)
    # spark.sparkContext.setLogLevel("WARN")
    df = spark.read.format("csv").option("header", "true").load("/data/train/webtables/data/*.csv")
    udf_eval = udf(eval_with_catch, ArrayType(StringType()))
    udf_clean = udf(clean_str, StringType())

    df = df.withColumn("data", udf_eval("data"))
    df = df.withColumn("data", explode("data"))
    df = df.withColumn("data", udf_clean("data"))
    df.sample(fraction=0.5).select("data").limit(1).coalesce().write.option(
        "header", "true"
    ).mode("overwrite").save("/data/train/webtables/all")

