import regex as re
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


def to_regex(x):
    try:
        if x is None:
            return ""
        x = re.sub(r"[A-Z]+", "A", x)
        x = re.sub(r"[0-9]+", "0", x)
        x = re.sub(r"[a-z]+", "a", x)
        return x
    except Exception as e:
        print(e)
        return x


if __name__ == "__main__":
    sc = SparkContext("local[*]")
    spark = SparkSession(sc)

    df = spark.read.format("csv").option("header", "true").load("data/all.csv")
    regex_udf = udf(to_regex, StringType())
    df = df.withColumn("pattern", regex_udf("data"))
    count_df = df.groupBy("pattern").count()
    count_df.coalesce(1).write.option("header", "true").mode("overwrite").csv(
        "data/pattern.csv"
    )
