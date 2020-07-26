import regex as re
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, udf
from pyspark.sql.types import ArrayType, StringType


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


def all_substrings(x):
    substrings = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            substrings.append(x[i:j])
    return substrings


if __name__ == "__main__":
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.driver.memory", "24g")
        .appName("my-cool-app")
        .getOrCreate()
    )

    df = spark.read.format("csv").option("header", "true").load("data/train/webtables/data/*.csv")
    regex_udf = udf(to_regex, StringType())
    substrs_udf = udf(all_substrings, ArrayType(StringType()))
    df = df.withColumn("pattern", regex_udf("data"))
    df = df.withColumn("pattern", substrs_udf("pattern"))
    df = df.withColumn("pattern", explode("pattern"))

    count_df = df.groupBy("pattern").count()
    count_df.coalesce(1).write.option("header", "true").mode("overwrite").csv("data/pattern.csv")
