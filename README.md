# Hadoop vs. Spark

In this project, we implemented the Logistic Regression algorithm in Hadoop and Spark respectively and compared their performance with different dataset sizes.

## Dataset

The data file contains lines of the form <label> <x1> <x2> ... <xD>.

```shell
usage: generate_dataset.py [-h] [--size SIZE]

optional arguments:
  -h, --help   show this help message and exit
  --size SIZE  The size (M) of the generated dataset.
```

Both the Hadoop and Spark implementation use HDFS as the underlying storage layer. After generating the dataset files on local storage, you need to put them under `hdfs:///dataset` . 

## Hadoop

Todo

## Spark

Install Spark following the [installation guide](https://spark.apache.org/downloads.html). 

Run the script below:

```shell
Usage: spark_logistic_regression.py <iterations>
```



