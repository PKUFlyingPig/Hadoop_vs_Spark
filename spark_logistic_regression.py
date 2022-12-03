import sys
import time
from typing import Iterable, List

import numpy as np
from pyspark.sql import SparkSession

D = 10  # Number of dimensions

# Read a batch of points from the input file into a NumPy matrix object. We operate on batches to
# make further computations faster.
# The data file contains lines of the form <label> <x1> <x2> ... <xD>. We load each block of these
# into a NumPy array of size numLines * (D + 1) and pull out column 0 vs the others in gradient().
def readPointBatch(iterator: Iterable[str]) -> List[np.ndarray]:
    strs = list(iterator)
    matrix = np.zeros((len(strs), D + 1))
    for i, s in enumerate(strs):
        matrix[i] = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return [matrix]

# Compute logistic regression gradient for a matrix of data points
def gradient(matrix: np.ndarray, w: np.ndarray) -> np.ndarray:
    Y = matrix[:, 0]    # point labels (first column of input file)
    X = matrix[:, 1:]   # point coordinates
    # For each point (x, y), compute gradient function, then sum these up
    return ((1.0 / (1.0 + np.exp(-Y * X.dot(w))) - 1.0) * Y * X.T).sum(1)

def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x += y
    return x

def run_one_case(spark, filename, iterations):
    start = time.time()

    points = spark.read.text(filename).rdd.map(lambda r: r[0])\
        .mapPartitions(readPointBatch).cache()

    # Initialize w to a random value
    w = np.random.ranf(size=D)

    for i in range(iterations):
        w -= points.map(lambda m: gradient(m, w)).reduce(add)

    return time.time() - start


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: logistic_regression <iterations>", file=sys.stderr)
        sys.exit(-1)
    
    size_list = [1, 4, 16, 64, 128]
    iterations = int(sys.argv[1])
    spark = SparkSession\
        .builder\
        .appName("PythonLR")\
        .getOrCreate()
    
    #warmup
    warmup_file = f"hdfs:///dataset/default.csv"
    warmup_time = run_one_case(spark, warmup_file, iterations)
    print(f"warmup time: {warmup_time:.5f}s")
    for s in size_list:
        filename = f"hdfs:///dataset/{s}M.csv"
        benchmark_time = run_one_case(spark, filename, iterations)
        print(f"{s}M case: {benchmark_time:.5f}s")

    spark.stop()


