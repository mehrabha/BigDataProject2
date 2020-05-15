# NOTE: The names of RDDs follow this rule: key__value (two "_" between key and value)
# Run the code using :
# "spark-submit main.py hdfs://localhost:9000/temp/project2_test.txt <gene_???_gene> <number-of-cores> <pair or stripe>"
# For example, you can run
# "spark-submit main.py hdfs://localhost:9000/temp/project2_test.txt gene_egf_gene 1 pair"

from pyspark import SparkContext
from os import path
import re
import sys
import timeit
import shutil
import features

# Input file
txt_files = sys.argv[1]

# Query term for term-term relevance
input_term = sys.argv[2]
# Regular expression for which we select all the words. In our case, it is gene_???_gene
regex = re.compile('^(gene_).*(_gene)$')
if not regex.match(input_term):
    print("Input tern should be gene_???_gene.")
    sys.exit()

# Number of cores/threads used
num_cores = sys.argv[3]

# Choose whether we're using pairs or stripes
pair_or_stripe = sys.argv[4]
if not (pair_or_stripe == "pair" or pair_or_stripe == "stripe"):
    print("Choose pair or stripe.")
    sys.exit()
result_directory = "result_pair" if (pair_or_stripe == "pair") else "result_stripe"

# Set up spark context
master_local = "local" if (num_cores == "1") else "local[" + num_cores + "]"
start = timeit.default_timer()
sc = SparkContext(master_local, "Project 2")

# RDD of files
rdd_files = sc.textFile(txt_files)

# Compute TF-IDF and store it in RDD word_doc__tfidf of the form ((word, doc), tfidf) for those words,
# which correspond to regex. It uses ('^(gene_).*(_gene)$') to select gene_???_gene.
word_doc__tfidf = features.compute_tfidf(rdd_files, sc, regex)

# Compute term-term relevance using either pairs or stripes for given input term and store it in
# RDD sorted_result containing ((input_term, word), similarity(input_term, word)).
if pair_or_stripe == "pair":
    sorted_result = features.compute_term_term_using_pairs(word_doc__tfidf, input_term)
else:
    sorted_result = features.compute_term_term_using_stripes(word_doc__tfidf, input_term)

# Remove directory result, if it was constructed before
if path.exists(result_directory):
    shutil.rmtree(result_directory)

# Use coalesce at the end to move everything to one partition and store the result as one file.
sorted_result \
    .coalesce(1, True) \
    .saveAsTextFile(result_directory)

stop = timeit.default_timer()
print("Total time for", master_local, "to create", result_directory, ":", stop - start)
