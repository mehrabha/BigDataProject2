import math
from pyspark import SparkContext
sc = SparkContext()

class Spark:
    def __init__(self, filesrc):
        self.filesrc = filesrc
        self.tfidf = None
        self.relations = None

    def task1(self):
        num_docs = 0
        # fill auxiliary array with tuples
        a = []
        for line in open(self.filesrc):
            num_docs += 1
            l = line.split()
            doc_name, doc_data = l[0], l[1:]

            for word in doc_data:
                tf_val = 1 / len(doc_data)
                tupl = ( (doc_name, word), tf_val )   # make tuple
                a.append(tupl)  # ( (docname, word), tf ) -> array
            
        # make rdd then reduce by key
        tfs = sc.parallelize(a).reduceByKey(lambda x, y: x + y)
        # make word key for tfs (docname, word: tf) -> (word: docname, tf)
        a = tfs.map(lambda x: ( x[0][1], (x[0][0], x[1]) ))
        # map (docname, word: tf) -> (word: 1), reduce to (word, total)
        b = tfs.map(lambda x: (x[0][1], 1)).reduceByKey(lambda x, y: x + y)
        b = b.map(lambda x: ( x[0], math.log(num_docs / x[1], 10)))
        # join tfs and w_count -> ( (word, tuple), count)
        join = a.join(b)
        # calculate tf * idf -> (tuple, tfidf)
        self.tfidf = join.map(lambda x: ( (x[0], x[1][0][0]), x[1][0][1] * x[1][1] ))

    def task2(self, query):
        # Calculate sum squared for word
        r = self.tfidf.map(lambda x: ( x[0][0], x[1] * x[1] )).reduceByKey(lambda x, y: x + y)
        # Take square root
        r = r.map(lambda x: ( x[0], math.sqrt(x[1]) ))
        # Calculate numerator



