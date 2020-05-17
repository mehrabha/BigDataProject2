import pyspark, math, os, shutil
from itertools import permutations
sc = pyspark.SparkContext('local', 'First App')

class Project2:
    def __init__(self, filesrc):
        self.filesrc = filesrc
        self.tfidf = None
        self.rel = None

    def task1(self):
        # load docs from hdfs
        docs = sc.textFile(self.filesrc)
        num_docs = docs.count()
        # calculate tf, reduce by (docname, word) -> (docname, word: tf)
        tfs = docs.flatMap(lambda x: extract(x.split())).reduceByKey(lambda x, y: x + y)
        # make word key for tfs (docname, word: tf) -> (word: docname, tf)
        a = tfs.map(lambda x: ( x[0][1], (x[0][0], x[1]) ))
        # map (docname, word: tf) -> (word: 1), reduce to (word, total)
        b = tfs.map(lambda x: (x[0][1], 1)).reduceByKey(lambda x, y: x + y)
        b = b.map(lambda x: ( x[0], math.log(num_docs / x[1], 10)))
        # join tfs and w_count -> ( (word, tuple), count)
        join = a.join(b)
        # calculate tf * idf -> (tuple, tfidf)
        self.tfidf = join.map(lambda x: ( (x[0], x[1][0][0]), x[1][0][1] * x[1][1] ))
        self.tfidf.saveAsTextFile('./results/task1/')
    
    def task2(self, pattern, query):
        # filter tfidf by query
        filtered = self.tfidf.filter(lambda x: queryfilter(x[0][0], pattern))
        # calculate sum squared for word
        r = filtered.map(lambda x: ( x[0][0], x[1] * x[1] )).reduceByKey(lambda x, y: x + y)
        # Take square root
        r = r.map(lambda x: ( x[0], math.sqrt(x[1]) ))
        # map tfidf to list for word -> (word: {docname: tfidf}),
        tlist = filtered.map(lambda x: (x[0][0], { x[0][1]: x[1] }))
        # reduce lists -> (word: {doc1, doc2...})
        tlist = tlist.reduceByKey(lambda x, y: reducer(x, y))
        # join mapped list and squareroots -> (word: [docs], root)
        join = tlist.join(r)
        # make word pairs -> (word:[docs], word:[docs])
        pairs = list(permutations(join.collect(), 2))
        pairs = filterByQuery(pairs, query)
        # calculate numerators, divide by roots -> (similarity: word,word)
        rel = map(lambda x: ( similarity(x[0][1], x[1][1]), (x[0][0], x[1][0]) ), pairs)
        self.rel = sc.parallelize(rel).sortByKey(False)
        self.rel.saveAsTextFile('./results/task2/')

# returns [ (docname, word: tf), ...] from word list
def extract(l) -> list:
    docname, words = l[0], l[1:]
    result = []
    # Fill result with tuples
    for word in words:
        tup = ( (docname, word), 1/len(words))
        result.append(tup)
    return result

def filterByQuery(l, query):
    filtered = []
    for elem in l:
        if elem[0][0] == query or elem[1][0] == query:
            filtered.append(elem)
    return filtered

# merges dict b into a, returns a
def reducer(a, b) -> dict:
    c = {}
    for key, val in b.items():
        c[key] = val
    return c

def similarity(I, A) -> float:
    numer = 0
    for key in I[0]:
        if key in A[0]:
            numer += I[0][key] * A[0][key]
    # final calculations
    return numer / (I[1] * A[1])
    
# Check if string matches pattern
def queryfilter(s, p) -> bool:
    p = p.replace('?', ' ').split()
    return (s.startswith(p[0]) and s.endswith(p[1]))

 