# NOTE: I was trying to name RDDs by the following rule: key__value (two "_" between key and value)
# Run the code using "spark-submit draft.py"
# Files "names.txt", "D1.txt", "D2.txt", "D3.txt" are stored in temp directory in hdfs

from pyspark import SparkContext
import math
import numpy as np

names_file = "hdfs://localhost:9000/temp/names.txt"
input_term = "I"    # Query term for term-term relevance
sc = SparkContext("local", "First App")

# RDD of names_file
all_names = sc.textFile(names_file)

# list containing tuples (docname, its url)
names_url = all_names.flatMap(lambda line: line.split("\n")) \
    .map(lambda name: (name, "hdfs://localhost:9000/temp/" + name)) \
    .collect()

# number of documents
size_of_corpus = len(names_url)



# PART 1: TF-IDF

# RDD containing tuples (docname, list of words in it)
name__list_of_words = sc.parallelize([(pair[0],
                                       sc.textFile(pair[1])
                                       .flatMap(lambda line: line.split(" "))
                                       .collect())
                                      for pair in names_url])

# RDD containing tuples ((docname, word), local tf)
# Local tf for current (docname, word) equals to 1/(number of words in the document)
doc_word__local_tf = name__list_of_words.flatMap(lambda pair: [((pair[0], word), 1 / len(pair[1])) for word in pair[1]])

# Reduce ((docname, word), local tf) to ((docname, word), tf) by summarizing local tf.
# Hence, total tf for each (docname, word) would be (count of word in doc / number of words in doc)
doc_word__tf = doc_word__local_tf.reduceByKey(lambda x, y: x + y)

# Reduced RDD (word, number of distinct docs in which this word occurs)
word__num_of_doc = sc.parallelize(list(name__list_of_words
                                       .flatMap(lambda pair: [(word, pair[0]) for word in pair[1]])
                                       .distinct()
                                       .countByKey()
                                       .items()))

# RDD (word, idf) computed by mapping word__num_of_doc into
# (word, log10(size_of_corpus/(number of docs in which word occur)))
word__idf = word__num_of_doc.map(lambda pair: (pair[0], math.log(size_of_corpus / pair[1], 10)))

# RDD (word, (docname, tf))
word__doc_tf = doc_word__tf.map(lambda pair: (pair[0][1], (pair[0][0], pair[1])))

# Reduce by joining RDDs into (word, ((docname, tf), idf))
joined = sc.parallelize(word__doc_tf.join(word__idf).collect())

# Map joined RDD into ((word, docname), tf*idf) by
word_doc__tfidt = joined.map(lambda pair: ((pair[0], pair[1][0][0]), pair[1][0][1] * pair[1][1]))

# You can print the listed table of tfidt here, which should correspond to the slide number 69
# print(word_doc__tfidt.collect())




# PART 2: TERM-TERM frequency


# Map joined RDD is (word, (docname, tf*idf))
word__doc_tfidt = joined.map(lambda pair: (pair[0], (pair[1][0][0], pair[1][0][1] * pair[1][1])))

# List of tuples (docname, tf*idf) for input_term
input_tuples = word__doc_tfidt \
    .filter(lambda pair: pair[0] == input_term) \
    .map(lambda pair: (pair[1][0], pair[1][1])) \
    .collect()
# Put them into dictionary (it basically corresponds to the row corresponding to input_term)
input_dictionary = dict(input_tuples)
# Constant sqrt(tfidf1^2+tfidf2^2+...) for input_term
input_sqrt_of_sqr_values = np.sqrt(np.sum([x ** 2 for x in input_dictionary.values()]))

# Mapping RDD is ((word, docname), tf*idf) into RDD (word, tfidf*input_dictionary[docname]).
# If that docname not in the dictionary, multiply by 0.
# Then reducing into RDD (word, numerator) by summarizing tfidf*input_dictionary[docname] for every docname,
# where numerator is a numerator for particular word built by the formula from page 70.
word__numerator = word_doc__tfidt \
    .map(lambda pair: (pair[0][0], pair[1] * input_dictionary.get(pair[0][1], 0))) \
    .reduceByKey(lambda x, y: x + y)

# From (word, ((docname, tf), idf)) map RDD is (word, (tf*idf)^2)
word__squared_tfidt = joined.map(lambda pair: (pair[0], (pair[1][0][1] * pair[1][1]) ** 2))
# Then reduced it to RDD (word, sqrt(tfidf1^2+tfidf2^2+...)*input_sqrt_of_sqr_values) to create a denominator
# from slide 70 for each word
word__denominator = word__squared_tfidt.reduceByKey(lambda x, y: x + y) \
    .map(lambda pair: (pair[0], math.sqrt(pair[1]) * input_sqrt_of_sqr_values))

# Join numerator and denominator to make RDD (word, (numerator, denominator))
joined_num_and_denum = sc.parallelize(word__numerator.join(word__denominator).collect())

# Finally, compute for each word tuples (numerator, denominator), which are exactly Similarity(input_word, word)
# and then sort them in descending order (sortByKey(False)) by swapping tuples two times
sorted_result = joined_num_and_denum \
    .map(lambda pair: (pair[0], pair[1][0] / pair[1][1])) \
    .map(lambda pair: (pair[1], pair[0])) \
    .sortByKey(False) \
    .map(lambda pair: (pair[1], pair[0])) \
    .collect()

print(sorted_result)
