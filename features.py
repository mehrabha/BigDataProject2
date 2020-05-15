# NOTE: The names of RDDs follow this rule: key__value (two "_" between key and value)
import re
import math
import numpy as np


# Computes TF-IDF by returning word_doc__tfidf containing ((word, docname), tf*idf)
# It uses regular expression to select particular words.
# Returns RDD containing ((word, docname), tf*idf).
def compute_tfidf(rdd_files_from_input, sc, regex):
    # RDD containing tuples (docname, list of words in it). split() string on whitespaces (any number of spaces)
    name__list_of_words = rdd_files_from_input.flatMap(lambda line: line.split("\n")) \
        .map(lambda line: line.split()) \
        .map(lambda line: (line[0], line[1:]))

    # number of documents
    size_of_corpus = name__list_of_words.count()

    # RDD containing tuples ((docname, word), local tf), where word matches the pattern gene_???_gene
    # Local tf for current (docname, word) equals to 1/(number of words in the document)
    doc_word__local_tf = name__list_of_words \
        .flatMap(lambda pair: [((pair[0], word), 1 / len(pair[1])) for word in pair[1]]) \
        .filter(lambda pair: regex.match(pair[0][1]))

    # Reduce ((docname, word), local tf) to ((docname, word), tf) by summarizing local tf.
    # Hence, total tf for each (docname, word) would be (count of word in doc / number of words in doc)
    doc_word__tf = doc_word__local_tf.reduceByKey(lambda x, y: x + y)

    # Reduced RDD (word, number of distinct docs in which this word occurs)
    word__num_of_doc = sc.parallelize(list(doc_word__tf
                                           .map(lambda pair: (pair[0][1], pair[0][0]))
                                           .distinct()
                                           .countByKey()
                                           .items()))

    # RDD (word, idf) computed by mapping word__num_of_doc into
    # (word, log10(size_of_corpus/(number of docs in which word occur)))
    word__idf = word__num_of_doc.map(lambda pair: (pair[0], math.log(size_of_corpus / pair[1], 10)))

    # RDD (word, (docname, tf))
    word__doc_tf = doc_word__tf.map(lambda pair: (pair[0][1], (pair[0][0], pair[1])))

    # Reduce by joining RDDs into (word, ((docname, tf), idf))
    joined = word__doc_tf.join(word__idf)

    # Map joined RDD into ((word, docname), tf*idf).
    word_doc__tfidf = joined.map(lambda pair: ((pair[0], pair[1][0][0]), pair[1][0][1] * pair[1][1]))

    return word_doc__tfidf


# Computes term-term frequency for input_term given the current TF-IDF from word_doc__tfidf using pairs.
# Returns sorted_result containing ((input_term, word), similarity(input_term, word))
def compute_term_term_using_pairs(word_doc__tfidf, input_term):
    # List of tuples (docname, tf*idf) for input_term
    input_tuples = word_doc__tfidf \
        .filter(lambda pair: pair[0][0] == input_term) \
        .map(lambda pair: (pair[0][1], pair[1])) \
        .collect()
    # Put them into dictionary (it basically corresponds to the row corresponding to input_term)
    input_dictionary = dict(input_tuples)
    # Constant sqrt(tfidf1^2+tfidf2^2+...) for input_term
    input_sqrt_of_sqr_values = np.sqrt(np.sum([x ** 2 for x in input_dictionary.values()]))

    # Mapping RDD is ((word, docname), tf*idf) into RDD (word, tfidf*input_dictionary[docname]).
    # If that docname not in the dictionary, multiply by 0.
    # Then reducing into RDD (word, numerator) by summarizing tfidf*input_dictionary[docname] for every docname,
    # where numerator is a numerator for particular word built by the formula from page 70.
    word__numerator = word_doc__tfidf \
        .map(lambda pair: (pair[0][0], pair[1] * input_dictionary.get(pair[0][1], 0))) \
        .reduceByKey(lambda x, y: x + y)

    # From ((word, docname), tf*idf) map RDD is (word, (tf*idf)^2)
    word__squared_tfidf = word_doc__tfidf.map(lambda pair: (pair[0][0], (pair[1]) ** 2))
    # Then reduced it to RDD (word, sqrt(tfidf1^2+tfidf2^2+...)*input_sqrt_of_sqr_values) to create a denominator
    # from slide 70 for each word
    word__denominator = word__squared_tfidf.reduceByKey(lambda x, y: x + y) \
        .map(lambda pair: (pair[0], math.sqrt(pair[1]) * input_sqrt_of_sqr_values))

    # Join numerator and denominator to make RDD (word, (numerator, denominator))
    joined_num_and_denum = word__numerator.join(word__denominator)

    # Finally, compute for each word tuples (numerator, denominator), which are exactly Similarity(input_word, word)
    # and then sort them in descending order (sortByKey(False)) by swapping tuples two times.
    # At the end we add input term to the key to make sorted_result = ((input_term, word), similarity(input_term, word))
    sorted_result = joined_num_and_denum \
        .map(lambda pair: (pair[0], pair[1][0] / pair[1][1])) \
        .map(lambda pair: (pair[1], pair[0])) \
        .sortByKey(False) \
        .map(lambda pair: ((input_term, pair[1]), pair[0]))

    return sorted_result


# Computes similarity by getting scalar multiplication of two dictionaries "stripe" and "input_dictionary"
# and divining it by denominator.
def similarity(stripe, input_dictionary, denominator):
    numerator = 0
    for key in stripe:
        numerator += stripe[key] * input_dictionary.get(key, 0)
    return numerator / denominator


# Computes term-term frequency for input_term given the current TF-IDF from word_doc__tfidf using stripes.
# Returns sorted_result containing ((input_term, word), similarity(input_term, word))
def compute_term_term_using_stripes(word_doc__tfidf, input_term):
    # Mapping into stripes. First map from ((word, docname), tf*idf) to (word, {docname: tf*idf}).
    # Then reduce it into (word, stripe), where stripe is {docname1: tfidf1, docname2: tfidf2, ...}.
    word__stripe = word_doc__tfidf \
        .map(lambda pair: (pair[0][0], {pair[0][1]: pair[1]})) \
        .reduceByKey(lambda x, y: {**x, **y})

    # Return dictionary for input_term
    input_word__stripe = word__stripe \
        .filter(lambda pair: pair[0] == input_term) \
        .collect()
    # Extract the dictionary
    input_dictionary = input_word__stripe[0][1]
    # Constant sqrt(tfidf1^2+tfidf2^2+...) for input_term
    input_sqrt_of_sqr_values = np.sqrt(np.sum([x ** 2 for x in input_dictionary.values()]))

    # From ((word, docname), tf*idf) map RDD is (word, (tf*idf)^2)
    word__squared_tfidf = word_doc__tfidf.map(lambda pair: (pair[0][0], (pair[1]) ** 2))
    # Then reduced it to RDD (word, sqrt(tfidf1^2+tfidf2^2+...)*input_sqrt_of_sqr_values) to create a denominator
    # from slide 70 for each word
    word__denominator = word__squared_tfidf.reduceByKey(lambda x, y: x + y) \
        .map(lambda pair: (pair[0], math.sqrt(pair[1]) * input_sqrt_of_sqr_values))

    # Join stripe and denominator to make RDD (word, (stripe, denominator))
    joined_stripe_and_denum = word__stripe.join(word__denominator)

    # Map (word, (stripe, denominator)) to (word, similarity(input_term, word))
    word__sim = joined_stripe_and_denum\
        .map(lambda pair: (pair[0], similarity(pair[1][0], input_dictionary, pair[1][1])))

    # Sort word__sim in descending order (sortByKey(False)) by swapping tuples two times.
    # At the end we add input term to the key to make sorted_result = ((input_term, word), similarity(input_term, word))
    sorted_result = word__sim \
        .map(lambda pair: (pair[1], pair[0])) \
        .sortByKey(False) \
        .map(lambda pair: ((input_term, pair[1]), pair[0]))

    return sorted_result
