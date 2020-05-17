# Run test using "python ./test_project2_test.py gene_egf_gene pair". It takes too long to test the new file.
# Tests algorithm for the first professor's file

import sys
import pandas as pd
import numpy as np

# Query term for term-term relevance
input_term = sys.argv[1]
# Choose whether we're using pairs or stripes
pair_or_stripe = sys.argv[2]

result_directory = "result_pair" if (pair_or_stripe == "pair") else "result_stripe"

input_term = 'gene_egf_gene'
with open('../resources/project2_test.txt', 'r') as f:
    lines = f.read().splitlines()

lines = [line.split() for line in lines]
files = [line[0] for line in lines]

genes = []
for line in lines:
    for word in line[1:]:
        if word[:5] == 'gene_' and word[-5:] == '_gene':
            genes.append(word)

genes = list(set(genes))
df = pd.DataFrame(columns=files, index=genes).fillna(value=0)

# Created df as in slide 66
for line in lines:
    for word in line[1:]:
        if word[:5] == 'gene_' and word[-5:] == '_gene':
            df[line[0]][df.index == word] += 1.

doc_sizes = np.array([len(line) - 1 for line in lines])
tf = df / doc_sizes
idf = np.log10(len(lines) * 1. / np.sum(df > 0, axis=1))
tfidf = tf * idf[:, np.newaxis]
# print(tfidf)
tfidf_norm = tfidf / np.sqrt(np.sum(tfidf**2, axis=1))[:, np.newaxis]
# print(tfidf_norm)

selected = np.array(tfidf_norm[tfidf_norm.index == input_term])
result = np.sum(tfidf_norm * selected, axis=1)
pd.set_option('display.max_rows', None)
# print(result.sort_values(ascending=False))


# Automatically compares results from the actual algorithm.
directory = "../"+result_directory+"/part-00000"
with open(directory, 'r') as f:
    lines = f.read().splitlines()

lines = [line.replace('(', '').replace(')', '').replace('\'', '').replace(',', '').split() for line in lines]
errors = []
for line in lines:
    gene = line[1]
    r = float(line[2])
    rr = float(result[result.index == gene])
    errors.append(np.abs(r - rr))

# Finds maximal error, which is the difference in results
errors = sorted(errors)
if errors[-1] < 0.0000000001:
    print("Test passed! Maximal error is", errors[-1])
else:
    print("Test failed")




