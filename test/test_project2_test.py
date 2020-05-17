# Run test using "python ./test_project2_test.py gene_egfr+_gene". It takes very long to test the new file.
# Tests algorithm for the first professor's file

import sys
import pandas as pd
import numpy as np

# Query term for term-term relevance
input_term = sys.argv[1]

result_directory = "result"

with open('../resources/project2_egfr.txt', 'r') as f:
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


def process_line(l):
    l = eval(l)
    return (l[0][1], l[1])

lines = [process_line(line) for line in lines]

true_values = {}
errors = {}
for line in lines:
    gene = line[0]
    r = float(line[1])
    true_values[gene] = r
    try:
        rr = float(result[result.index == gene])
    except:
        print(result[result.index==gene])
        print(gene, r, line)

    errors[gene] = (np.abs(r - rr))

# Finds maximal error, which is the difference in results

gene_sorted = sorted(errors.keys(), key=lambda x: errors[x])

for gene in gene_sorted[-10:]:
    print(gene, errors[gene])
    print(float(result[result.index == gene]), true_values[gene])

if errors[gene_sorted[-1]] < 1e-5:
    print("Test passed!")
else:
    print("Test failed")




