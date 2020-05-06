# Run test using "python ./test_slides_data.py"
# It tests only second part by taking the table from slide 69 and couning all similarities with input word
import pandas as pd
import numpy as np

input_term = 'I'
with open('./resources/slides_data.txt', 'r') as f:
    lines = f.read().splitlines()

lines = [line.split() for line in lines]
files = [line[0] for line in lines]

genes = []
for line in lines:
    for word in line[1:]:
            genes.append(word)

genes = list(set(genes))
df = pd.DataFrame(columns=files, index=genes).fillna(value=0)

# Created df as in slide 66
for line in lines:
    for word in line[1:]:
            df[line[0]][df.index == word] += 1.

doc_sizes = np.array([len(line) - 1 for line in lines])
tf = df / doc_sizes
idf = np.log10(len(lines) * 1. / np.sum(df > 0, axis=1))
tfidf = tf * idf[:, np.newaxis]
# Print this tfidf to compare it with the one in the slides
print(tfidf)

dic = {"I": [0.044, 0.059, 0.0],
       "like": [0.119, 0.0, 0.0],
       "data": [0.044, 0.059, 0.0],
       "science": [0.119, 0.0, 0.0],
       "hate": [0.0, 0.159, 0.0],
       "want": [0.0, 0.0, 0.238],
       "A": [0.0, 0.0, 0.238]}

for k in dic.keys():
    dic[k] = np.array(dic[k])


fstd = np.sqrt(np.sum(dic[input_term] ** 2))
res = {}

for k in dic.keys():
    std = np.sqrt(np.sum(dic[k] ** 2))
    cov = np.sum(dic[k] * dic[input_term])
    res[k] = cov / (std * fstd)

print(sorted(res.items(), key=lambda x: -x[1]))
# Tested
# [('I', 1.0), ('data', 1.0), ('hate', 0.8016274750930587), ('like', 0.5978238797304165),
# ('science', 0.5978238797304165), ('want', 0.0), ('A', 0.0)]
# Actual
# [('I', 1.0), ('data', 1.0), ('hate', 0.7999999999999999), ('like', 0.6),
# ('science', 0.6), ('want', 0.0), ('A', 0.0)]
