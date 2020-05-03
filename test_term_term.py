# Run test using "python ./test_term_term.py"
# It tests only second part by taking the table from slide 69 and couning all similarities with input word
import numpy as np

dic = {"I": [0.044, 0.059, 0.0],
       "like": [0.119, 0.0, 0.0],
       "data": [0.044, 0.059, 0.0],
       "science": [0.119, 0.0, 0.0],
       "hate": [0.0, 0.159, 0.0],
       "want": [0.0, 0.0, 0.238],
       "A": [0.0, 0.0, 0.238]}

for k in dic.keys():
    dic[k] = np.array(dic[k])

input_word = 'I'

fstd = np.sqrt(np.sum(dic[input_word] ** 2))
res = {}

for k in dic.keys():
    std = np.sqrt(np.sum(dic[k] ** 2))
    cov = np.sum(dic[k] * dic[input_word])
    res[k] = cov / (std * fstd)

print(sorted(res.items(), key=lambda x: -x[1]))
# Tested
# [('I', 1.0), ('data', 1.0), ('hate', 0.8016274750930587), ('like', 0.5978238797304165),
# ('science', 0.5978238797304165), ('want', 0.0), ('A', 0.0)]
# Actual
# [('I', 1.0), ('data', 1.0), ('hate', 0.7999999999999999), ('like', 0.6),
# ('science', 0.6), ('want', 0.0), ('A', 0.0)]
