import pandas as pd
import wordcloud
import matplotlib.pyplot as plt
import numpy as np

data_v2 = pd.read_csv('dataset-final-processed.csv', names=["id", "sentiment", "review"])
common_words=''

# for positive reviews

for np.where((data_v2['sentiment'] == '1')):
    for j in data_v2.review:
        for np.where((data_v2['sentiment'] == '0')):
            for i in data_v2.review:
                if j not in i:
                    i = str(i)
                    tokens = i.split()
                    common_words += " ".join(tokens)+" "
wordcloud = wordcloud.WordCloud().generate(common_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# for negative reviews
for np.where((data_v2['sentiment'] == '0')):
    for i in data_v2.review:
        for np.where((data_v2['sentiment'] == '1')):
            for j in data_v2.review:
                if i not in j:
                    j = str(j)
                    tokens = j.split()
                    common_words1 += " ".join(tokens)+" "
wordcloud = wordcloud.generate(common_words1)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

