# Importing the packages for data analyses purpose

import matplotlib.pyplot as plt
import numpy as np
import pickle

# Frequency Distribution pickle files

FREQ_DIST_FILE = 'dataset-final-processed-freqdist.pkl'
BI_FREQ_DIST_FILE = 'dataset-final-processed-freqdist-bi.pkl'

# Opening the pickle file to check the most common uni-grams

with open(FREQ_DIST_FILE, 'rb') as pkl_file:
    freq_dist = pickle.load(pkl_file)
unigrams = freq_dist.most_common(20)
unigrams

# Plotting the uni-grams

plt.figure(1, [10, 7])
x = np.array(range(0, 40, 2))
y = np.array([i[1] for i in unigrams])
my_xticks = [i[0] for i in unigrams]
plt.xticks(x, my_xticks, rotation=90)
plt.plot(x, y)
plt.show()

# Opening the pickle file to check the most common Bi-grams

with open(BI_FREQ_DIST_FILE, 'rb') as pkl_file:
    freq_dist = pickle.load(pkl_file)
bigrams = freq_dist.most_common(20)
bigrams

# Plotting the Bi-grams

plt.figure(1, [10, 7])
x = np.array(range(0, 40, 2))
y = np.array([i[1] for i in bigrams])
my_xticks = [', '.join(i[0]) for i in bigrams]
plt.xticks(x, my_xticks, rotation=90)
plt.plot(x, y)
plt.show()

# Plotting Zipf\'s Law

with open(FREQ_DIST_FILE, 'rb') as pkl_file:
    freq_dist = pickle.load(pkl_file)
unigrams = freq_dist.most_common(100)
log_ranks = np.log(range(1, 101))
log_freqs = np.log([i[1] for i in unigrams])
z = np.polyfit(log_ranks, log_freqs, 1)
p = np.poly1d(z)
p

plt.figure(3, [8,6])
plt.plot(log_ranks, log_freqs, 'ro')
plt.plot(log_ranks,p(log_ranks),'b-')
plt.xlabel('log (Rank)')
plt.ylabel('log (Freqeuncy)')
plt.title('Zipf\'s Law')
plt.show()

networks = ['Basic Neural Networks' 'CNN', 'Bi-Directional RNN', 'RNN', 'LSTM']
accuracies = [63.48, 72, 80, 98]
plt.figure(4, [8,6])
plt.bar(range(len(networks)), accuracies, align='center', alpha=0.5)
plt.xticks(range(len(networks)), networks, rotation=90)
plt.ylabel('Accuracy')
plt.title('Comparison of Various Neural Networks used for analysis')
plt.ylim([60, 90])
plt.show()