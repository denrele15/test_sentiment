import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv
import sys
import pandas
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

df = pandas.read_csv('data.csv', encoding='ISO-8859-1')
print(df['comment_text'])
lines2 = df['comment_text']

sys.stdout = open("output.txt", "w", encoding='ISO-8859-1')
sid = SentimentIntensityAnalyzer()
i = -1

for sentence in lines2:
    print('     ' + i.__str__() + '     ,' + sentence)  # outputs sequential no of each line of sentence
    i = i + 1
    ss = sid.polarity_scores(sentence)
    for k in ss:
        print('{0}, {1},'.format(k, ss[k]), end='::   ')

for sentence in lines2:
            ss = sid.polarity_scores(sentence)
            for k in ss:
                plt.bar(k, ss[k])
                plt.savefig('graph.png')
                plt.show()

# fig, ax1 = plt.subplots()
# ax1.bar(k.keys(), ss[k].values())
# fig.autofmt_xdate()
