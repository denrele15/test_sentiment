import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv
import sys
import pandas

import matplotlib.pyplot as plt

import matplotlib.animation as animation
from matplotlib import style
import time


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


style.use = ("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pullData = open("output.txt","r").read()
    lines = pullData.split('\n')

    xs = []
    ys = []

    x = 0
    y = 0

    for l in lines:
        x += 1
        if "pos" in l:
            y += 1
        elif "neg" in l:
            y -= 1

            xs.append(x)
            ys.append(y)

            ax1.clear()
            ax1.plot(xs, ys)


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()








#BarChart
# for sentence in lines2:
#             ss = sid.polarity_scores(sentence)
#             w = ('{0}, {1},'.format(k, ss[k]))
#             for k in ss:
#                 plt.bar(k, w, Label='Bar Chart')
#                 plt.savefig('graph.png')
#                 plt.show()


#Scatterplot
# x = [k]
# y = [ss[k]]
# plt.scatter(x,y, label='Sentiments', color='red', s=25, marker="o")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Scatterplot')
# plt.legend()
# plt.show()