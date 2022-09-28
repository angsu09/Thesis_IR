import numpy as np
import matplotlib.pyplot as plt

labels = ['Top 1', 'Top 5', 'Top 10']
with_cluster_means = [40, 67, 70]
without_cluster_means = [31, 63, 69]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, with_cluster_means, width, label='With Clusters')
rects2 = ax.bar(x + width/2, without_cluster_means, width, label='Without Clusters')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
plt.yticks(np.arange(0, 100, 10.0))
fig.tight_layout()


plt.show()