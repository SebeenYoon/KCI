import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

f = open("./segmented_label.txt", "r")
contents = f.readlines()
coor = []
for line in contents:
    if line[0] == "0":
        tmp = line.split()
        tmp = list(map(int, tmp[1:-1]))
        coor.append(tmp)
coord = np.array(coor)
coor_x = []
coor_y = []
for k in range(len(coord)):
    coor_x += coord[k][0::2]
    coor_y += coord[k][1::2]
coor_x = np.array(coor_x)
coor_y = np.array(coor_y)
point_data = np.column_stack([coor_x, coor_y])

k = 6
model = KMeans(n_clusters=k)
model.fit(point_data)
predict = model.predict(point_data)


colors = plt.cm.Spectral(np.linspace(0, 1, len(set(predict))))
k_means_labels = model.labels_
k_means_cluster_centers = model.cluster_centers_


fig = plt.figure(figsize=(6, 4))
# plot 생성
ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(6), colors):
    my_members = (k_means_labels == k)

    # 중심 정의
    cluster_center = k_means_cluster_centers[k]

    # 중심 그리기
    ax.plot(coor_x[my_members], coor_y[my_members], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

print(np.array(k_means_cluster_centers, dtype=int))
print(k_means_labels)

distances =[]
from shapely.geometry import Point
from itertools import *

point_list = list(combinations(k_means_cluster_centers, 2))
for k in point_list:
    distances.append(Point(k[0]).distance(Point(k[1])))
print(distances)

ax.set_title('Dap Clustering')
ax.set_xticks(())
ax.set_yticks(())
plt.show()

print()

