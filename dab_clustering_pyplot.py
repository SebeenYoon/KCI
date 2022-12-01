dir = "./Measurement/cropped-removebg-preview (2).png"

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


img = cv2.imread(dir)
img_trans = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_trans = cv2.inRange(img_trans, (0,0,0), (200,15,200))
plt.imshow(img_trans)
plt.show()

adhesive = np.where(img_trans == 0)
rest = np.where(img_trans == 255)
print(np.sum(rest)+np.sum(adhesive))
print(round(np.sum(adhesive)/(np.sum(rest)+np.sum(adhesive))*100 ,2))


y = 125
x = 62

cropped_img = img_trans[y:y+560,x:x+280]


k_xy = np.where(cropped_img == 0)
k_x = k_xy[0]
k_y = k_xy[1]

point_data = np.column_stack([k_x,k_y])

def cluster(*args):
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
        #
        # # 중심 그리기
        # # ax.plot(point_data[my_members], point_data[my_members], 'w', markerfacecolor=col, marker='.')
        # ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    centers = np.array(k_means_cluster_centers, dtype=int)
    return centers
centers = cluster(point_data)

# centers 정렬 ++++++++++++++++++++++++++
# centers' x arrange

def sort(x):
    a = []
    for i in range(6):
        a.append(x[i][0])
    a = np.array(a)
    s = a.argsort()
    centers_sorted = x[s]
    # centers crop with size 2
    b = []
    for k in range(3):
        b.append(centers_sorted[2*k:2*k+2])
    b = np.array(b)
    # in cropped centers, arrange y values
    c = []
    for i in range(3):
        if b[i][0][1] > b[i][1][1]:
            c.append(b[i])
        else:
            c.append(b[i][::-1])
    c = np.array(c)
    c = np.reshape(c,(6,-1))
    return c
centers = sort(centers)

print("centers\n", sort(centers))



from shapely.geometry import Point



distances = []
for k in range(0, int(len(centers)),2):
    distances.append(round(Point(centers[k]).distance(Point(centers[k+1]))*1.5,2))
for k in range(0, int(len(centers))-2):
    distances.append(round(Point(centers[k]).distance(Point(centers[k+2]))*1.5,2))
distances = np.array(distances)
print("distances",distances)
centers_x = centers[:,0]
centers_y = centers[:,1]

for idx in range(0, len(centers), 2):
    plt.plot((centers_x[idx], centers_x[idx+1]),(centers_y[idx],centers_y[idx+1]), marker="o",linestyle=":")
    # for k in range(3):
    #     plt.text((centers_x[idx], centers_x[idx+1])/2,((centers_y[idx],centers_y[idx+1])/2), distances[k])
for idx in range(0, len(centers)-2):
    plt.plot((centers_x[idx], centers_x[idx+2]),(centers_y[idx],centers_y[idx+2]), marker="s",linestyle="--")
    # plt.text((centers_x[idx], centers_x[idx+2]) / 2, ((centers_y[idx], centers_y[idx+2]) / 2), distances[idx])
print(centers[0])
print(centers[1])



cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
cropped_img = cv2.flip(cropped_img,0)
plt.imshow(cropped_img)
# img_trans = cv2.flip(img_trans, 0)
# ax.set_title('Dap Clustering')
# ax.set_xticks(())
# ax.set_yticks(())
plt.show()