'''
point cloud data is stored as a 2D matrix
each row has 3 values i.e. the x, y, z value for a point

Project has to be submitted to github in the private folder assigned to you
Readme file should have the numerical values as described in each task
Create a folder to store the images as described in the tasks.

Try to create commits and version for each task.

'''
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()

def show_scatter(x, y):
    plt.scatter(x, y)
    plt.show()

def get_ground_level(pcd):
    """
    Bestämmer marknivån baserat på 5:e percentilen av z-värdena.
    """
    return np.percentile(pcd[:, 2], 5)

#%% read file containing point cloud data
pcd = np.load("dataset1.npy")

#%% show downsampled data in external window
show_cloud(pcd)
#show_cloud(pcd[::10]) # keep every 10th point

#%% remove ground plane

'''
Task 1 (3)
find the best value for the ground level
One way to do it is useing a histogram 
np.histogram

update the function get_ground_level() with your changes

For both the datasets
Report the ground level in the readme file in your github project
Add the histogram plots to your project readme
'''
est_ground_level = get_ground_level(pcd)
print(f"Beräknad marknivå: {est_ground_level}")

pcd_above_ground = pcd[pcd[:, 2] > est_ground_level]

plt.hist(pcd[:, 2], bins=50, color='blue', alpha=0.7)
plt.title("Histogram över Z-värden (Dataset 1)")
plt.xlabel("Höjd (z)")
plt.ylabel("Antal punkter")
plt.savefig("histogram_dataset1.png")
plt.close()

show_cloud(pcd_above_ground)


#%%
'''
Task 2 (+1)

Find an optimized value for eps.
Plot the elbow and extract the optimal value from the plot
Apply DBSCAN again with the new eps value and confirm visually that clusters are proper

https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/

For both the datasets
Report the optimal value of eps in the Readme to your github project
Add the elbow plots to your github project Readme
Add the cluster plots to your github project Readme
'''

unoptimal_eps = 10
kdtree = KDTree(pcd_above_ground)
distances, _ = kdtree.query(pcd_above_ground, k=5)
mean_distances = np.mean(distances[:, 1:], axis=1)
mean_distances.sort()

plt.plot(mean_distances)
plt.title("Elbow-metod för optimal eps")
plt.xlabel("Datapunkter sorterade efter avstånd")
plt.ylabel("Medelavstånd")
plt.savefig("elbow_plot.png")
plt.close()

optimal_eps = mean_distances[int(len(mean_distances) * 0.9)]
print(f"Optimalt eps-värde: {optimal_eps}")

clustering = DBSCAN(eps=optimal_eps, min_samples=5).fit(pcd_above_ground)
labels = clustering.labels_

plt.scatter(pcd_above_ground[:, 0], pcd_above_ground[:, 1], c=labels, cmap='viridis', s=2)
plt.title("DBSCAN-klustring efter optimering")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("dbscan_clusters.png")
plt.close()


#%%
'''
Task 3 (+1)

Find the largest cluster, since that should be the catenary, 
beware of the noise cluster.

Use the x,y span for the clusters to find the largest cluster

For both the datasets
Report min(x), min(y), max(x), max(y) for the catenary cluster in the Readme of your github project
Add the plot of the catenary cluster to the readme

'''
unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
largest_cluster = unique_labels[np.argmax(counts)]

catenary_cluster = pcd_above_ground[labels == largest_cluster]
min_x, min_y = np.min(catenary_cluster[:, :2], axis=0)
max_x, max_y = np.max(catenary_cluster[:, :2], axis=0)

print(f"Catenary Cluster Boundaries: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")

plt.scatter(catenary_cluster[:, 0], catenary_cluster[:, 1], c='red', s=2)
plt.title("Catenary-kluster")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("catenary_cluster.png")
plt.close()
# %%
