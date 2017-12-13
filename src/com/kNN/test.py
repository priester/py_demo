import knn
import matplotlib
import matplotlib.pylab as plt

from imp import reload

reload(knn)
datingDataMat,datingLabels = knn.file2matrix('datingTestSet2.txt')

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
# plt.show()

# group,labels = knn.createDataSet();
# print(group);
# print(labels);

# knn.classify0([0,0],group,labels,3 )
# print(knn.classify0([0,0],group,labels,3))

normDataSet , ranges, minVals = knn.autoNorm(datingDataMat);
print(normDataSet);
print(ranges);
print(minVals);





