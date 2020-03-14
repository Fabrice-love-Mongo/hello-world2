import numpy as np

def createDataSet():
    group = np.array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5],[1.1,1.0],[0.5,1.5]])
    labels = np.array(['A','A','B','B','A','B'])
    return group,labels
def KNN_classify(k,dis,X_train,x_train,Y_test):
    assert dis == 'M'or dis == 'E','dis must E or M,E means Euclidean Metric,M means Manhattan distance'
    num_test = Y_test.shape[0]
    num_train = X_train.shape[0]
    labels_test = []
    if (dis == 'E'):
        for i in range(num_test):
            distances = np.sqrt(np.sum(np.square(X_train - np.tile(Y_test[i,:],(num_train,1))),axis=1))
            index_nearest = np.argsort(distances)
            index_k = index_nearest[:k]
            count_A=0
            count_B=0
            for j in range(k):
                if x_train[index_k[j]] == 'A':
                    count_A = count_A + 1
                if x_train[index_k[j]] == 'B':
                    count_B = count_B + 1
            if count_A >= count_B:
                labels_test.append('A')
            else :
                labels_test.append('B')
    return labels_test
if __name__=='__main__':
    group,labels = createDataSet()
    y_test_pred = KNN_classify(1,'E',group,labels,np.array([[1.0,2.1],[0.4,2.0]]))
    print(y_test_pred)