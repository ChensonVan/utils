import matplotlib.pyplot as plt

def plot_data_distribution(X, y):
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)  
    plt.title('data distribution') 
    plt.show()  
    
# 2D
def plot_decision_boundary(clf, X, title=''):  
    # 设定最大最小值，附加一点点边缘填充  
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5  
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5  
    h = 0.01  
  
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  
  
    # 用预测函数预测一下  
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  
    Z = Z.reshape(xx.shape)  
  
    # 然后画出图  
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)  
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 
    if title:
        plt.title("Logistic Regression") 
    plt.show()