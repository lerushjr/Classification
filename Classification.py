import pandas as pd
raw_data = pd.read_csv('heart.csv') 
corr_matrix = raw_data.corr()
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
heatmap = sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
features = raw_data.iloc[:,0:-1] 
features.head()
target = features['sex']
print(target)
from sklearn.model_selection import train_test_split
##split features and targets into training and test sets, where the test set is 33% of the data.
##random_state allows for repeatable experiments as it gives the starting point for the random selecting of the test records
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=3442)
from sklearn.neighbors import KNeighborsClassifier

#define model parameters
n_neighbors = 5 #<--Here we define the number of neighbors our model will compare

#Here we fit our training data to the model
knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train,y_train)##using the score method we'll test the accuracy of our model using the test data set we made earlier
results_knn = knn.score(X_test, y_test)
#Print the results of the model as a percentage here
dp = raw_data.iloc[2,:]
new_data_point = np.array([41,0,1,130,204,0,0,172,0,1.4,2,0,2]).reshape(1,-1)
result = knn.predict(new_data_point)#<---what knn method would you use to predict the target of new_data_point?
print(result)
def knn_k_tester(k_list, X_train, X_test, y_train, y_test):
    '''This function will take a list of integers that define k neighbors to evalutate
    inputs:
    k_list: List of integer values
    X_train: Training dataset as pd.dataframe
    X_test: Test dataset as pd.dataframe
    y_train: Training labels as pd.series
    y_test: Test labels as pd.series
    
    Outputs:
    Results: List of k values and corresponding model accuracy
    '''
    from sklearn.neighbors import KNeighborsClassifier
    k_lst = []
    accuracy = []
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
        results = knn.score(X_test, y_test)
        k_lst.append(k)
        accuracy.append(results*100)
    results = [k_list,accuracy]    
    return results
#first define our list of k values to test
k_list = list(range(1,21,1))
knn_test = knn_k_tester(k_list=k_list, X_train=X_train, X_test=X_test,y_train=y_train,y_test=y_test)
plt.scatter(x=knn_test[0],
            y=knn_test[1])
plt.xlabel('k_neighbors')
plt.ylabel('accuracy %')
plt.title('KNN Accuracy vs. k_neighbors')
plt.show()
from sklearn.metrics import plot_confusion_matrix
confusion=plot_confusion_matrix(knn, X_test, y_test)
plt.xlabel('Predicted Heart Disease')
plt.ylabel('True Heart Disease')
plt.xticks([0,1], ['No', 'Yes'])
plt.yticks([0,1], ['No', 'Yes'])
plt.show()
from sklearn.neural_network import MLPClassifier ##https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
#setup model parameters
activation         = 'relu' ##<----Accepted activation types are 'identity', 'logistic', 'tanh', 'relu'
random_state       = 12345
hidden_layer_sizes = (10,12,15,17,20) ##<---This is where we can adjust the number of layers, and the number of nodes in each layer
max_iter           = 50000
solver             = 'adam'
n_iter_no_change = 100

mlp = MLPClassifier(activation=activation, 
                    random_state=random_state, 
                    hidden_layer_sizes=hidden_layer_sizes, 
                    max_iter=max_iter,
                    solver=solver,
                   verbose=False,
                   n_iter_no_change=n_iter_no_change)

mlp.fit(X=X_train, y=y_train)
import matplotlib.pyplot as plt

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)
#First we need a list of the layer sizes of our model
layer_sizes = [mlp.n_features_in_] + list(mlp.hidden_layer_sizes) + [mlp.n_outputs_]

#next let's use the fuction we just defined to plot the model
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .9, layer_sizes=layer_sizes)
#Test accuracy of model
results_mlp = mlp.score(X_test, y_test)
print(f'Model accuracy: {results_mlp*100}%')
confusion = plot_confusion_matrix(mlp, X_test, y_test)
plt.xlabel('Predicted Heart Disease')
plt.ylabel('True Heart Disease')
plt.xticks([0,1], ['No', 'Yes'])
plt.yticks([0,1], ['No', 'Yes'])
plt.show()
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import plot_roc_curve
y_mlp_pred = mlp.predict(X_test)
y_mlp_pred_prob = mlp.predict_proba(X_test)
false_pos_rate, true_pos_rate, roc_thresholds = roc_curve(y_true=y_test, 
                                                          y_score=y_mlp_pred_prob[:,1], 
                                                          pos_label=mlp.classes_[1], 
                                                          drop_intermediate=False)
ax = plt.subplot()
roc_plot = plot_roc_curve(mlp,X_test,y_test,ax=ax)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
plt.show()


