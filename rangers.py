import numpy as np
from sklearn.cluster import KMeans

from common import encode_as_lines
from sklearn.metrics import roc_curve

#---------------simple methods in one function---------------------------
def to_binary_by_KS(X, y):
    '''
        Recodes the input row as to classes by point corresponds to KS stat.
        Look idea description at simple_methods.ipynb.
        Inputs:
            X - np.array of shape (n_sample, ), the series which should be encoded;
            y - np.array of shape (n_sample, ), the classes of observations.
        Output
            np.array of shape (n_sample, ) which desribes ranges as 
            (-inf, KS_point] (KS_point, inf)
    '''
    
    var_min = np.min(X)
    var_max = np.max(X)
    
    tpr, fpr, thresholds = \
       (res[1:] for res in roc_curve(y, (X - var_min)/(var_max - var_min)))

    F_p = 1 - tpr
    F_n = 1 - fpr
    
    KS_ind = np.argmax(np.abs(fpr - tpr))
    
    bins = np.array([thresholds[KS_ind]*(var_max - var_min) + var_min])
    return encode_as_lines(X, bins)
#------------------------------------------------------------------------




class kmeans_ranges():
    '''
        Applying the kmeans algorithm for 1 dimention data 
        wich helps divide data into ranges.
        Look idea description at disk_kmeans_example.ipynb
    '''
    
    
    def __init__(self, max_clusters = 10, kmeans_kwarg = {}):
        '''
        inputs:
            max_clusters - maximum number of clusters wich will be fitted;
            kmeans_kwarg - optional arguments for kmeans class,
                            but arguments n_init, n_clusters and init
                            will be ignored.
        '''
        
        self.max_clusters = max_clusters
        self.kmeans_kwarg = kmeans_kwarg
        self.kmeans_insts = []
        
    def build_kmeans_instances(self, var_range):    
        
        '''
            Creates and initialise sklearn.cluster.kmeans instances
            for different number of clusters.
             list of sklearn.cluster.kmeans instances for different
            count of clusters. The same object is placed at the
            self.kmeans_insts.
            
            inputs:
                var_range - numpy.array with shape (1,) 
                            the range which will used for fitting
        '''
        
        self.kmeans_insts = []
    
        self.kmeans_kwarg['n_init'] = 1
        for n_clusters in range(1, self.max_clusters + 1):
            # inital clusters positions initialises as
            # as 1-st, 2-nd, ..., (n_clusters - 1)-s 
            # quatiles of n_clusters range
            self.kmeans_kwarg['init'] = np.quantile(
                var_range, 
                [[i/(n_clusters + 1)] for i in range(1, n_clusters + 1)]
            )
            self.kmeans_kwarg['n_clusters'] = n_clusters


            self.kmeans_insts.append(KMeans(
                **self.kmeans_kwarg
            ).fit(var_range[:, np.newaxis]))

        self.SSE_arr = np.array(
            [kmean_inst.inertia_ for kmean_inst in self.kmeans_insts])
            
        return self
        
    
    def elbow_choose(self):
        '''
            Selects the best number os cluscters according to Elbow Method.
            If between i and i-1 cluster num we gets the most relative
            decreasing of SSE thet i is the best count of clusters.
            
            output:  the best clusters number to choose accordig to the
                     elbow method for kmeans algorithm. The same value
                     contains in self.best_idx variable.
        '''
        self.best_idx = np.argmax(self.SSE_arr[0:-1]/self.SSE_arr[1:None]) + 1
        return self
    
    
    def init_bins(self, var_range):
        
        '''
            Init bins for selected best sklearn.cluster.kmeans.
            Beans is values which will used in predict function,
            in numpy.digitize for getted range.
            
            Inputs:
                var_range - numpy.array with shape (1,) 
                            centroid values will used as a bins.
        '''
        
        y_hat = self.kmeans_insts[self.best_idx].predict(var_range[:, np.newaxis])
        self.bins = []
                
        for cluster_mark in np.unique(y_hat)[0:-1]:
            self.bins.append(np.max(var_range[y_hat == cluster_mark]))
            
        return self
            
    
        
    def fit(self, var_range):
        '''
            Fit the instance on given data. Fitting
            includes building the instances of sklearn.clusters.kmeans,
            selection of the best clusters number with Elbow method and
            creating bins for discretization in predict method.
        
            inputs:
                var_range - numpy.array with shape (1,) 
                            the range which will used for fitting
        '''
        
        self.build_kmeans_instances(var_range)
        self.elbow_choose()
        self.init_bins(var_range)
        
        return self
        
        
    
    def transform(self, var_range):
        '''
            Transforms input series with fitted above discretisator.
            
            Inputs:
                var_range - numpy.array with shape (1,) 
                            the range which will used for transforming
        '''
        return encode_as_lines(var_range, self.bins)
    
    
    
    def fit_transform(self, var_range):
        '''
            Applies fit and trainsform methods to the given data range.
            
            Inputs:
                var_range - numpy.array with shape (1,) 
                            the range which will used for fitting and trainsforming
        '''
        
        self.fit(var_range)
        return self.transform(var_range)