import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn import metrics
import pyfpgrowth
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from clustviz.chameleon.chameleon import cluster

### First Dataset Preprocessing ###
def preprocess_1st():
    
    # Read dataset
    df1st = pd.read_csv("/home/canberk/Desktop/2D.csv", sep = " ")
    
    # Check if any nan value in dataset
    print("\nIs there any nan value in 2D dataset:",df1st.isna().values.any())
    # There is no nan value in 51D dataset
    
    # Since all values are between 0 and 1, no need to normalization
    # Also there is no categorical value in dataset 
    # therefore no need to turn them into numerical values
    return df1st


### Second Dataset Preprocessing ###
def preprocess_2nd():
    
    # Read dataset
    df2nd = pd.read_csv("/home/canberk/Desktop/51D.csv", sep = ",")
    
    # Copy dataset to do operations on it
    c2nd = df2nd.copy()
    
    # Check if any nan value in 51D dataset
    print("\nIs there any nan value in 51D dataset:",c2nd.isna().values.any())
    # There is no nan value in 51D dataset
    
    # Also there is no categorical value in dataset 
    # therefore no need to turn them into numerical values
    
    # Product_Code feature in dataset will not be beneficial because
    # it's unique and not useful for clustering therefore I drop it.
    print("\nProduct_Code Feature is unique and not useful for clustering therefore I drop it.")
    c2nd.drop(['Product_Code'], axis=1, inplace=True)
    print("\nProduct_Code column is dropped\n")
    
    # There are values like 0 and 59 in dataset, bigger values can have bigger effects on clustering
    # therefore I need to standardize values of numerical features 
    
    scaler = StandardScaler()
    
    c2nd = pd.DataFrame(scaler.fit_transform(c2nd), columns = c2nd.columns)
    
    return c2nd

### KMeans ###
def k_means(df):
    k_values = range(1,10)
    
    if df.shape[1] < 20:
        inertias = []
        # Create KMeans model with k clusters
        for i in k_values:
            model = KMeans(n_clusters=i)
            model.fit(df)
            inertias.append(model.inertia_)
        
        # Plot inertias and k values to find elbow point 
        # in order to find optimum k value
        plt.plot(k_values, inertias, '-o', color='turquoise')
        plt.xlabel('k values')
        plt.ylabel('inertia')
        plt.xticks(k_values)
        plt.show()
        # We can see that elbow point is 3
        # k is 3 and we got labels of clusters    
        model = KMeans(n_clusters=3)
        clusters = model.fit_predict(df)
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        sns.scatterplot(data = df)
        plt.show()
        sil_sc = silhouette_score(df, clusters, metric='euclidean')
        print("\nThe Silhouette Coefficient for KMeans with 2D Dataset(DS1): %.3f\n", sil_sc)
        print('\n')
    
    else:
        inertias = []
        # Create KMeans model with k clusters
        for i in k_values:
            model = KMeans(n_clusters=i)
            model.fit(df)
            inertias.append(model.inertia_)
        
        # Plot inertias and k values to find elbow point 
        # in order to find optimum k value
        plt.plot(k_values, inertias, '-o', color='turquoise')
        plt.xlabel('k values')
        plt.ylabel('inertia')
        plt.xticks(k_values)
        plt.show()
        # We can see that there is no big change after k is 3 
        # therefore k is 3
        model = KMeans(n_clusters=3)
        clusters = model.fit_predict(df)
        df['clusters'] = clusters
        # Since visualization of 2nd dataset is not wanted
        # I just print clusters and how many points belong to them
        print("Clusters and number of points belong to them(KMeans)")
        print(df['clusters'].value_counts())
        
        sil_sc = silhouette_score(df, clusters, metric='euclidean')
        print("\nThe Silhouette Coefficient for KMeans with 51D Dataset(DS2): %.3f\n", sil_sc)
        print('\n')
      
def fp_growth(df):
    patterns = pyfpgrowth.find_frequent_patterns(df.columns, 10)
    rules = pyfpgrowth.generate_association_rules(patterns,0.6)
    print("FREQUENT PATTERNS")
    print(patterns)
    print('\n')
    print("ASSOCIATION RULES")
    print(rules)
    
    # I find optimal epsilon value by using NearestNeighbors,
    # it gives distances between each point and elbow point on  
    # its graph gives optimal epsilon.
    # Related article: https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf
def find_optimal_eps(df,min_samples):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(df)
    distances, _ = neighbors_fit.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()
    
### DBSCAN ###
def dbscan(df):
    
    if df.shape[1] < 20:
        # For 2D data, I will use DBSCAN’s default value of min_samples = 4
        min_samples = 4
        find_optimal_eps(df,min_samples)
        # According to elbow point, optimal eps value is 0.05 for 2D dataset 
        true_labels = DBSCAN(eps=0.05, min_samples=min_samples).fit_predict(df)
      
        plt.title("DBSCAN Clustering for DS1")
        sns.scatterplot(df.iloc[:,0], df.iloc[:,1], hue=["cluster-{}".format(x) for x in true_labels])
        plt.show()
        db_sc1 = silhouette_score(df, true_labels, metric='euclidean')
        print("\nThe Silhouette Coefficient for DBSCAN with 2D Dataset(DS1): %.3f\n" % db_sc1)
        
    else:
        # 2nd data has more than 2 dimensions therefore I choose min_samples = 2*dim, 
        # where dim = the dimension of the dataset
        min_samples = 2*df.shape[1]
        find_optimal_eps(df,min_samples)
        # According to elbow point, optimal eps value is 3 for 51D dataset
        true_labels = DBSCAN(eps=3, min_samples=min_samples).fit_predict(df)
       
        # Visualization for 51D dataset is not wanted therefore 
        # I print clusters and number of points belong to them
        cnt = Counter(true_labels)
        label_counts = {i: cnt[i] for i in true_labels}
        print("Clusters and number of points belong to them(DBScan)")
        print(label_counts)
        
        db_sc2 = silhouette_score(df, true_labels, metric='euclidean')
        print("\nThe Silhouette Coefficient for DBSCAN with 51D Dataset(DS2): %.3f\n" % db_sc2)

### Chameleon ###
def chameleon(df):
    
    if df.shape[1] < 20:
        chameleon_df,_ = cluster(df, k=3, knn=10, m=20, verbose2=False, plot=False)
        chameleon_df.plot.scatter(x=0, y=1, c=chameleon_df['cluster'], colormap='gist_rainbow')
        ch_sc1 = silhouette_score(df, df['cluster'], metric='euclidean')
        print("\nThe Silhouette Coefficient for Chameleon with 2D Dataset(DS1): %.3f\n" % ch_sc1)

    else:
        chameleon_df,_ = cluster(df, k=15, knn=10, m=20, verbose2=False, plot=False)
        unique, counts = np.unique(chameleon_df['cluster'], return_counts=True)
        print("Clusters and number of points belong to them(Chameleon)")
        print(dict(zip(unique, counts)))
        ch_sc2 = silhouette_score(df, df['cluster'], metric='euclidean')
        print("\nThe Silhouette Coefficient for Chameleon with 51D Dataset(DS2): %.3f\n" % ch_sc2)

if __name__ == "__main__":
    df_1st = preprocess_1st()
    df_2nd = preprocess_2nd()
    
    ### FPGrowth
    # 2D Dataset
    fp_start = time.time()
    fp_growth(df_1st)
    fp_end = time.time()
    print("Computational Time of FPGrowth Clustering Technique with 2D Dataset(DS1): {}".format(fp_end-fp_start))
    print('\n')
    
    # 51D Dataset
    fp_start = time.time()
    fp_growth(df_2nd)
    fp_end = time.time()
    print("Computational Time of FPGrowth Clustering Technique with 51D Dataset(DS2): {}".format(fp_end-fp_start))
    print('\n')
    
    ### KMeans
    # 2D Dataset
    k_means_start = time.time()
    k_means(df_1st)
    k_means_end = time.time()
    print("Computational Time of KMeans Clustering Technique with 2D Dataset(DS1): {}".format(k_means_end-k_means_start))
    print('\n')
    
    # 51D Dataset
    k_means_start = time.time()
    k_means(df_2nd)
    k_means_end = time.time()
    print("Computational Time of KMeans Clustering Technique with 51D Dataset(DS2): {}".format(k_means_end-k_means_start))
    print('\n')
    
    ### DBSCAN
    # 2D Dataset
    db_start = time.time()
    dbscan(df_1st)
    db_end = time.time()
    print("Computational Time of DBScan Clustering Technique with 2D Dataset(DS1): {}".format(db_end-db_start))
    print('\n')
    
    # 51D Dataset
    db_start = time.time()
    dbscan(df_2nd)
    db_end = time.time()
    print("Computational Time of DBScan Clustering Technique with 51D Dataset(DS2): {}".format(db_end-db_start))
    print('\n')
    
    ### Chameleon
    # 2D Dataset
    chm_start = time.time()
    chameleon(df_1st)
    chm_end = time.time()
    print("Computational Time of Chameleon Clustering Technique with 2D Dataset(DS1): {}".format(chm_end-chm_start))
    print('\n')
    
    # 51D Dataset
    chm_start = time.time()
    chameleon(df_2nd)
    chm_end = time.time()
    print("Computational Time of Chameleon Clustering Technique with 51D(DS2) Dataset: {}".format(chm_end-chm_start))
    print('\n')
    