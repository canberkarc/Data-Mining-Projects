import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import metis

def euclidean_dist(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
 
# Function to build knn graph
def build_knnG(df, k):
    vertices = []
    for v in df.itertuples():
        vertices.append(v[1:]) 
    g = nx.Graph()
    for i in range(0, len(vertices)):
        g.add_node(i)
    for i, p in enumerate(vertices):
        distances = find_distances(p,vertices)
        close_ones = np.argsort(distances)[1:k+1] 
        for c in close_ones:
            g.add_edge(i, c, weight=1.0 / distances[c], similarity=int(
                1.0 / distances[c] * 1e4))
        g.nodes[i]['pos'] = p
    g.graph['attr'] = 'sim'
    return g

# Function to find euclidean distances between points
def find_distances(p1,points):
    lst = []
    for p2 in points:
        lst.append(euclidean_dist(p1,p2))
    return lst

def partition_graph(part, dframe=None): 
    _, subs = metis.part_graph(part, 2, objtype = 'cut')
    for ind, n in enumerate(part.nodes()):
        part.nodes[n]['cluster'] = subs[ind]
    if dframe is not None:
        dframe['cluster'] = nx.get_node_attributes(part, 'cluster').values()
    return part

def sub_partition(p_graph, dframe=None): 
    check, num = 0, 30
    for n in p_graph.nodes():
        p_graph.nodes[n]['cluster'] = 0
    len_nodes = {}
    len_nodes[0] = len(p_graph.nodes())

    while check < num - 1:
        node_val = 0
        node_key = -1
        for key, val in len_nodes.items():
            if val > node_val:
                node_val = val
                node_key = key
        subnodes = [n for n in p_graph.nodes if p_graph.nodes[n]['cluster'] == node_key]
        subgraph = p_graph.subgraph(subnodes)
        _, subs = metis.part_graph(subgraph, 2, objtype='cut')
        partition_num = 0
        for i, p in enumerate(subgraph.nodes()):
            if subs[i] == 1:
                p_graph.nodes[p]['cluster'] = check + 1
                partition_num = partition_num + 1
        len_nodes[node_key] = len_nodes[node_key] - partition_num
        len_nodes[check + 1] = partition_num
        check = check + 1

    _, subs = metis.part_graph(p_graph, num)
    if dframe is not None:
        dframe['cluster'] = nx.get_node_attributes(p_graph, 'cluster').values()
    return p_graph

# Function to get edges
def getting_edges(partitions, graph):
    return [(a, b)  for a in partitions[0] for b in partitions[1] if b in graph[a] if a in graph]
   
# The internal inter-connectivity of a cluster can be found by
# using the size of its min-cut bisector
def internal_interconnectivity(graph, clust): 
    clust = graph.subgraph(clust)

    # Find min-cut bisector for average weights of edges
    clust = clust.copy()
    clust = partition_graph(clust)
    partitions = [n for n in clust.nodes if clust.nodes[n]['cluster'] in [0]], [n for n in clust.nodes if clust.nodes[n]['cluster'] in [1]]
    edges = getting_edges(partitions, clust) 

    # Find weights
    weights = []
    for e in edges:
        weights.append(clust[e[0]][e[1]]['weight'])
    return np.sum(weights)

# Function to find high val of RI and RC for merging 
def high_RiAndRc(g, ci, cj): 
    # Calculate relative interconnectivity 
    edges = getting_edges((ci, cj), g)
    # Find weights
    weights = []
    for e in edges:
        weights.append(g[e[0]][e[1]]['weight'])
    Ec = np.sum(weights)
    Ecci = internal_interconnectivity(g, ci)
    Eccj = internal_interconnectivity(g, cj)
    RI = Ec / ((Ecci + Eccj) / 2.0)

    # Calculate relative closeness
    edges = getting_edges((ci, cj), g)
    if not edges:
        return 0
    else:
        # SEC is the average weight of the edges that connect vertices in Ci
        # to vertices in Cj
        # Find weights
        weights = []
        for e in edges:
            weights.append(g[e[0]][e[1]]['weight'])
        SEC = np.mean(weights)

    # Find internal closeness of ci
    ci = g.subgraph(ci)
    edges = ci.edges()
    weights_ICci = [ci[e[0]][e[1]]['weight'] for e in edges]
    Ci = np.sum(weights_ICci)

    # Find internal closeness of ci
    cj = g.subgraph(cj)
    edges = cj.edges()
    weights_ICcj = [cj[e[0]][e[1]]['weight'] for e in edges]
    Cj = np.sum(weights_ICcj)

    # SECci and SECcj are the average weights of the edges that belong to the 
    # min-cut bisector of clusters Ci and Cj respectively
    ci = g.subgraph(ci)
    # Find min-cut bisector for average weights of edges
    ci = ci.copy()
    ci = partition_graph(ci)
    partitions = [n for n in ci.nodes if ci.nodes[n]['cluster'] in [0]], [n for n in ci.nodes if ci.nodes[n]['cluster'] in [1]]   
    edges = getting_edges(partitions, ci) 
    # Get weights
    weights_ci = [ci [e[0]][e[1]]['weight'] for e in edges]


    # SECci and SECcj are the average weights of the edges that belong to the 
    # min-cut bisector of clusters Ci and Cj respectively
    cj = g.subgraph(cj)
    # Find min-cut bisector for average weights of edges
    cj = cj.copy()
    cj = partition_graph(cj)
    partitions = [n for n in cj.nodes if cj.nodes[n]['cluster'] in [0]], [n for n in cj.nodes if cj.nodes[n]['cluster'] in [1]]    
    edges = getting_edges(partitions, cj) 
    # Get weights
    weights_cj = [cj [e[0]][e[1]]['weight'] for e in edges]
    
    SECci = np.mean(weights_ci)
    SECcj = np.mean(weights_cj)

    RC = SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))

    Ri_Rc_high = RI * RC * RC

    return Ri_Rc_high

def weights_finding(graph, edges):
    return [graph[edge[0]][edge[1]]['weight'] for edge in edges]

# Function to find combinations
def find_combinations(part, len_, NULL=object()):
    if len_ <= 0:
        combs = [NULL]
    else:
        combs = []
        for i, item in enumerate(part, 1):
            part_items = part[i:]
            part_combs = find_combinations(part_items, len_-1)
            combs.extend(item if comb is NULL else [item, comb]
                            for comb in part_combs)
    return combs

def merging(graph, df, k): 
    clusters = np.unique(df['cluster'])
    max_s = 0
    ci = -1
    cj = -1

    for combination in find_combinations(clusters, 2):
        i, j = combination
        if i != j:
            # Getting clusters
            g1 = [n for n in graph.nodes if graph.nodes[n]['cluster'] in [i]]
            g2 = [n for n in graph.nodes if graph.nodes[n]['cluster'] in [j]]
            graph_edges = getting_edges((g1,g2), graph)
            if not graph_edges:
                continue
            ms = high_RiAndRc(graph, g1, g2)
            if ms > max_s:
                max_s = ms
                ci = i
                cj = j

    if max_s > 0:
        df.loc[df['cluster'] == cj, 'cluster'] = ci
        for i, p in enumerate(graph.nodes()):
            if graph.nodes[p]['cluster'] == cj:
                graph.nodes[p]['cluster'] = ci
    return max_s > 0

def clustering(df, knn, num_of_clusters):
    graph = build_knnG(df, knn)
    graph = sub_partition(graph, df)
    for i in range(30 - num_of_clusters):
        merging(graph, df, num_of_clusters)

def plotting_before(df):
    df.plot.scatter(x = 0, y = 1, title = 'Before Clustering')
    plt.show()

def plotting_after(df):
    df.plot(kind='scatter', c=df['cluster'], cmap='gist_rainbow', x=0, y=1, title = 'After Clustering')
    plt.show()


df = pd.read_csv('/home/canberk/Desktop/data1.csv', sep=',', header=None)

plotting_before(df)

clustering(df, knn=10, num_of_clusters=7)

plotting_after(df)
