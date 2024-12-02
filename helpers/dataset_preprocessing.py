import numpy as np
import pandas as pd
import json
import networkx as nx
import igraph as ig
import os
import argparse

parser = argparse.ArgumentParser(description="Dataset Preprocessing Script")

#Note that the dataset must be in csv format and in the data directory
parser.add_argument(
    "--name", type=str, help="Name of the Dataset"
)

def preprocess_dataset(dataset_name):
    #Reading and Cleaning Dataset------------------------------------------------------------------------
    print('Reading & Cleaning Dataset')
    if not os.path.exists('./data'):
        print("Directory does not exist, creating it.")
        os.makedirs('./data')

    path = './data/{}.csv'.format(dataset_name)
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df.drop_duplicates(keep ='last', inplace= True, ignore_index= True)

    #Dataset Properties----------------------------------------------------------------------------------
    print("Dataset Properties Calculation")
    total_count = len(df)
    properties = {
        "name": dataset_name,
        "length": total_count,
    }
    num_benign = len(df[df['Label'] == 0])
    num_attack = len(df[df['Label'] == 1])

    properties["num_benign"] = num_benign
    properties["percentage_of_benign_records"] = ((num_benign * 100)/total_count)

    properties["num_attack"] = num_attack
    properties["percentage_of_attack_records"] = ((num_attack * 100)/total_count)

    properties["attacks"] = list(df["Attack"].unique())  
    properties["attack_benign_ratio"] = num_attack/num_benign

    filename = ('./data/{}_properties.json'.format(dataset_name))
    outfile = open(filename, 'w')
    outfile.writelines(json.dumps(properties))
    outfile.close()

    #Graph properties-----------------------------------------------------------------------------------
    print("Graph Properties Calculation")
    graph_properties = {}
    communities = []

    G = nx.from_pandas_edgelist(df, 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', create_using=nx.DiGraph())
    graph_properties["number_of_nodes"] = G.number_of_nodes()
    graph_properties["number_of_edges"] = G.number_of_edges()
    degrees = [degree for _, degree in G.degree()]
    graph_properties["max_degree"] = max(degrees)
    graph_properties["avg_degree"] = sum(degrees) / len(degrees)
    graph_properties["transitivity"] = nx.transitivity(G)
    graph_properties["density"] =  nx.density(G)

    G1 = ig.Graph.from_networkx(G)
    part = G1.community_infomap()

    for com in part:
        communities.append([G1.vs[node_index]['_nx_name'] for node_index in com])

    node_to_community = {}
    for community_index, community in enumerate(communities):
        for node in community:
            node_to_community[node] = community_index

    inter_cluster_edges = 0
    for u, v in G.edges():
        if node_to_community[u] != node_to_community[v]:
            inter_cluster_edges += 1


    graph_properties["mixing_parameter"] = inter_cluster_edges / G.number_of_edges()
    graph_properties["modularity"] = nx.community.modularity(G, communities)

    #High values of density and transitivity are above 0.01 and for mixing parameter it is above 0.1
    if ((graph_properties["transitivity"] <0.01 or graph_properties["density"]<0.01) and (graph_properties["mixing_parameter"]>0.1) ):
        message='Low Density/Transitivity & High Mixing'

    if ((graph_properties["transitivity"] <0.01 or graph_properties["density"]<0.01) and (graph_properties["mixing_parameter"]<0.1) ):
        message='Low Density/Transitivity & Low Mixing'

    if ((graph_properties["transitivity"] >0.01 or graph_properties["density"]>0.01) and (graph_properties["mixing_parameter"]>0.1) ):
        message='High Density/Transitivity & High Mixing'

    if ((graph_properties["transitivity"] >0.01 or graph_properties["density"]>0.01) and (graph_properties["mixing_parameter"]<0.1) ):
        message='High Density/Transitivity & Low Mixing'

    print("Identification of Properties: ", message)
    graph_properties["description"] = message

    filename = ('./data/{}_graph_properties.json'.format(dataset_name))
    outfile = open(filename, 'w')
    outfile.writelines(json.dumps(graph_properties))
    outfile.close()

    #Calculating Features------------------------------------------------------------------------
    print("Appending Centrality Measures as Features")

    graph_path = "./data/{}_preprocessed_graph.gexf".format(dataset_name)

    def _rescale(betweenness, n, normalized, directed=False, k=None, endpoints=False):
        if normalized:
            if endpoints:
                if n < 2:
                    scale = None  # no normalization
                else:
                    # Scale factor should include endpoint nodes
                    scale = 1 / (n * (n - 1))
            elif n <= 2:
                scale = None  # no normalization b=0 for all nodes
            else:
                scale = 1 / ((n - 1) * (n - 2))
        else:  # rescale by 2 for undirected graphs
            if not directed:
                scale = 0.5
            else:
                scale = None
        if scale is not None:
            if k is not None:
                scale = scale * n / k
            for v in betweenness:
                betweenness[v] *= scale
        return betweenness

    def cal_betweenness_centrality(G):
        G1 = ig.Graph.from_networkx(G)  # type: ignore
        estimate = G1.betweenness(directed=True)
        # for v in G1.vs:
        #     print(v)
        b = dict(zip(G1.vs["_nx_name"], estimate))
        return _rescale(b, G1.vcount(), True)

    degrees = nx.degree_centrality(G)
    betwe = cal_betweenness_centrality(G)
    pagerank = nx.pagerank(G, alpha=0.85)

    df["src_degree"] = df.apply(
                lambda row: degrees.get(row['IPV4_SRC_ADDR'], -1), axis=1)
    df["dst_degree"] = df.apply(
                lambda row: degrees.get(row['IPV4_DST_ADDR'], -1), axis=1)

    df["src_betweenness"] = df.apply(
                lambda row: betwe.get(row['IPV4_SRC_ADDR'], -1), axis=1)
    df["dst_betweenness"] = df.apply(
                lambda row: betwe.get(row['IPV4_DST_ADDR'], -1), axis=1)

    df["src_pagerank"] = df.apply(
                lambda row: pagerank.get(row['IPV4_SRC_ADDR'], -1), axis=1)
    df["dst_pagerank"] = df.apply(
                lambda row: pagerank.get(row['IPV4_DST_ADDR'], -1), axis=1)

    print(df.shape)
    print(df.head())
    new_path = "./data/{}_preprocessed.pkl".format(dataset_name)
    pd.to_pickle(df, new_path)

if __name__ == "__main__":
    args = parser.parse_args()
    preprocess_dataset(args.name)