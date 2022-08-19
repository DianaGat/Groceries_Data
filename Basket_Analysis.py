# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 14:35:38 2022

@author: chibi
"""

### What people ususally buy together? aka Basket analsyis
### Association rule learning , Apriori
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
import seaborn as sns

#load data
df = pd.read_csv("C:/datas/Groceries_dataset.csv")
df.info() #38765 obs, 3 cols
df.head()
#NAs
df.isnull().sum()
#transformations
df.Member_number = df.Member_number.astype("string")
df.itemDescription = df.itemDescription.astype("string")
df.Date = pd.to_datetime(df.Date, dayfirst = True)
df_grouped_by_day_customer = df.groupby(["Date", "Member_number"], as_index = False).agg({"itemDescription": set})
df_grouped_by_day_customer = df_grouped_by_day_customer.iloc[:,-1]
df_grouped_by_day_customer = df_grouped_by_day_customer.values.tolist()
maximum = 0
while maximum != len(df_grouped_by_day_customer):
    df_grouped_by_day_customer[maximum] = list(df_grouped_by_day_customer[maximum])
    maximum = maximum + 1

import mlxtend
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(df_grouped_by_day_customer).transform(df_grouped_by_day_customer)
data = pd.DataFrame(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import apriori
frequent_items = apriori(data, min_support=0.001, use_colnames=True) #many items
frequent_items = frequent_items.sort_values(['support'], ascending=False)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
from mlxtend.frequent_patterns import association_rules
ar = association_rules(frequent_items, metric="support", min_threshold=0.001)
ar2 =association_rules(frequent_items, metric="lift", min_threshold=1.2)

graph = ar2[["antecedents", "consequents", "lift"]]
graph["antecedents"] = graph["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
graph["consequents"] = graph["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")

import networkx as nx
import matplotlib as mpl
seed = 13648 
G = nx.from_pandas_edgelist(graph, 'antecedents', 'consequents', "lift", create_using = nx.DiGraph) 
pos = nx.spring_layout(G, k = 0.20, seed = seed)
from matplotlib.pyplot import figure
from matplotlib.cm import ScalarMappable

plt.figure(figsize=(15, 15), dpi = 150, frameon = False)
M = G.number_of_edges()
edge_colors = range(2, M + 2)
cmap = plt.cm.inferno
sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,max(edge_colors)))
sm.set_array([])
cbar = plt.colorbar(sm, ticks = [], fraction = 0.01, pad = - 0.02)
cbar.set_label("", rotation=0)
nx.draw_networkx(G, pos, 
                 node_color="yellow",
                 edge_color=edge_colors, edge_cmap=cmap, width = 2,
                 with_labels=True, font_size = 6, alpha = 0.75, font_weight = "bold",
                 arrowsize = 4, linewidths = 0)
plt.savefig("C:/datas/Graph.pdf")
plt.show()

### By number of conncetions
degree_centrality = pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index', columns = ["degree_centrality"])
degree_centrality = degree_centrality.sort_values(['degree_centrality'], ascending=False)
#plot
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [10, 10]})
ax = sns.barplot(x = degree_centrality.index[:10], y = degree_centrality.degree_centrality[:10])
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.set_xticklabels(degree_centrality.index[:10], rotation = 15)
plt.title("Top 10 items by degree centrality")
plt.show()

### sum of the path lengths from the given node to all other nodes
### closest to every other node, cental
close_centrality = pd.DataFrame.from_dict(nx.closeness_centrality(G), orient='index', columns = ["closeness"])
close_centrality = close_centrality.sort_values(['closeness'], ascending=False)
#plot
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [10, 10]})
ax = sns.barplot(x = close_centrality.index[:10], y = close_centrality.closeness[:10])
ax.yaxis.set_major_locator(MultipleLocator(0.01))
ax.set_xticklabels(close_centrality.index[:10], rotation = 15)
plt.title("Top 10 items by closeness")
plt.show()

close_centrality.closeness.mean()


###important nodes connect other nodes
#bridge from one part of a graph to another
bet_centrality = pd.DataFrame.from_dict(nx.betweenness_centrality(G, endpoints = False),
                                        orient='index', columns = ["between"])
bet_centrality = bet_centrality.sort_values(['between'], ascending=False)
#plot
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [10, 10]})
ax = sns.barplot(x = bet_centrality.index[:10], y = bet_centrality.between[:10])
ax.yaxis.set_major_locator(MultipleLocator(0.01))
ax.set_xticklabels(bet_centrality.index[:10], rotation = 15)
plt.title("Top 10 items by betweeness centrality")
plt.show()

#Important nodes are those with many inlinks from important  nodes
pr = pd.DataFrame.from_dict(nx.pagerank(G, alpha = 0.8), orient='index', columns = ["pr"])
pr = pr.sort_values(['pr'], ascending=False)
#plot
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [10, 10]})
ax = sns.barplot(x = pr.index[:10], y = pr.pr[:10])
ax.yaxis.set_major_locator(MultipleLocator(0.005))
ax.set_xticklabels(pr.index[:10], rotation = 15)
plt.title("Top 10 items by betweeness Page Rank")
plt.show()

### lift sets
#datatransformation
graph2 = graph.sort_values(['lift'], ascending=False, ignore_index=True)
#plot
sns.set(style = "whitegrid", 
        font_scale = 1,
        rc={"figure.figsize": [10, 10]})
ax = sns.scatterplot(graph2.antecedents[:88], graph2.consequents, size = graph2.lift, hue = graph2.lift, palette = "dark:salmon_r")
plt.title("Lift for items and sets of items, lift > 1.2")
plt.legend(title = "lift", loc = (-0.12,-0.17))
ax.set_xticklabels(graph2.antecedents[:88], rotation = 90)
plt.show()

###lift vs support
#data transformation
ar =association_rules(frequent_items, metric="lift", min_threshold=1)
graph3 = ar[["antecedents", "consequents", "support", "lift"]]
graph3["antecedents"] = graph3["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
graph3["consequents"] = graph3["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
#plot
sns.set(style = "whitegrid", 
        font_scale = 1,
        palette= "rocket",
        rc={"figure.figsize": [10, 10]})
ax=sns.scatterplot(graph3.support, graph3.lift)
plt.title("Lift vs support")
ax.yaxis.set_major_locator(MultipleLocator(0.1))
plt.show()