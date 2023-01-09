import numpy as np

'''
https://homel.vsb.cz/~kro080/PAI-2022/U3/ukol3.html

https://youtrack.jetbrains.com/issue/PY-52273/Debugger-multiprocessing-hangs-pycharm-2021.3
https://youtrack.jetbrains.com/issue/PY-37366/Debugging-with-multiprocessing-managers-no-longer-working-after-update-to-2019.2-was-fine-in-2019.1#focus=Comments-27-4690501.0-0
'''

import networkx as nx
from pprint import pprint
import random
import matplotlib.pyplot as plt
import numpy as np
# from base import *
import ast

try:
    import seaborn as sns
except UserWarning:
    pass
import matplotlib.animation
import random

matplotlib.use("TkAgg")

import networkx as nx
from pprint import pprint
from pprint import pformat

def V(G: nx.Graph):
    return G.nodes()

def E(G: nx.Graph):
    return G.edges()

def dbg(v, *args):
    if len(args) > 0:
        # if len(args) == 1:
        #     pprint(v + ": " + pformat(args[0]))
        #     pass
        # else:
        print(v + ": ")
        for i in args:
            pprint(i,  width=200)
    else:
        pprint(v)

class Vis2D(object):
    frameNo = 0
    min_individual_price = 0

    frameTimeout = 0.1
    # nxgraphType = "cubical_graph"
    nxgraphOptions = None
    graphData = None
    G = None
    D = None
    plt = None
    layout = None

    g1 = list()  # generation
    g2 = list()  # generation
    # ng = list()  # next generation


    pop_cnt = 10
    cg = list(list())
    ng = list(list())

    distances = (0,500)

    start_node = 0

    # NP = 20
    # G = 200
    # D = 20  # In TSP, it will be a number of cities
    nxgraphType = "complete_graph"

    NP = 20  # population cnt
    GC = 200  # generation cnt
    DC = 20  # In TSP, it will be a number of cities
    figsize = (10, 6)
    # NP = 20  # population cnt
    # GC = 40  # generation cnt
    # DC = 8  # In TSP, it will be a number of cities
    # figsize = (6, 4)

    population = None

    def __init__(s, nxgraphType=None, nxgraphOptions=None, graphData=None):
        if nxgraphType is not None:
            s.nxgraphType = nxgraphType
        s.nxgraphOptions = nxgraphOptions
        s.graphData = graphData
        if graphData is None:
            if nxgraphOptions is None:
                if s.nxgraphType == "complete_graph":
                    s.G = nx.complete_graph(s.DC)
                else:
                    s.G = getattr(nx, s.nxgraphType)()
            else:
                s.G = getattr(nx, s.nxgraphType)(s.nxgraphOptions)
        else:
            s.G = nx.from_edgelist(ast.literal_eval(s.graphData))
        # generates random weights to graph
        for (u, v) in s.G.nodes(data=True):
            v['pos'] = (random.randint(s.distances[0], s.distances[1]), random.randint(s.distances[0], s.distances[1]))
        for (u, v, w) in s.G.edges(data=True):
            # w['weight'] = (random.randint(1, 40))
            u1 = s.G.nodes()[u]['pos']
            v1 = s.G.nodes()[v]['pos']
            real_dist = np.sqrt(np.power(u1[0]-v1[0], 2) + np.power(u1[1]-v1[1], 2))
            w['weight'] = int(real_dist)

        # or load graph with weights them directly...
        # s.G = nx.from_edgelist(list(
        #     [(0, 1, {'weight': 15}), (0, 3, {'weight': 34}), (0, 4, {'weight': 25}), (1, 2, {'weight': 5}),
        #      (1, 7, {'weight': 23}), (2, 3, {'weight': 33}), (2, 6, {'weight': 29}), (3, 5, {'weight': 13}),
        #      (4, 5, {'weight': 5}), (4, 7, {'weight': 20}), (5, 6, {'weight': 38}), (6, 7, {'weight': 3})]
        #     ))


        s.plt = plt
        # s.fig, s.ax = plt.fig("BIA - #3 - Genetic alg. on Traveling Salesman Problem (TSP) ", figsize=s.figsize)
        s.fig, s.ax = plt.subplots()
        # s.fig, s.ax = plt.subplots()
        s.idx_weights = range(2, 30, 1)

        # s.layout = nx.circular_layout(s.G)
        # list(map(lambda x: x[1]['pos'],s.G.nodes(data=True)))
        pos = {point: point for point in list(map(lambda x: x[1]['pos'], s.G.nodes(data=True)))}
        s.layout = list(pos)
        # s.fig = s.plt.figure("BIA - #3 - Genetic alg. on Traveling Salesman Problem (TSP) ", figsize=s.figsize)
        # s.fig.set_title("BIA - #3 - Genetic alg. on Traveling Salesman Problem (TSP) ")
        # s.ax = s.plt.axes()
        # s.plt.axis("on")
        s.ax.set_xlim(s.distances[0], s.distances[1])
        s.ax.set_ylim(s.distances[0], s.distances[1])
        s.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        s.update()
        # s.plt.pause(3)


    def f(s, i):
        """
            price function, idk why it is called 'f()' in this case,
            but for sake of balancing readeability and documentation
            I use internal function 'price()' (note: should I rename to 'cost()'?)
        """
        return s.price(i)

    def crossover(s,i1, i2):
        i3 = i1.copy()
        for i in range(0, len(i1)-1):
            if bool(random.getrandbits(1)):
                # due to the way the data structures are used I cannot simply merge some nodes together
                # because it would lead tu duplicate elements, so I deal with it by
                # swapping elements according to parent nodes on given position and so the
                # offspring has nodes ordered according to both parents
                # this is not a mutation because I need information from both parents to do this
                offspring_i = i3.index(i2[i])
                i3[i], i3[offspring_i] = i3[offspring_i], i3[i]
        return i3

    def show_min_path(s, color='blue'):
        # eval final population
        s.min_individual = s.population[0]
        Vis2D.min_individual_price = s.f(s.min_individual)
        for i in range(1, len(s.population)):
            price = s.f(s.population[i])
            if (Vis2D.min_individual_price > price):
                s.min_individual = s.population[i]
                Vis2D.min_individual_price = price
        s.update()
        s.show_path(s.min_individual, color=color, w=1)
        s.plt.pause(s.frameTimeout)

    def show_path(s, ga, ars = '->', color = 'k', w = None, draw = True):
        ea = s.get_edges(ga)
        if draw:
            for e in ea:
                if w == None:
                    ew = w
                else:
                    # ew = s.idx_weights[:len(ga)]
                    ew = 1
                nx.draw_networkx_edges(s.G, pos=s.layout, edgelist=e, width=ew, arrowstyle=ars, arrows=True, edge_color=color)
                s.show_axes()
        return ga

    def price(s, g):
        sum = 0
        for i in range(0, len(g) - 2):
            sum = sum + s.G[g[i]][g[i + 1]]['weight']
        return sum

    def alg(s):
        """
            Genetic alg. for solving TSP
        """
        # population = Generate NP random individuals Evaluate individuals within population
        s.population = list()
        for i in range(0, s.NP):
            s.population.append(s.random_circle(list(s.G.nodes())))

        for i in range(0, s.GC):
            s.new_population = s.population.copy()  # Offspring is always put to a new population

            for j in range(0, s.NP-1):
                parent_A = s.population[j]
                # parent_B = random individual from population (parent_B != parent_A)
                parent_B_i = None
                while parent_B_i is None or parent_B_i == j:
                    parent_B_i = random.randint(0, s.NP - 1)
                parent_B = s.population[parent_B_i]

                offspring_AB = s.crossover(parent_A, parent_B)
                if random.uniform(0.0, 1.0) < 0.5:
                    offspring_AB2 = s.mutate(offspring_AB)
                    offspring_AB = offspring_AB2
                s.g1 = parent_A
                s.g2 = offspring_AB
                # s.evaluate(offspring_AB)

                if s.f(offspring_AB) < s.f(parent_A):
                    s.new_population[j] = offspring_AB
            s.population = s.new_population
            s.show_min_path(color='green')

        s.show_min_path(color='red')
        s.plt.pause(10)
        pass

    def get_edges(s, nodes):
        edges = list()

        def get_edge(i1, i2):
            return list(filter(lambda i: i[1] == i2, s.G.edges(i1)))

        for i in range(0, len(nodes) - 1):
            edges.append(get_edge(nodes[i], nodes[i + 1]))
        edges.append(get_edge(nodes[len(nodes) - 1], nodes[0]))
        return edges

    def random_circle(s, nodes):
        x = nodes.copy()
        random.shuffle(x)
        return x

    def mutate(s, nodes_, cnt = 1):
        # swaps 2 random elements in array 'cnt' times
        # nodes = nodes_.copy()
        # swapped = random.shuffle(nodes)
        swapped = nodes_.copy()
        for i in range(0, cnt):
            pos1 = random.randint(0, len(nodes_)-1)
            pos2 = random.randint(0, len(nodes_)-1)
            swapped[pos1], swapped[pos2] = swapped[pos2], swapped[pos1]
        return swapped

    def update(s, edges=None):
        """
        clear and update default network
        """
        if edges is None:
            edges = list(list())
        s.ax.clear()
        s.show_axes()

        # Background nodes
        pprint(s.G.edges())
        # nx.draw_networkx_edges(s.G, pos=s.layout, edge_color="gray", arrowstyle='-|>', arrowsize=10)
        forestNodes = list([item for sublist in (([l[0], l[1]]) for l in edges) for item in sublist])

        # dbg("forestNodes", forestNodes)
        forestNodes = list(filter(None, forestNodes))
        # dbg("forestNodes -!None", forestNodes)
        # dbg(set(self.G.nodes()))
        # null_nodes = nx.draw_networkx_nodes(s.G, pos=s.layout, nodelist=set(s.G.nodes()) - set(forestNodes),
        #                                     node_color="white", ax=s.ax)
        null_nodes = nx.draw_networkx_nodes(s.G, pos=s.layout, nodelist=set(s.G.nodes()), node_color="white", ax=s.ax)

        # start node highlight
        nx.draw_networkx_nodes(s.G, pos=s.layout, nodelist=set(filter(lambda i: i == s.start_node, s.G.nodes())), node_color="green", ax=s.ax)

        if (null_nodes is not None):
            null_nodes.set_edgecolor("black")
            nullNodesIds = set(s.G.nodes()) - set(forestNodes)
            # dbg("nullNodes", nullNodes)
            nx.draw_networkx_labels(s.G, pos=s.layout, labels=dict(zip(nullNodesIds, nullNodesIds)),
                                    font_color="black",
                                    ax=s.ax)

        # Query nodes
        s.idx_colors = sns.cubehelix_palette(len(forestNodes), start=.5, rot=-.75)[::-1]
        color_map = []

        query_nodes = nx.draw_networkx_nodes(s.G, pos=s.layout, nodelist=forestNodes,
                                             node_color=s.idx_colors[:len(forestNodes)], ax=s.ax)

        if query_nodes is not None:
            query_nodes.set_edgecolor("white")
        # nx.draw_networkx_labels(s.G, pos=s.layout, labels=dict(zip(forestNodes[0], forestNodes[0])), font_color="red", ax=s.ax)
        nx.draw_networkx_labels(s.G, pos=s.layout, labels=dict(zip(forestNodes, forestNodes)), font_color="white", ax=s.ax)

        edges = list((l[0], l[1]) for l in edges)
        dbg("edges", edges)
        # nx.draw_networkx_edges(s.G, pos=s.layout, edgelist=edges, width=s.idx_weights[:len(edges)] ) # , ax=s.ax, arrowstyle='->', arrowsize=10

        # draw weights
        labels = nx.get_edge_attributes(s.G, 'weight')
        # nx.draw_networkx_edge_labels(s.G, s.layout, edge_labels=labels)

        # Scale plot ax
        # s.ax.set_xticks([])
        # s.ax.set_yticks([])

        # s.ax.set_title("Step #{}, Price: {}".format(Vis2D.frameNo, Vis2D.min_individual_price))
        s.ax.set_title("Step #{}, NP: {}, GC {}, DC: {}, Price: {}".format(Vis2D.frameNo,Vis2D.NP,Vis2D.GC,Vis2D.DC, Vis2D.min_individual_price))
        s.show_axes()

        # self.plt.pause(5)
        # s.plt.pause(s.frameTimeout)
        # self.plt.pause(3)
        Vis2D.frameNo += 1
    def show_axes(s):
        s.ax.set_xlim(s.distances[0]-(s.distances[1]*0.1), s.distances[1]+(s.distances[1]*0.1))
        s.ax.set_ylim(s.distances[0]-(s.distances[1]*0.1), s.distances[1]+(s.distances[1]*0.1))
        s.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

class TSP(Vis2D):
    pass

r = TSP()
r.alg()
