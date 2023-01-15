import numpy as np
from functools import reduce

'''
https://homel.vsb.cz/~kro080/PAI-2022/U3/ukol3.html

https://youtrack.jetbrains.com/issue/PY-52273/Debugger-multiprocessing-hangs-pycharm-2021.3
https://youtrack.jetbrains.com/issue/PY-37366/Debugging-with-multiprocessing-managers-no-longer-working-after-update-to-2019.2-was-fine-in-2019.1#focus=Comments-27-4690501.0-0
'''
import itertools as it
import multiprocessing as mp
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

    # run_vis = True
    run_vis = False
    debug_loading = True
    debug_loading_divider = 10000
    debug_eval = True
    show_n_best = 3
    load_threads_cnt = 4

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

    threshold = 1

    prs = dict()

    distances = (-1,1)

    start_node = 0

    nxgraphType = "complete_graph"

    NP = 10
    DC = 20  # In TSP, it will be a number of cities

    figsize = (10, 6)

    def split_file(s, file_path = None, cnt = 1):
        if file_path is None:
            return
        f_in = open(file_path)
        count_all_lines = 0
        while True:
            count_all_lines += 1
            line = f_in.readline()
            if not line:
                break
        f_in.close()

        f_in = open(file_path)
        part_i = 0
        part_cnt = 0
        f_out = open(file_path + "." + str(part_i), 'ab+')
        one_part_all_cnt = count_all_lines / cnt

        while True:

            if part_cnt >= one_part_all_cnt:
                part_cnt = 0
                f_out.close()
                part_i = part_i + 1
                f_out = open(file_path + "." + str(part_i))

            line = f_in.readline()
            if not line:
                break
            f_out.write(line)
        f_in.close()
        f_out.close()
        pass


    def load_graph(s, whole_file_path):

        s.split_file(whole_file_path, s.load_threads_cnt)

        for part_i in range(0, s.load_threads_cnt):
            file_path = whole_file_path + "." + str(part_i)

            # pool_size = 8
            # # budeme pracovat ve sdilene pameti
            # with mp.Manager() as manager:
            #     lock = manager.Lock()
            #     # max val and empty permutation
            #     best_found = manager.Value('d', [max_val, []])
            #     splitter = list(range(0, len(c) - 1))
            #     # spustime ve vice procesech
            #     with mp.Pool(processes=pool_size) as pool:
            #         # kazde vlakno v poolu se stara o cast vypoctu, rozdeleno dle seznamu 'splitter'
            #         ret = pool.starmap(paral_srflp_permutation,
            #                            zip(it.repeat(l), it.repeat(c), splitter, it.repeat(best_found), it.repeat(lock)))
            #     # vypsani nejlepsich reseni
            #     print("Best Value:", best_found.value[0], "postfix permutation: ", best_found.value[1])
            #
            #     # ulozeni vysledku do souboru
            #     output_file = open('1.py.res.txt', 'w')
            #     output_file.write('Val:\n' + str(best_found.value[0]))
            #     output_file.write('Perm:\n' +

            f = open(file_path)
            count = 0
            s.G = nx.DiGraph()
            while True:
                count += 1

                # Get next line from file
                line = f.readline()
                if not line:
                    break
                if not line.startswith('#') and len(line) > 0 and line[0].isdigit():
                    edge = line.split("	")
                    s.G.add_edge(int(edge[0]), int(edge[1]))
                if Vis2D.debug_loading:
                    if count % Vis2D.debug_loading_divider == 0:
                        print("Line {}: {}".format(count, line.strip()))

            # print("Line{}: {}".format(count, line.strip()))
            f.close()
            return s.G

    def __init__(s, nxgraphType=None, nxgraphOptions=None, graphData=None, graph=None, file_path=None):
        if nxgraphType is not None:
            s.nxgraphType = nxgraphType
        s.nxgraphOptions = nxgraphOptions
        s.graphData = graphData
        s.graph = graph
        if file_path is not None:
            s.graph = s.load_graph(file_path)
        if s.graph is None:
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
        else:
            s.G = s.graph
            s.number_of_nodes = s.G.number_of_nodes()
            init_pr = 1.0 / s.number_of_nodes
            idx = 0
            for n in s.G.nodes(data=True):
                # n[1]['pos'] = idx
                n[1]['cur_pr'] = init_pr
                n[1]['old_threshold'] = 0
                idx = idx + 1
        s.labels = None
        s.labels_old = None
        # generates random weights to graph
        # for (u, v) in s.G.nodes(data=True):
        #     v['pos'] = (random.randint(s.distances[0], s.distances[1]), random.randint(s.distances[0], s.distances[1]))
        # for (u, v, w) in s.G.edges(data=True):
        #     # w['weight'] = (random.randint(1, 40))
        #     u1 = s.G.nodes()[u]['pos']
        #     v1 = s.G.nodes()[v]['pos']
        #     real_dist = np.sqrt(np.power(u1[0]-v1[0], 2) + np.power(u1[1]-v1[1], 2))
        #     w['weight'] = int(real_dist)

        # or load graph with weights them directly...
        # s.G = nx.from_edgelist(list(
        #     [(0, 1, {'weight': 15}), (0, 3, {'weight': 34}), (0, 4, {'weight': 25}), (1, 2, {'weight': 5}),
        #      (1, 7, {'weight': 23}), (2, 3, {'weight': 33}), (2, 6, {'weight': 29}), (3, 5, {'weight': 13}),
        #      (4, 5, {'weight': 5}), (4, 7, {'weight': 20}), (5, 6, {'weight': 38}), (6, 7, {'weight': 3})]
        #     ))

        if Vis2D.run_vis:
            s.plt = plt
            # s.fig, s.ax = plt.fig("BIA - #3 - Genetic alg. on Traveling Salesman Problem (TSP) ", figsize=s.figsize)
            s.fig, s.ax = plt.subplots()
            # s.fig, s.ax = plt.subplots()
            s.idx_weights = range(2, 30, 1)

            s.layout = nx.circular_layout(s.G)
            # s.layout = nx.random_layout(s.G)
            nx.set_node_attributes(s.G, s.layout, 'pos')

            s.ax.set_xlim(s.distances[0], s.distances[1])
            s.ax.set_ylim(s.distances[0], s.distances[1])
            s.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

            s.update()
            s.plt.pause(1)

    d = 0.85

    def price(s, u):
        first = (1.0 - s.d) / s.number_of_nodes
        second_sum = 0
        out_nodes = s.G.out_edges(nbunch=u[0])
        for v in out_nodes:
            v_node = list(filter(lambda x: x[0] == v[1], list(s.G.nodes(data=True))))[0]
            # second_sum_div_top = s.price(v_node)
            second_sum_div_top = v_node[1]['cur_pr']
            second_sum_div_bottom = len(s.G.in_edges(nbunch=v_node[0]))  # in nodes
            second_sum_div = second_sum_div_top / second_sum_div_bottom
            second_sum = second_sum + second_sum_div
        total = first + s.d * second_sum
        return total

    def sum_prices(s, which='cur_pr'):
        return sum(x[1][which] for x in s.G.nodes(data=True))

    def check_threshold_condition(s,n, min_threshold = 0):
        threshold_cond = (n["cur_threshold"] <= min_threshold or n["cur_threshold"] > 0)
        threshold_cond_2 = n["old_threshold"] != n["cur_threshold"]
        return threshold_cond and threshold_cond_2


    def alg(s):
        """
            Genetic alg. for solving TSP
        """

        idx = 0

        min_threshold = 0
        iters = 0
        while True:
            threshold_reached = True

            for i in s.G.nodes(data=True):
                pprint(i)
                new_price = s.price(i)
                i[1]["old_pr"] = i[1]["cur_pr"]
                i[1]["cur_pr"] = new_price
                i[1]["cur_threshold"] = i[1]["old_pr"] - i[1]["cur_pr"]
                if s.check_threshold_condition(i[1], min_threshold):
                    threshold_reached = False
                idx = idx + 1
                s.update()
            if threshold_reached:
                pprint(s.G.nodes(data=True))
                pprint("threshold_reached at " + str(iters) + "")
                s.update()
                break
            iters = iters + 1

        sorted_nodes = sorted(list(s.G.nodes(data=True)), key=lambda x: x[1]['cur_pr'])

        print("Best nodes:")
        for i in range(0, Vis2D.show_n_best):
            print("Line {}: {}".format(i, str(sorted_nodes[i])))

        if Vis2D.run_vis:
            s.plt.pause(15)
        pass

    def update(s, edges=None):
        """
        clear and update default network
        """
        if not Vis2D.run_vis:
            return

        if edges is None:
            edges = list(list())
        s.ax.clear()
        # s.show_axes()

        # Background nodes
        pprint(s.G.edges())
        nx.draw_networkx_edges(s.G, pos=s.layout, edge_color="gray", arrowstyle='->', arrows=True, arrowsize=10)
        # forestNodes = list([item for sublist in (([l[0], l[1]]) for l in edges) for item in sublist])

        # dbg("forestNodes", forestNodes)
        # forestNodes = list(filter(None, forestNodes))
        # dbg("forestNodes -!None", forestNodes)
        # dbg(set(self.G.nodes()))
        # null_nodes = nx.draw_networkx_nodes(s.G, pos=s.layout, nodelist=set(s.G.nodes()) - set(forestNodes),
        #                                     node_color="white", ax=s.ax)
        # null_nodes = nx.draw_networkx_nodes(s.G, pos=s.layout, nodelist=set(s.G.nodes()), node_color="white", ax=s.ax)
        null_nodes = nx.draw_networkx_nodes(s.G, pos=s.layout, nodelist=set(s.G.nodes()), node_color="white")

        # start node highlight

        if (null_nodes is not None):
            null_nodes.set_edgecolor("black")
            s.labels_old = s.labels
            # s.labels = dict(zip(set(s.G.nodes()), set(list(
            s.labels3 = dict()
            for n in s.G.nodes(data=True):
                n[1]['label'] = str(n[0]) + "\n" + str(round(n[1]['cur_pr'], 4))
                s.labels3[n[0]] = n[1]['label']
            #     map(lambda x: str(x[0]) + "\n" + str(round(x[1]['cur_pr'], 4)), s.G.nodes(data=True))))))
            s.labels = dict(zip(set(s.G.nodes()), set(list(
                # map(lambda x: str(round(x[1]['cur_pr'], 4)), s.G.nodes(data=True))))))
            map(lambda x: x[1]['label'], s.G.nodes(data=True))))))
            # s.labels = dict(zip(set(s.G.nodes()), set(list(
            #     map(lambda x: str(x[0]) + "\n" + str(round(x[1]['cur_pr'], 4)), s.G.nodes(data=True))))))
            # nx.draw_networkx_labels(s.G, pos=s.layout, labels=s.labels, font_size=6, font_color="red", ax=s.ax)
            s.labels2 = {key: val for key, val in sorted(s.labels.items(), key=lambda ele: ele[0], reverse=False)}
        nx.draw_networkx_labels(s.G, pos=nx.circular_layout(s.G), labels=s.labels3, font_size=6, font_color="red", ax=s.ax)
        # Query nodes
        # s.idx_colors = sns.cubehelix_palette(len(forestNodes), start=.5, rot=-.75)[::-1]
        color_map = []

        # query_nodes = nx.draw_networkx_nodes(s.G, pos=s.layout, nodelist=forestNodes,
        #                                      node_color=s.idx_colors[:len(forestNodes)], ax=s.ax)

        # if query_nodes is not None:
        #     query_nodes.set_edgecolor("white")
        # nx.draw_networkx_labels(s.G, pos=s.layout, labels=dict(zip(forestNodes[0], forestNodes[0])), font_color="red", ax=s.ax)
        # nx.draw_networkx_labels(s.G, pos=s.layout, labels=dict(zip(forestNodes, forestNodes)), font_color="white", ax=s.ax)

        edges = list((l[0], l[1]) for l in edges)
        dbg("edges", edges)
        # nx.draw_networkx_edges(s.G, pos=s.layout, edgelist=edges, width=s.idx_weights[:len(edges)] ) # , ax=s.ax, arrowstyle='->', arrowsize=10

        # draw weights
        labels = nx.get_edge_attributes(s.G, 'weight')
        # nx.draw_networkx_edge_labels(s.G, s.layout, edge_labels=labels)

        # Scale plot ax
        # s.ax.set_xticks([])
        # s.ax.set_yticks([])

        s.ax.set_title("Step #{}, Price: {}".format(Vis2D.frameNo, s.sum_prices()))
        # s.show_axes()

        s.plt.pause(s.frameTimeout)
        Vis2D.frameNo += 1

    def show_axes(s):
        s.ax.set_xlim(s.distances[0]-(s.distances[1]*0.1), s.distances[1]+(s.distances[1]*0.1))
        s.ax.set_ylim(s.distances[0]-(s.distances[1]*0.1), s.distances[1]+(s.distances[1]*0.1))
        s.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

class PageRank(Vis2D):
    pass

# sample data
# file_path = "web-BerkStan.cube.txt"
# file_path = "web-BerkStan.one-way-cube.txt"
# file_path = "web-BerkStan.pentagram-full.txt"
# file_path = "web-BerkStan.pentagram-one-way.txt"
# file_path = "web-BerkStan.pentagram-one-way-reverse.txt"
# file_path = "web-BerkStan.pentagram-one-way-with-one-sink.txt"
# file_path = "web-BerkStan.pentagram-one-way-with-one-sink.txt"
# file_path = "web-BerkStan.txt"
file_path = "web-BerkStan.head_200.txt"
# Vis2D.run_vis = True
Vis2D.run_vis = False

# graph_data_str = "[("+"1   2\n2   3\n1   4".replace("   ", ",").replace("\n","),(")+")]"
# graph_data_str = "[("+text3.replace("	", ",").replace("\n","),(")+")]"

# r = TSP(graphData=[(1, 2), (1, 3)])
r = PageRank(file_path=file_path)
# r = TSP()
r.alg()
