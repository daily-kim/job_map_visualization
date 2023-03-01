import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse

from similarity import save_weights
from visulaization import graph_construction, graph_ploting


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--jobname', type=str, default='통계학연구원')
    parser.add_argument('--grade', type=int, default='0')
    parser.add_argument('--s2s_threshold', type=float, default=0.9)
    parser.add_argument('--similarity_method', type=str, default='fasttext')
    parser.add_argument('--saveimgdir', type=str, default='result/img/')
    args = parser.parse_args()

    fn_img = f"{args.saveimgdir}{args.jobname}_grade{args.grade}_s2s{args.s2s_threshold}"

    graph = graph_construction(
        jobname=args.jobname,
        grade=args.grade,
        similarity_method=args.similarity_method,
        s2s_threshold=args.s2s_threshold
    )
    graph_ploting(graph, node_scale=(0.2, 1), edge_scale=None, filename=fn_img)
    save_weights(args.jobname, args.similarity_method, grade=args.grade)
