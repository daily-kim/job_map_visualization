import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import pickle

from similarity import save_weights
from visulaization import graph_construction, graph_ploting


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--jobname', type=str, default='통계학연구원', help='직업명')
    parser.add_argument('--grade', type=int, default=0,
                        help='진로맵 대상 학년, 0:전체, 1:1학년, 2:2학년, 3:3학년')
    parser.add_argument('--similarity_method', type=str,
                        default='fasttext', help='유사도 측정 방법, fasttext, sbert')
    parser.add_argument('--s2s_threshold', type=float,
                        default=0.9, help='과목과 과목사이 유사도 임계값')

    parser.add_argument('--node_decay', type=bool,
                        default=True, help='노드 감소 여부')
    parser.add_argument('--node_decay_method', type=str,
                        default="num_total", help='노드 감소 방법, weight, num_total, num_per_grade')
    parser.add_argument('--node_decay_weight', type=float,
                        default=1.3, help='노드 감소 기준, weight')
    parser.add_argument('--node_decay_num_total', type=int,
                        default=15, help='노드 감소 기준, num_total')
    parser.add_argument('--node_decay_num_per_grade', type=int,
                        default=5, help='노드 감소 기준, num_per_grade')

    parser.add_argument('--node_scale', type=tuple,
                        default=(0.5, 1), help='그래프 생성시 노드 크기 범위')
    parser.add_argument('--edge_scale', type=tuple,
                        default=(0.5, 1), help='그래프 생성시 엣지 크기 범위')

    parser.add_argument('--imgdir', type=str,
                        default='result/img/', help='그래프 이미지 저장 경로')
    parser.add_argument('--weightdir', type=str,
                        default='result/weight/', help='가중치 저장 경로')
    args = parser.parse_args()

    graph = graph_construction(
        jobname=args.jobname,
        grade=args.grade,
        similarity_method=args.similarity_method,
        s2s_threshold=args.s2s_threshold
    )
    graph_ploting(graph, args)
    save_weights(args)
