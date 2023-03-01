import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


from similarity import similarity_grade
from utils import minmax_scaler


def subject_labeling(jobname, similarity_method, s2s_threshold):
    subject_grade = {}
    for grade in [1, 2, 3]:
        _, _, job_subject_sim, _ = similarity_grade(
            jobname, similarity_method, s2s_threshold, grade=grade)
        subject_grade.update(
            {subject: grade for subject in job_subject_sim.keys()})
    return subject_grade


def graph_construction(jobname, grade=0, similarity_method="tasttext", s2s_threshold=0.9):
    _, _, job_subject_sim, subject_subject_sim = similarity_grade(
        jobname, similarity_method, s2s_threshold, grade)

    if grade == 0:
        subject_grade = subject_labeling(
            jobname, similarity_method, s2s_threshold)
    else:
        subject_grade = {
            subject: grade for subject in job_subject_sim.keys()}

    subject_subject = list(subject_subject_sim.keys())
    job_subject = []

    subject_subject = [
        (
            list(subject_subject_sim.keys())[idx][0],
            list(subject_subject_sim.keys())[idx][1],
            list(subject_subject_sim.values())[idx],
        )
        for idx in range(len(subject_subject_sim))
    ]

    for _ in range(len(job_subject_sim)):
        job_subject.extend(
            (jobname, subject, job_subject_sim[subject])
            for subject in job_subject_sim
        )

    g = nx.Graph()
    g.add_node(jobname, kind='job', weight=2, color='silver')
    g.add_nodes_from([(node, {'weight': attr, 'kind': 'subject'})
                     for (node, attr) in job_subject_sim.items()])
    # set node color by grade
    color_dict = {1: 'green', 2: 'lightgreen', 3: 'yellowgreen'}
    for node in g.nodes():
        if node in subject_grade.keys():
            g.nodes[node]['color'] = color_dict[subject_grade[node]]

    g.add_weighted_edges_from(
        subject_subject, kind='subject-subject', color='lightblue')
    g.add_weighted_edges_from(job_subject, kind='job-subject', color='silver')

    return g


def graph_ploting(g, node_scale=(0, 1), edge_scale=(0, 1), filename="graph"):
    node_colors = nx.get_node_attributes(g, 'color').values()
    node_weight = nx.get_node_attributes(g, 'weight').values()
    edge_colors = nx.get_edge_attributes(g, 'color').values()
    edge_weight = nx.get_edge_attributes(g, 'weight').values()
    num_node = g.number_of_nodes()

    pos = nx.nx_agraph.graphviz_layout(g, prog="neato")
    plt.figure(figsize=(max(15, g.number_of_nodes()),
               max(15, g.number_of_nodes())))

    if node_scale != None:
        node_weight = minmax_scaler(node_weight, scale=node_scale)

    if edge_scale != None:
        edge_weight = minmax_scaler(edge_weight, scale=edge_scale)
    nx.draw(
        g,
        pos=pos,
        with_labels=True,
        font_family='Applegothic',
        edge_color=edge_colors,
        node_color=node_colors,
        node_size=[x * 10000 for x in node_weight],
        font_size=np.mean(list(node_weight))*20,
        width=[x * 5 for x in edge_weight],
    )
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
