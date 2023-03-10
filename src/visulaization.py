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


def mask_nodes(g, node_decay_method, weight=10, num_total=1000, num_per_grade=100):

    node_weights = nx.get_node_attributes(g, 'weight')
    node_grade = nx.get_node_attributes(g, 'grade')

    if node_decay_method == 'weight':
        # Mask nodes with weights below threshold
        masked_nodes = [n for n in g.nodes() if node_weights[n]
                        < weight]
    elif node_decay_method == 'num_total':
        # Mask nodes except top n nodes (in terms of weight)
        sorted_nodes = sorted(
            g.nodes(), key=lambda n: node_weights[n], reverse=True)
        masked_nodes = sorted_nodes[num_total:]
    elif node_decay_method == 'num_per_grade':
        # Mask nodes except top n nodes per grade (in terms of weight)
        masked_nodes = []
        for grade in [1, 2, 3]:
            grade_nodes = [n for n in g.nodes() if node_grade[n] == grade]
            grade_nodes.sort(key=lambda n: node_weights[n], reverse=True)
            masked_nodes += grade_nodes[num_per_grade:]
    else:
        return g

    # Remove masked nodes from graph object
    g.remove_nodes_from(masked_nodes)
    return g


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
    g.add_node(jobname, type='job', weight=1, color='silver', grade=-1)
    g.add_nodes_from([(node, {'weight': attr, 'type': 'subject'})
                     for (node, attr) in job_subject_sim.items()])
    # set node color by grade
    color_dict = {1: 'green', 2: 'lightgreen', 3: 'yellowgreen'}
    for node in g.nodes():
        if node in subject_grade.keys():
            g.nodes[node]['color'] = color_dict[subject_grade[node]]
            g.nodes[node]['grade'] = subject_grade[node]

    g.add_weighted_edges_from(
        subject_subject, type='subject-subject', color='lightblue')
    g.add_weighted_edges_from(job_subject, type='job-subject', color='silver')

    return g


def graph_ploting(g, args):
    if args.node_decay == True:
        g = mask_nodes(g,
                       node_decay_method=args.node_decay_method,
                       weight=args.node_decay_weight,
                       num_total=args.node_decay_num_total,
                       num_per_grade=args.node_decay_num_per_grade)

    node_colors = nx.get_node_attributes(g, 'color').values()
    node_weight = nx.get_node_attributes(g, 'weight').values()
    edge_colors = nx.get_edge_attributes(g, 'color').values()
    edge_weight = nx.get_edge_attributes(g, 'weight').values()

    pos = nx.nx_agraph.graphviz_layout(g, prog="neato")
    plt.figure(figsize=(max(15, g.number_of_nodes()),
               max(15, g.number_of_nodes())))

    if args.node_scale != None:
        node_weight = minmax_scaler(node_weight, scale=args.node_scale)

    if args.edge_scale != None:
        edge_weight = minmax_scaler(edge_weight, scale=args.edge_scale)
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
    fn_img = f"{args.imgdir}{args.jobname}_grade{args.grade}_s2s{args.s2s_threshold}"
    plt.savefig(f'{fn_img}.png', dpi=300, bbox_inches='tight')
