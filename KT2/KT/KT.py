import json
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import numpy as np
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from KT2.module.data_cache import initialize_data, load_data, initialize_train_uids, get_train_uids
from KT2.KT.graph_update import update_graph

from config.config import (
    graph_dir,
    EM_output_dir,
    KT_output_dir,
    dataset,
    save_intermediate_graph,
    burn_in_size,
    root_node
)
from KC_tree.io import save_graph

EM_output_path = os.path.join(EM_output_dir, dataset, f'EM_results-Set{root_node}-burn-in{burn_in_size}')
KT_output_path = os.path.join(KT_output_dir, dataset, f'KT_results-Set{root_node}-burn-in{burn_in_size}')
merged_mapping_path = os.path.join(graph_dir, dataset, 'merged_mapping.json')
graph_path = os.path.join(graph_dir, dataset)

os.makedirs(KT_output_path, exist_ok=True)

translation_mapping = {
    'Application_Module': '应用题模块',
    'Computation_Module': '计算模块',
    'Counting_Module': '计数模块',
    'Wine_Knowledge': '白酒知识',
    'Circuit_Design': '运算放大器与电路设计',
    'Education_Theory': '教育理论与实践'
}

if root_node in translation_mapping.keys():
    root_name = translation_mapping[root_node]
else:
    root_name = root_node


def main():
    uids, train_size, _ = get_train_uids()
    difficulty_mapping = {"easy": 0, "medium": 1, "hard": 2}     


    with open(merged_mapping_path, 'r') as f:
        mapping = json.load(f)
    
    pred_prob_results = []
    true_labels = []
    student_ids = []
    questions = []
    kcs = []
    prior_probs = []
    phis = []
    epsilons = []
    r_diffs = []

    initialize_data()

    '''
    Incremental KT
    '''
    count_student = 0

    continue_from_last = False

    if continue_from_last:
        result = pd.read_csv(KT_output_path+'/KT_results.csv')
        done_uids = result['student_id'].unique()
        uids = [uid for uid in uids if uid not in done_uids]

    
    for uid in uids:
        student_true_labels = []
        student_pred_probs = []

        print(f'Processing student {count_student+1} of {len(uids)}')
        count_student += 1

        intermediate_graph_path = KT_output_path + f'/intermediate_graphs/student_{uid}'
        if not os.path.exists(intermediate_graph_path):
            os.makedirs(intermediate_graph_path)
        # Reset parameter graph for each student
        with open(EM_output_path+'/parameter_graphs/parameter_graph_step_final.json', 'r') as f:
            parameter_graph = json.load(f) # Share parameter graph for all students

        phi = parameter_graph[root_name]['phi'] # Share phi for all KCs
        r_diff = parameter_graph[root_name]['r_diff']
        epsilon = parameter_graph[root_name]['epsilon']


        # print('phi: ', phi)
        # print('r_diff: ', r_diff)
        # print('epsilon: ', epsilon)

        # Knowledge State Graph
        datas, graphs = load_data()
        original_graph = copy.deepcopy(graphs[uid])
        graph = graphs[uid]
        
        # Historical Exercise Data
        data = copy.deepcopy(datas[uid])
        data_size = train_size[uid]
        data = data[data_size:]

        # Knowledge Tracing
        index = 0
        while index < len(data):
            item = data[index]

            kc = item['kc']
            if kc in mapping:
                kc = mapping[kc]

            if index > 0:
                # NOTE: update all parameters by the new data
                graph, parameter_graph, r_diff = update_graph(graph, index, parameter_graph, r_diff, uids, uid, mapping)
                phi = parameter_graph[list(parameter_graph.keys())[0]]['phi'] # Share phi for all KCs
                

                # Save the intermediate graph
                if save_intermediate_graph:
                    with open(intermediate_graph_path + f'/parameter_graph_student_{uid}_index_{index}.json', 'w') as f:
                        json.dump(parameter_graph, f, indent=4, ensure_ascii=False)
                    save_graph(graph, intermediate_graph_path,f'graph_student_{uid}_index_{index}.json')

            
            student_ids.append(uid)

            true_answer = item['response']
            true_labels.append(true_answer)
            student_true_labels.append(true_answer)
            questions.append(item['question'])
            kcs.append(kc)
            difficulty_label = item['difficulty']

            prior_prob = calculate_prior(graph, kc, parameter_graph)
            prior_probs.append(prior_prob)

            # Get the emission probability of the kc
            phi = r_diff[difficulty_mapping[difficulty_label]]

            epsilon = parameter_graph[kc]['epsilon']

            phis.append(phi)
            epsilons.append(epsilon)
            r_diffs.append(r_diff)

            # Get the student's posterior 1 on the kc
            posterior1 = graph[kc].posterior1


            if posterior1 == None: # NOTE: for a new student, the posterior1 is the KC's prior probability
                posterior1 = calculate_prior(graph, kc, parameter_graph)

            # Prediction Result
            pred_prob = phi * posterior1 + epsilon * (1-posterior1)
            pred_prob_results.append(pred_prob)
            student_pred_probs.append(pred_prob)

            index += 1


        print('Final phi: ', phi)
        print('Final epsilon: ', epsilon)
        print('Final r_diff: ', r_diff)
        
        graphs[uid] = original_graph # restore the original graph

        result = pd.DataFrame({'student_id': student_ids, 'pred_prob': pred_prob_results, 'prior_prob': prior_probs, 'epsilon': epsilons, 'phi': phis, 'true_label': true_labels, 'kc': kcs, 'question': questions})
    
        # Save Results
        if not os.path.exists(KT_output_path):
            os.makedirs(KT_output_path)
        result.to_csv(KT_output_path+'/KT_results.csv', index=False)
    


    auc, accuracy, f1, best_threshold = evaluate(true_labels, pred_prob_results)

    # Compute 2-sigma error bars
    # mean_f1, se_f1, two_sigma_f1 = bootstrap_metric(true_labels, pred_prob_results, f1_score, threshold=best_threshold)
    # mean_auc, se_auc, two_sigma_auc = bootstrap_metric(true_labels, pred_prob_results, roc_auc_score, threshold=best_threshold)

    # print(f'2-sigma F1 Score: {two_sigma_f1:.4f}')
    # print(f'2-sigma AUC: {two_sigma_auc:.4f}')

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(true_labels, pred_prob_results)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    plt.savefig(KT_output_path+'/ROC_Curve.png')



# def bootstrap_metric(y_true, prob_scores, metric_fn, threshold, n_bootstrap=1000):
#     scores = []
#     n = len(y_true)
#     y_true = np.array(y_true)
#     prob_scores = np.array(prob_scores)

#     for _ in range(n_bootstrap):
#         indices = np.random.choice(n, size=n, replace=True)
#         y_sample = y_true[indices]
#         prob_sample = prob_scores[indices]

#         if metric_fn == f1_score:
#             binary_pred = (prob_sample >= threshold).astype(int)
#             score = f1_score(y_sample, binary_pred)
#         else:
#             score = metric_fn(y_sample, prob_sample)

#         scores.append(score)

#     scores = np.array(scores)
#     mean_score = np.mean(scores)
#     std_score = np.std(scores)
#     return mean_score, std_score, 2 * std_score


def evaluate(true_labels, pred_prob_results):
    # AUC
    auc = roc_auc_score(true_labels, pred_prob_results)
    print(f'AUC: {auc:.4f}')

    # Treshold Search to find the accuracy
    # thresholds = np.linspace(0.4, 0.6, 200)
    thresholds = np.linspace(0, 1, 1000)
    accuracies = []
    for threshold in thresholds:
        pred_labels = [1 if prob >= threshold else 0 for prob in pred_prob_results]
        accuracy = accuracy_score(true_labels, pred_labels)
        accuracies.append(accuracy)
    
    best_threshold = thresholds[np.argmax(accuracies)]
    print(f'Accuracy: {max(accuracies):.4f}')

    # F1 Score
    f1_scores = []
    for threshold in thresholds:
        pred_labels = [1 if prob >= threshold else 0 for prob in pred_prob_results]
        f1 = f1_score(true_labels, pred_labels)
        f1_scores.append(f1)
    best_f1_score = f1_scores[np.argmax(f1_scores)]

    print(f'F1 Score: {best_f1_score:.4f}')

    data_length = len(true_labels)

    # acc_se = 2 * np.std(accuracies) / np.sqrt(data_length)
    # print(f'2-sigma Accuracy SE: {acc_se:.4f}')

    return auc, max(accuracies), best_f1_score, best_threshold


def calculate_prior(graph, kc, parameter_graph):
    # Get the prior probability of the kc
    track_path = [kc]
    while True:
        if graph[kc].parents == []:
            prior_prob = parameter_graph[kc]['gamma_root']
            break
        kc = graph[kc].parents[0].name
        track_path.append(kc)

    while len(track_path) > 1:
        kc = track_path.pop()
        try:
            prior_prob = prior_prob + (1-prior_prob) * parameter_graph[kc]['gamma']
        except Exception as e:
            print(f'Error: {e}')
            breakpoint()
    return prior_prob


if __name__ == '__main__':
    initialize_train_uids()
    main()