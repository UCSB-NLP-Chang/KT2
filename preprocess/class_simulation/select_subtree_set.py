import os
import json
import sys
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KC_tree.io import load_graph
from preprocess.preprocess_config.config import (
    dataset,
    dataset_dir,
    graph_dir
)

recording_path = os.path.join(dataset_dir, dataset, 'recordings.jsonl')
question_path = os.path.join(dataset_dir, dataset, 'question_info.json')
graph_path = os.path.join(graph_dir, dataset)
merged_mapping_path = os.path.join(graph_dir, dataset, 'merged_mapping.json')

target_mean = 0.65
target_std = 0.15
k = 100

translation_mapping = {
    '应用题模块': 'Application_Module',
    '计算模块': 'Computation_Module', 
    '计数模块': 'Counting_Module',
    '白酒知识': 'Wine_Knowledge',
    '运算放大器与电路设计': 'Circuit_Design',
    '教育理论与实践': 'Education_Theory'
}

def main(target_node):
    with open(question_path, 'r') as f:
        questions = json.load(f)

    single_graph = load_graph(graph_path, f'pruned_knowledge_graph.json')

    all_students_level_2_nodes = dict()
    all_students_level_2_nodes_questions = dict()
    all_students_level_2_nodes_accuracy = dict()
    all_uids = []

    with open(recording_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            uid = data['student_id']
            logs = data['exercises_logs']
            all_responses = []
            all_questions = []
            all_kcs = []
            if len(logs) < 100:
                continue
            all_uids.append(uid)
            for index in range(len(logs)):
                tmp = dict()
                que = logs[index]
                response = int(data['is_corrects'][index])
                if response == -1:
                    breakpoint()
                assert que in questions.keys()
                question = questions[que]
                tmp['question'] = question['content']
                tmp['response'] = response
                tmp['kc'] = question['kc']
                all_questions.append(tmp['question'])
                all_responses.append(tmp['response'])
                all_kcs.append(question['kc'])
            level_2_nodes , level_2_nodes_questions, level_2_nodes_responses = identify_subtree(all_kcs, all_questions, all_responses, single_graph)
            if level_2_nodes != None:
                all_students_level_2_nodes[uid] = level_2_nodes
                all_students_level_2_nodes_questions[uid] = level_2_nodes_questions
                level_2_nodes_accuracy = dict()
                for key, value in level_2_nodes_responses.items():
                    accuracy = np.mean(value)
                    level_2_nodes_accuracy[key] = accuracy
                all_students_level_2_nodes_accuracy[uid] = level_2_nodes_accuracy

    print_nodes_statistics(all_students_level_2_nodes)

    select_subtree(all_students_level_2_nodes, all_students_level_2_nodes_questions)


    save_dir = os.path.join(dataset_dir, dataset, 'final_split/subtree_split/')


    # Select students of a given subtree
    selected_students = select_students_normal_distribution(target_node, all_students_level_2_nodes, all_students_level_2_nodes_questions, all_students_level_2_nodes_accuracy)
    assert len(selected_students) == 100


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if target_node in translation_mapping.keys():
        target_node = translation_mapping[target_node]
    with open(os.path.join(save_dir, f'selected_classroom_students_{target_node}{target_mean}.txt'), 'w') as f:
        for student in selected_students:
            f.write(f"{student}\n")
    

def print_nodes_statistics(all_students_nodes):
    overall_nodes = dict()
    for student_nodes in all_students_nodes.values():
        for node in student_nodes:
            if node not in overall_nodes.keys():
                overall_nodes[node] = 0
            overall_nodes[node] += student_nodes[node]
    # Print maximum 5 nodes
    for node, count in sorted(overall_nodes.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{node}: {count}")


def identify_subtree(kcs, questions, responses, graph):

    with open(merged_mapping_path, 'r') as f:
        mapping = json.load(f)

    level_2_nodes = dict()
    level_2_nodes_questions = dict()
    level_2_nodes_responses = dict()

    for kc, question, response in zip(kcs, questions, responses):
        if kc in mapping.keys():
            kc = mapping[kc]
        if kc not in graph.keys():
            breakpoint()
        assert kc in graph.keys()
        level_3_node, level_2_node = trace_up(graph[kc])
        if level_2_node not in level_2_nodes.keys():
            level_2_nodes[level_2_node] = 1
            level_2_nodes_questions[level_2_node] = [question]
            level_2_nodes_responses[level_2_node] = [response]
        else:
            level_2_nodes[level_2_node] += 1
            level_2_nodes_questions[level_2_node].append(question)
            level_2_nodes_responses[level_2_node].append(response)

    return level_2_nodes, level_2_nodes_questions, level_2_nodes_responses

def trace_up(node):
    previous_previous_parent = None
    previous_parent = None
    current_parent = None
    while True:
        current_parent = node.parents
        if len(current_parent) == 0:
            break
        previous_previous_parent = previous_parent
        previous_parent = node
        node = current_parent[0]
    if previous_parent != None:
        previous_parent = previous_parent.name
    if previous_previous_parent != None:
        previous_previous_parent = previous_previous_parent.name
    return previous_previous_parent, previous_parent

def select_students(target_node, all_students_nodes, all_students_nodes_questions, top_k=200):
    selected_students = dict()

    for student in all_students_nodes.keys():
        student_nodes = all_students_nodes[student]
        if target_node not in student_nodes.keys():
            continue
        selected_students[student] = student_nodes[target_node]
    
    # Filter out students that have at least 80 interactions in the target node
    # selected_students = {student: count for student, count in selected_students.items() if count >= 100}
    selected_students = sorted(selected_students.items(), key=lambda x: x[1], reverse=True)[:top_k]
    # print('Number of students that have at least 100 interactions in the target node: ', len(selected_students))

    # Count how many interactions the last student has that is in the target node
    last_uid = list(selected_students)[-1][0]
    last_student_nodes = all_students_nodes[last_uid]
    print('Total Number of Students: ', len(all_students_nodes))
    print('The last student has', last_student_nodes[target_node], 'interactions in the target node ', target_node)
    return selected_students

def select_students_normal_distribution(target_node, all_students_nodes, all_students_nodes_questions, all_students_level_2_nodes_accuracy):

    interaction_threshold = 50
    unique_questions_threshold = 50

    selected_students = dict()

    node_accuracy = dict()
    for key in all_students_level_2_nodes_accuracy.keys():
        if target_node not in all_students_level_2_nodes_accuracy[key].keys():
            continue
        node_accuracy[key] = all_students_level_2_nodes_accuracy[key][target_node]

    for student in all_students_nodes.keys():
        student_nodes = all_students_nodes[student]
        if target_node not in student_nodes.keys():
            continue
        selected_students[student] = student_nodes[target_node]
    
    # Filter out students that have at least 80 interactions in the target node
    selected_students = {student: count for student, count in selected_students.items() if count >= interaction_threshold}
    # selected_students = sorted(selected_students.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print('Number of students that have at least 50 interactions in the target node: ', len(selected_students))

    unique_questions_count = dict()
    for student in selected_students.keys():
        student_questions = all_students_nodes_questions[student][target_node]
        unique_questions_count[student] = len(set(student_questions))
    
    # Filter out students that have at least 50 unique questions in the target node
    selected_students = {student: count for student, count in selected_students.items() if unique_questions_count[student] >= unique_questions_threshold}
    print('Number of students that have at least 50 unique questions in the target node: ', len(selected_students))


    selected_node_accuracy = dict()
    for student in selected_students.keys():
        selected_node_accuracy[student] = node_accuracy[student]

    sampled_students = sample_normal(selected_node_accuracy, target_mean, target_std, k)

    # print total number of questions in the selected students and the minimum number of questions
    total_questions = 0
    min_questions = float('inf')
    for student in sampled_students:
        total_questions += len(all_students_nodes_questions[student][target_node])
        min_questions = min(min_questions, len(all_students_nodes_questions[student][target_node]))
    print('Total number of questions in the selected students: ', total_questions)
    print('Minimum number of questions in the selected students: ', min_questions)
    return sampled_students

def sample_normal(node_accuracy, target_mean=0.7, target_std=0.15,k=100):
    # Sample k students from the node accuracy without replacement
    # The sample should be drawn from a normal distribution with the mean and std as the target_mean and target_std

    # 1. Generate target distribution (normal distribution) with 100 samples
    random.seed(42)
    target_accs = np.random.normal(loc=target_mean, scale=target_std, size=k)
    target_accs = np.clip(target_accs, 0, 1)  # Clip to the valid range

    # 2. Convert the original data to a list for easier processing
    uid_acc_list = list(node_accuracy.items())
    available_uids = set(node_accuracy.keys())

    # 3. Match each target acc with the closest student in node_accuracy
    sampled_students = []
    used_uids = set()

    for target in target_accs:
        # Find the student with the closest accuracy to the target that hasn't been used yet
        remaining = [(uid, acc) for uid, acc in uid_acc_list if uid not in used_uids]
        if not remaining:
            break
        uid, acc = min(remaining, key=lambda x: abs(x[1] - target))
        sampled_students.append(uid)
        used_uids.add(uid)

    # Print the sampled uid list
    print('Average accuracy: ', np.mean(list(node_accuracy[student] for student in sampled_students)))
    print('Std accuracy: ', np.std(list(node_accuracy[student] for student in sampled_students)))

    return sampled_students


def select_subtree(all_students_nodes, all_students_nodes_questions):
    all_subtree_nodes = []
    for student in all_students_nodes.keys():
        subtree_nodes = all_students_nodes[student].keys()
        all_subtree_nodes.extend(subtree_nodes)
    all_subtree_nodes = list(set(all_subtree_nodes))

    subtree_stats = dict()

    for target_node in all_subtree_nodes:
        selected_students = dict()
        for student in all_students_nodes.keys():
            student_nodes = all_students_nodes[student]
            if target_node not in student_nodes.keys():
                continue
            selected_students[student] = student_nodes[target_node]
        # Filter out students that have at least 80 interactions in the target node
        selected_students = {student: count for student, count in selected_students.items() if count >= 80}
        # selected_students = sorted(selected_students.items(), key=lambda x: x[1], reverse=True)[:top_k]
        # print('Number of students that have at least 100 interactions in the target node: ', len(selected_students))

        unique_questions_count = dict()
        for student in selected_students.keys():
            student_questions = all_students_nodes_questions[student][target_node]
            unique_questions_count[student] = len(set(student_questions))
        
        # Filter out students that have at least 50 unique questions in the target node
        selected_students = {student: count for student, count in selected_students.items() if unique_questions_count[student] >= 50}

        for student in selected_students.keys():
            if target_node not in subtree_stats.keys():
                subtree_stats[target_node] = dict()
            subtree_stats[target_node] = len(selected_students)
    
    # Fiter out the largest 3 subtrees
    sorted_subtree_stats = sorted(subtree_stats.items(), key=lambda x: x[1], reverse=True)[:5]
    print(sorted_subtree_stats)
        
if __name__ == "__main__":
    main()