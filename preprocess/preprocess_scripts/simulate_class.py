import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.class_simulation.select_subtree_set import main as select_subtree_set

from preprocess.preprocess_config.config import dataset


if dataset == 'MOOCRadar':
    target_nodes = [
        '白酒知识',
        '运算放大器与电路设计',
        '教育理论与实践'
    ]
elif dataset == 'XES3G5M':
    target_nodes = [
        '计数模块',
        '计算模块',
        '应用题模块'
    ]


def main():
    for target_node in target_nodes:
        select_subtree_set(target_node)

if __name__ == '__main__':
    main()