# KT²: Knowledge-Tree-based Knowledge Tracing

This repository contains the official implementation of the paper:

**A Hierarchical Probabilistic Framework for Incremental Knowledge Tracing in Classroom Settings**

This framework introduces a probabilistic approach that models student knowledge over a structured KC tree, with support for real-time updates via expectation-maximization (EM) and graph-based propagation. The method is designed for classroom scenarios and emphasizes interpretability, modularity, and incremental updates.


## Project Structure

```
KT2/
├── scripts/             # Entry scripts for graph init and training
├── KT2/                 # Core algorithm modules
├── KC_tree/             # Knowledge concept tree construction and pruning
├── config/              # Configuration files and defaults
├── data/                # Datasets and KC trees
├── output/              # Output directory
├── preprocess/          # Data preprocessing tools
└── requirements.txt     # Python dependencies
```

## Installation

To install dependencies:

```
conda create -n kt2 python=3.10
conda activate kt2
pip install -r requirements.txt
```

## Quick Start

### 1. Build the Knowledge Graph

To build and prune the KC graph for a given dataset:

```
python scripts/init_graph.py --dataset DATASET_NAME
```

This will load data from `data/dataset/DATASET_NAME/` and save the KC tree template to `data/KC_tree/DATASET_NAME/`.

### 2. Run the KT² Algorithm

Once the graph is ready, run the full knowledge tracing pipeline:

```
python scripts/run_KT2.py
```

Paths and parameters can be configured in `config/config.yaml`. To run the experiment of each classroom setting, set `dataset` as `XES3G5M` or `MOOCRadar`. Set `root_node` as `Application_Module`/`Computation_Module`/`Counting_Module` for XES3G5M; `Wine_Knowledge`/`Circuit_Design`/`Education_Theory` for MOOCRadar.  

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Citation

If you find this project helpful or use it in your research, please consider citing:
```
@misc{kt2_arxiv2025,
title={A Hierarchical Probabilistic Framework for Incremental Knowledge Tracing in Classroom Settings},
author={Gao et al.},
year={2025},
eprint={arXiv:XXXX.XXXXX},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
```
A final citation with the official arXiv link will be added here once available.