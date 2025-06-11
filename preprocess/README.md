# Preprocessing Pipeline for KT²

This folder contains the preprocessing scripts for preparing datasets and KC trees for use in the KT² framework. The pipeline includes KC tree construction, question translation, and post-processing.

## Folder Structure

```
preprocess/
├── full_data/                   # Intermediate merged raw data
├── question_construction/       # Question translation and formatting
├── tree_construction/           # Tree construction for MOOCRadar
├── class_simulation/            # Class-based scenario simulation
├── preprocess_config/           # Configuration files (e.g., OpenAI key, model selection)
├── preprocess_scripts/          # Preprocess scripts
```

## 1. Build Tree for MoocRadar

Construct the full KC graph with concept metadata:

```
python -m preprocess.preprocess_scripts.build_tree
```

This script processes the raw KC routes and constructs the hierarchical KC trees for each module.

## 2. Question Construction

### Step-by-step translation and validation pipeline:

Each of the following scripts should be executed **in order**, and all scripts rely on the setting `handle_fail` in the config file.

- `handle_fail = False`: process new data only
- `handle_fail = True`: reprocess previously failed or malformed items (e.g., invalid LLM output)

### a. Translate questions (initial)

```
python -m preprocess.question_construction.translate_question
```

### b. Check translation format and failures

```
python -m preprocess.question_construction.translate_check
```

### c. Fix or re-translate invalid entries

```
python -m preprocess.question_construction.fix_translation
```

### d. Merge corrected and original translations

```
python -m preprocess.question_construction.combine_fix
```

### e. Final output aggregation

```
python -m preprocess.question_construction.combine_question_file
```

This produces a clean merged file containing all questions, formatted and validated for further use in the KT² pipeline.

## Configuration

Most scripts read from:

```
preprocess/preprocess_config/config.yaml
```

You may change `handle_fail` and other parameters there as needed.

## Notes

- To re-run only failed samples, set `handle_fail = True` before each script.