<!-- - .datasets
- .exp
  - index.db



todo:
- dataset all return `Dataset` instance
- change cache
- add new dataset
- add new model
- produce plot



## Database
- results
- exp -->

# Weigthed Importance Distance with Active Learning Toolkits

This repo is used in COMP8800 research project in ANU. The structure is shown below:

- `allib`: This is an active learning toolkit consists of dataset, model, active learning strategy, evaluation, distance metrics and visualization.
  - `datasets`: Contains the dataset utilities, preprocessing, and example dataset implementation. It will generate cache files for dataset when the dataset is loaded for the first time, you can also refresh the cache by setting `reload=True` when loading the dataset.
  - `models`: Contains the example model implementations, pipeline, and active learning strategies.
    - `al`: Contains the active learning strategies
  - `metrics`: Contains the evaluation and distance metrics.
  - `plots`: Contains the visualization utilities.
  - `utils`: Contains the utilities for the toolkit.

- `app`: Contains the application using the toolkit to perform some of active learning experiments.
- `examples`: Contains the example of how to using the toolkit, and also some of experiments.
- `tutorial`: *Currently not in used*

## Usage

The toolkit is currently not published in PyPi, so you need to install it locally. After cloning the repo, you can install the dependencies by running:

```bash
pip install -r requirements.txt
```

Then you can perform or run the experiments in `app` or `examples` folder (if data is available).

### Generating Experimental Data

The finished experiment data recording the predictions, probabilities, selected batch, random seeds, etc. is too large and hard to be stored in the github repo. To generate new experimental data, for example, testing different distance metrics, you can find code examples in `app/comp_dist/draft.ipynb`. It will generate the `pickle` files in `ppl_cache` by default but you can change the path in the code.

### Plotting Results

We will use the `ppl_cache` files to generate the plots. You can find the code examples in `app/comp_dist/__main__.py`. The code will generate the plots in the `<specified_path>/pl_metrics/` folder.

> To run `app/comp_dist/__main__.py`, you need ensure you are in the project root and run the following command:
> ```bash
> python -m app.comp_dist
> ```
>


We also used a new type of plot in `examples/budget_plot.ipynb`. It is currently not implemented in `allib/plots`.

## Explaining the `examples` Folder

- `al_pipeline.ipynb` and `run_pipeline.ipynb`: These notebooks are used to test and demonstrate the active learning pipeline.
- `al_selection.ipynb`: This notebook is used to visualize the active learning selection strategies.
- `alstrategy.ipynb`: (Used in the early stage of the project) This notebook is used to test the active learning strategies.
- `cache_check.ipynb`: This notebook is used to inspect the cache files for debugging.
- `distance_validation`: This notebook is used to guarantee the distance metrics are implemented and indexed correctly.
- `feature_importance.ipynb`: This notebook is used to test the feature importance of the model.
- `generate_dist_mat.ipynb`: This notebook is used to generate the pairwise distance matrix for the dataset.
- `kmodes.ipynb`: This notebook is used to test the k-modes clustering and visualize the envaluation metrics of clustering.
- `load_datasets.ipynb`: This notebook is used to inspect the new dataset, which will be implemented `allib/datasets/avail_datasets.py` later.
- `budget_plot.ipynb`: This notebook is used to generate the budget plot for the paper.
- `random_cat.ipynb`: This notebook is used to test the impact of batch size for random strategy.
