# Kaggle Competition Pipeline

This pipeline is aims to facilitate the workflow from cleaning dataset to evaluate a model for a competition.

## Environment
Using pipenv:
```shell
pipenv shell
pipenv install
```

Using Anaconda:
```shell
conda create -n <env_name> --file requirements.txt
```

Using pip:
```shell
pip install -r requirements.txt
```

## Workflow
The basic workflow of a competition:
> data cleaning > train test split > model training > model evaluation

To ensure any kind of competition can implement such workflow, interfaces are integrated to necessary classes (IData, IModel) such that regardless of the formats of data and types of models, the pipeline can still run smoothly. <br>

Therefore, please make sure to inherit `IData` for DataManager and `IModel` for any models when you are integrating them for a competition. <br>

Start:
```shell
python main.py -c <competition_name> -m <model_name>
```