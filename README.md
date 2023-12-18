# De2: Debug from Design

## Getting Started

Create conda environment:
```
conda create --name de2 python=3.7.6
```

Then install other dependency:
```
pip install -r requirements.txt
```


## Quick Demo
We provide a demo to run the adapter on the 'NPC cut in' example.

1. Run Extract_Dataframe.ipynb to preprocess the raw record dataset from Apollo.
2. Two-stage event detection on the record dataset.
```
conda activate de2
python .\approach\events_detection.py
```
3. Critical feature evaluation.
```
python .\approach\feature_importance.py
```



