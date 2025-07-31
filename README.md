# Inducing Neural Network Behavior via Constraint Optimization

This is the artifact associated with our AAAI-26 submission titled **"Inducing Neural Network Behavior via Constraint Optimization."**

## Requirements

- Python (≥3.7)
- PyTorch
- Gurobi Optimizer (we used the academic license)
  - Install Gurobi following instructions at: https://www.gurobi.com/documentation/

##  Suppress Training Confidence (STC)

You can run STC on either image or tabular datasets.

### For Image Datasets:

```
cd Image
python Main.py STC
```

### For Tabular Datasets:

```
cd Tabular
python Main.py STC
```

##  Change $m$ Classifications (CmC)

You can run CmC on either image or tabular datasets.

### For Image Datasets:

```
cd Image
python Main.py CmC <Misclassification Type> <Misclassification Count>
```

### For Tabular Datasets:

```
cd Tabular
python Main.py CmC <Misclassification Type> <Misclassification Count>
```

### Arguments

- `<Misclassification Type>`:
  - `A` – Misclassify **any** datapoints (randomly selected)
  - `C` – Misclassify **only correctly classified** datapoints

- `<Misclassification Count>`:
  - Integer specifying how many datapoints to modify via MILP
## Add Your Own Dataset

### Image

To add a custom image dataset:

- Update the `GetDataset` function in `Image/Main.py`.
- You can refer to examples using datasets from `torchvision.datasets` as well as datasets loaded from local files.

### Tabular

- If your dataset is available on OpenML, simply pass the appropriate dataset name as an argument.
- To use a local file, see the example in the `LoadDataset` function in `Tabular/Main.py`.

---

## Add Your Own Architecture

### Image

- Modify the `Image/CNNetworks.py` file to define your architecture.
- Also update the `GetModel` function in `Image/Main.py` to load your model.

### Tabular

- Modify the `Tabular/Networks.py` file to define your architecture.
- Also update the `TrainNN` function in `Tabular/Main.py` to incorporate your model.
