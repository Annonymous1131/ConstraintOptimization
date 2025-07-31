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

- `<Misclassification_Type>`:
  - `A` – Misclassify **any** datapoints (randomly selected)
  - `C` – Misclassify **only correctly classified** datapoints

- `<Misclassification_Count>`:
  - Integer specifying how many datapoints to modify via MILP


