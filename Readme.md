# Data Mining Lab 1-12

Data mining lab scripts covering Python basics, preprocessing, EDA, frequent
pattern mining, classification, clustering and ensemble methods.

| Lab | Topic | File |
|-----|-------|------|
| 1 | Python basics, variables, lists/tuples/dicts, functions, loops, CSV with Pandas | `lab1.py` |
| 2 | Data preprocessing, missing values, duplicates, encoding, scaling, statistics | `lab2.py` |
| 3 | Data visualisation and EDA | `lab3.py` |
| 4 | Frequent pattern mining - Apriori | `lab4.py` |
| 5 | Frequent pattern mining - FP-Growth | `lab5.py` |
| 6 | Classification - Decision Trees | `lab6.py` |
| 7 | Classification - Naive Bayes | `lab7.py` |
| 8 | Classification - Support Vector Machines | `lab8.py` |
| 9 | Clustering - K-Means | `lab9.py` |
| 10 | Clustering - Hierarchical clustering and DBSCAN | `lab10.py` |
| 11 | Outlier detection and ensemble methods | `lab11.py` |
| 12 | Fraud detection web app (Streamlit) | `Lab-12/app.py` |

## Folder structure

```
.
├── lab1.py ... lab11.py     lab scripts
├── generate_datasets.py     recreates the datasets in data/ (standard library only)
├── data/                    datasets used by the scripts (generated, not tracked)
├── outputs/                 generated at run time (figures and result tables)
├── Lab-12/                  Streamlit app and the trained model (Base_model.pkl)
├── requirements.txt
├── .gitignore
└── Readme.md
```

## Setup

### 1. Create the virtual environment

Run these commands inside the project folder (the one containing `requirements.txt`).

**Windows - CMD**

```cmd
python -m venv .venv
```

**Windows - PowerShell**

```powershell
python -m venv .venv
```

**Linux - bash / zsh**

```bash
python3 -m venv .venv
```

**Linux - fish**

```fish
python3 -m venv .venv
```

### 2. Activate the virtual environment

**Windows - CMD**

```cmd
.venv\Scripts\activate
```

**Windows - PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux - bash / zsh**

```bash
source .venv/bin/activate
```

**Linux - fish**

```fish
source .venv/bin/activate.fish
```

Deactivate (all systems)

```text
deactivate
```

### 3. Install the packages

```text
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the installation

```text
pip list
```

### 4. Generate the datasets

The `data/` folder is not tracked by git, so it has to be created once after
cloning the repository:

```text
python generate_datasets.py
```

The script only uses the standard library and writes five CSV files into `data/`:

| File | Rows | Used by |
|------|------|---------|
| `credit_card_transactions.csv` | 15025 | labs 2, 3, 6, 7, 8, 11 |
| `market_basket_transactions.csv` | 250 | labs 4, 5 |
| `customer_segments.csv` | 240 | labs 9, 10 |
| `text_corpus.csv` | 24 | lab 7 |
| `student_marks.csv` | 12 | lab 1 |

### 5. Run the labs

```text
python lab1.py
python lab2.py
python lab3.py
python lab4.py
python lab5.py
python lab6.py
python lab7.py
python lab8.py
python lab9.py
python lab10.py
python lab11.py
```

Each script prints its results in the terminal and saves the figures and result
tables in the `outputs/` folder.

### 6. Run the Streamlit app (Lab 12)

The app loads `Base_model.pkl` from its own folder, so start it from there:

```text
cd Lab-12
streamlit run app.py
```

## Notes

* The scripts use the matplotlib backend `Agg`, so the figures are saved as PNG
  files in `outputs/` and no window is opened. To see the plots on screen,
  remove the line `matplotlib.use("Agg")` from the top of the lab file and add
  `plt.show()` after the plot.
* `data/`, `outputs/`, `.venv/` and `__pycache__/` are ignored by git, so only
  the code, `generate_datasets.py`, `Readme.md` and `requirements.txt` are pushed
  to the repository.
* The random seed is fixed (`random_state = 42`), so the results stay identical
  from one run to another.
