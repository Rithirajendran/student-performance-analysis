"""Project-wide configuration: paths, random seed, column groups and plot style."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")                      # headless backend (no display needed)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"
EDA_DIR = ROOT / "outputs" / "eda"
EVAL_DIR = ROOT / "outputs" / "model_evaluation"
MODEL_DIR = ROOT / "models"
for d in (EDA_DIR, EVAL_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "Target"
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]      # alphabetical = LabelEncoder order

# ---- Column groups (36 input features) -------------------------------------
BINARY_COLS = ["Daytime/evening attendance", "Displaced", "Educational special needs",
               "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder",
               "International"]
NOMINAL_COLS = ["Marital status", "Application mode", "Course", "Previous qualification",
                "Nacionality", "Mother's qualification", "Father's qualification",
                "Mother's occupation", "Father's occupation"]      # integer codes, no order
# everything else is numeric (continuous grades, counts, macro-economic indicators)

# ---- Plot style: validated categorical slots 1-3 (dataviz skill palette) ----
CLASS_COLORS = {"Graduate": "#2a78d6", "Dropout": "#eb6834", "Enrolled": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

def save_fig(fig, path):
    fig.savefig(path)
    plt.close(fig)
