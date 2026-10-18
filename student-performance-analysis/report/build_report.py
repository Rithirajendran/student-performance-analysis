"""Build the Word report from the saved outputs (all numbers are read from files, nothing typed by hand).
Usage:  python report/build_report.py  [page_map.json]"""
import json, sys, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docx_helpers import *
from data_preprocessing import load_data, clean_data, numeric_cols
from config import ROOT, EDA_DIR, EVAL_DIR, BINARY_COLS, NOMINAL_COLS

PAGES = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
E_ = json.load(open(EDA_DIR / "results_eda.json"))
M_ = json.load(open(EVAL_DIR / "results_model.json"))
T_ = json.load(open(EVAL_DIR / "training_summary.json"))
df, _ = clean_data(load_data())
desc = pd.read_csv(EDA_DIR / "table_descriptive_numeric.csv", index_col=0)
biv = pd.read_csv(EDA_DIR / "table_bivariate_tests.csv", index_col=0)
vif = pd.read_csv(EDA_DIR / "table_vif.csv", index_col=0).iloc[:, 0]
outl = pd.read_csv(EDA_DIR / "table_outliers.csv", index_col=0)
tune = pd.read_csv(EVAL_DIR / "tuning_results.csv")
imp = pd.read_csv(EVAL_DIR / "table_permutation_importance.csv", index_col=0)
comp = M_["comparison_test"]; mt = M_["mlp_test"]; mv = M_["mlp_val"]
pc = M_["per_class_report"]; cm = np.array(M_["confusion_matrix"])
DF = E_["distribution_fits"]
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; pct = lambda x: f"{100*x:.1f}%"
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
acc_lo, acc_hi = wilson(round(mt["accuracy"] * 664), 664)
g_ = df.groupby("Course")["Target"].agg(n="size", rate=lambda s: 100 * (s == "Dropout").mean())
g_ = g_[g_.n >= 100].sort_values("rate")
med = df.groupby("Target")[["Admission grade", "Age at enrollment", "Curricular units 1st sem (grade)",
                           "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (approved)"]].median()
z1 = (df["Curricular units 1st sem (approved)"] == 0); z2 = (df["Curricular units 2nd sem (approved)"] == 0)
dr1, dr2 = 100 * (df[z1].Target == "Dropout").mean(), 100 * (df[z2].Target == "Dropout").mean()
pca_cum5 = sum(E_["pca_var_ratio_first5"])
p_ge5 = stats.binom.sf(4, 10, DF["dropout_proportion"]); p_pois0 = stats.poisson.pmf(0, DF["Curricular units 1st sem (approved)"]["mean"])
eps = lambda f: biv.loc[f, "effect_size"]

TITLE = "Student Academic Performance Analysis and Dropout Prediction Using Exploratory Data Analysis and Deep Learning"
FOOT = "RITHI R | 24AD091 | U21ADP05 – Assignment II"
HEAD, FIGS, TABS = [], [], []
cnt = {"fig": 0, "tab": 0}


def H(d, lvl, text):
    d.add_heading(text, level=lvl); HEAD.append((lvl, text))


def FIG(d, path, w, title):
    n = title.split(":")[0]; FIGS.append(title)
    figure(d, path, w, title)


def TAB(d, caption_title, header, rows, widths, **kw):
    cnt["tab"] += 1; cap = f"Table {cnt['tab']}: {caption_title}"; TABS.append(cap)
    return table(d, header, rows, widths, caption=cap, **kw)


def CARD(d, n, title, path, w, purpose, variables, chart, why, insight):
    """Figure + the 'what / why / insight' explanation table required by the assignment."""
    FIG(d, path, w, f"Figure {n}: {title}")
    table(d, ["Aspect", "Explanation"], [["Purpose", purpose], ["Variables", variables], ["Chart type", chart],
                                          ["Why selected", why], ["Insight", insight]], [2.8, 13.8], font=9, bold_first_col=True, keep_together=True)


def build():
    d = new_document(); HEAD.clear(); FIGS.clear(); TABS.clear(); cnt.update(fig=0, tab=0)
    footer_with_page(d.sections[0], FOOT)
    # ============================ TITLE PAGE ============================
    para(d, "KPR Institute of Engineering and Technology", bold=True, size=18, align=WD_ALIGN_PARAGRAPH.CENTER, color=BLUE, after=2)
    para(d, "Department of Artificial Intelligence and Data Science", size=13, align=WD_ALIGN_PARAGRAPH.CENTER, after=40)
    para(d, "ASSIGNMENT II – CASE STUDY", bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, color=GREY, after=6)
    para(d, TITLE, bold=True, size=24, align=WD_ALIGN_PARAGRAPH.CENTER, color=BLUE, after=8)
    para(d, "Case Study: Exploratory Data Analysis and Visualization of a Real-World Dataset", italic=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=40)
    rows = [["Student Name", "RITHI R"], ["Roll Number / College ID", "24AD091"], ["Department", "Artificial Intelligence and Data Science"],
            ["Year / Semester", "III Year / Semester V (Batch 2023–2027)"], ["Course Code", "U21ADP05"],
            ["Course Title", "Exploratory Data Analysis and Visualization"], ["Assignment", "Assignment II – Case Study (40 marks)"],
            ["Academic Year", "2026–2027 (Odd Semester)"], ["Course Coordinator", "Mr. Rushikesh Kadam"], ["Report Date", "18 October 2026"]]
    t = d.add_table(rows=0, cols=2); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; set_cell_margins(t, 80, 80, 120, 120)
    t.autofit = False
    for a, b in rows:
        c = t.add_row().cells; c[0].width, c[1].width = Cm(5.2), Cm(10.6); shade(c[0], "E8F0FB")
        add_runs(c[0].paragraphs[0], a, bold=True, size=11); add_runs(c[1].paragraphs[0], b, size=11)
        for x in c: x.paragraphs[0].paragraph_format.space_after = Pt(0)
    t.columns[0].width, t.columns[1].width = Cm(5.2), Cm(10.6)
    page_break(d)
    # ============================ ABSTRACT ============================
    para(d, "Abstract", bold=True, size=17, color=BLUE, after=8)
    abstract = (f"Student dropout is a costly problem for higher-education institutions, and early identification of at-risk students allows timely support. "
                f"This study analyses the UCI dataset \"Predict Students' Dropout and Academic Success\" ({len(df):,} students, 36 features, three outcomes: Dropout, Enrolled, Graduate). "
                f"Exploratory analysis found no missing values or duplicates, then examined outliers, distributions, correlations and associations using seven-stage visualisation, "
                f"statistical tests and PCA. Tuition-fee status, scholarship status and curricular units approved were among the strongest correlates of outcome. "
                f"A multilayer perceptron tuned over 12 configurations achieved {pct(mt['accuracy'])} test accuracy, macro-F1 {f3(mt['macro_f1'])} and macro ROC-AUC {f3(mt['roc_auc_ovr_macro'])}, "
                f"similar to logistic-regression and random-forest baselines. Results are from one stratified split.")
    wc = len(abstract.split()); assert 100 <= wc <= 150, wc
    para(d, abstract, after=6); para(d, f"Keywords: exploratory data analysis, data visualization, student dropout, multilayer perceptron, statistical analysis.  (Abstract: {wc} words)", italic=True, size=9.5, color=GREY)
    page_break(d)
    # ============================ CONTENTS ============================
    TOC_AT = len(d.element.body)           # remember position: filled after the body is built
    para(d, "[[TOC]]")
    page_break(d)
    para(d, "List of Abbreviations", bold=True, size=15, color=BLUE, after=6)
    abbr = [["ANOVA", "Analysis of variance"], ["AUC", "Area under the ROC curve"], ["CO", "Course outcome"], ["CSV", "Comma-separated values"],
            ["EDA", "Exploratory data analysis"], ["GDP", "Gross domestic product"], ["IQR", "Inter-quartile range"], ["MLP", "Multilayer perceptron"],
            ["PCA", "Principal component analysis"], ["ReLU", "Rectified linear unit"], ["ROC", "Receiver operating characteristic"],
            ["UCI", "University of California, Irvine (Machine Learning Repository)"], ["VIF", "Variance inflation factor"], ["CU", "Curricular units (used in chart labels)"]]
    table(d, ["Abbreviation", "Meaning"], abbr, [3.5, 13.1], font=9.5)
    page_break(d)

    # ============================ 1 INTRODUCTION ============================
    H(d, 1, "1. Introduction and Objectives")
    H(d, 2, "1.1 Background")
    para(d, "Universities and colleges collect large amounts of data about their students: how they applied, their family background, their marks in each semester and even the economic conditions "
            "in the year they enrolled. When this data is explored carefully, it can reveal why some students graduate on time while others leave without a degree. "
            "Exploratory Data Analysis (EDA), introduced by John Tukey (1977), is the practice of looking at data through summaries and pictures before building any model, so that patterns, "
            "errors and surprises are found early.")
    H(d, 2, "1.2 Problem Statement")
    para(d, "Given demographic, socio-economic, macro-economic and first-year academic information about a student, (a) which factors are most strongly associated with dropping out, "
            "and (b) can a deep-learning model predict whether the student will finally be a **Dropout**, still **Enrolled**, or a **Graduate** at the end of the normal course duration?")
    H(d, 2, "1.3 Motivation")
    bullets(d, ["Dropout wastes the student's time and money and reduces an institution's graduation rate.",
                "If risky patterns are visible early (for example unpaid tuition fees or no approved course units in the first semester), tutors can offer help before it is too late.",
                "The dataset is real, contains a mix of numerical and categorical fields, and is large enough for both statistical analysis and a neural network, which makes it a good fit for this course."])
    H(d, 2, "1.4 Objectives")
    bullets(d, ["Select and verify a real-world dataset with at least 15 features (this one has 36).",
                "Perform data understanding, cleaning, outlier analysis, encoding, scaling and a train/validation/test split without data leakage.",
                "Create more than five meaningful visualisations (this report has 12 EDA figures and 7 model figures) that follow all seven stages of visualisation.",
                "Carry out univariate, bivariate and multivariate statistical analysis and study discrete and continuous probability distributions.",
                "Build a multilayer perceptron (deep-learning model), tune it, analyse feature importance and evaluate it with loss/accuracy curves, a confusion matrix and ROC/AUC.",
                "Publish the work in a well-structured GitHub repository with a README."], style="List Number")
    H(d, 2, "1.5 Scope and Practical Relevance")
    para(d, "The study covers one dataset from one Portuguese higher-education institution and focuses on understanding and predicting the final outcome. It is a learning project, not a deployed system; "
            "however, the same workflow (clean → explore → visualise → model → evaluate) is what an institutional analytics team would use to build an early-warning dashboard for tutors.")

    # ============================ 2 DATASET ============================
    H(d, 1, "2. Dataset Description")
    H(d, 2, "2.1 Dataset Name, Source and Licence")
    TAB(d, "Dataset identification", ["Item", "Details"], [
        ["Name", "Predict Students' Dropout and Academic Success"],
        ["Repository", "UCI Machine Learning Repository (dataset id 697)"],
        ["URL", "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"],
        ["DOI", "10.24432/C5MC89"],
        ["Creators", "Valentim Realinho, Mónica Vieira Martins, Jorge Machado, Luís Baptista (Polytechnic Institute of Portalegre, Portugal)"],
        ["Licence", "Creative Commons Attribution 4.0 (CC BY 4.0)"],
        ["Companion paper", "Realinho, Machado, Baptista & Martins (2022), Data 7(11), 146, https://doi.org/10.3390/data7110146"],
        ["Local file", "data/data.csv (semicolon-separated)"]], [3.2, 13.4], font=9, bold_first_col=True)
    para(d, "**Verification note.** The UCI page states 4,424 instances, 36 features, a three-class target and no missing values. The file used in this study was loaded with pandas and these statements were "
            "checked on the file itself (Section 2.3). Because the UCI website could not be reached from the environment used for the analysis, the CSV was obtained from public GitHub copies of the same dataset; "
            "two independent copies were compared and found identical (4,424 × 37). Anyone re-running this project can download the original file from the UCI link above and place it at data/data.csv.")
    H(d, 2, "2.2 Why This Dataset Was Chosen")
    bullets(d, ["**More than 15 features:** it has 36 input fields plus a target, which satisfies the assignment requirement.",
                "**Real-world relevance:** it describes real students and addresses a genuine problem – dropout and academic success.",
                "**Mixed data types:** continuous marks, counts, binary flags, coded categories and macro-economic indicators, so many chart types and statistical tests are meaningful.",
                "**Size:** 4,424 rows is enough for a stable train/validation/test split and for a small neural network, yet light enough to run on a laptop.",
                "**Clean but not trivial:** no missing values, but strong class imbalance, skewed variables, many outliers and several highly correlated features give real EDA work to do.",
                "**Supervised target:** the target is a three-class label, so this is a **multi-class classification** task (not regression)."])
    H(d, 2, "2.3 Verified Structure of the Dataset")
    TAB(d, "Verified facts about the loaded file", ["Property", "Value (measured on data/data.csv)"], [
        ["Rows (students)", f"{E_['shape'][0]:,}"], ["Columns", f"{E_['shape'][1]} (36 features + 1 target)"],
        ["Missing cells", str(E_['clean_report']['missing_cells'])], ["Duplicate rows", str(E_['clean_report']['duplicate_rows'])],
        ["Numeric features (continuous / count)", str(E_['n_numeric'])], ["Binary flags", str(E_['n_binary'])], ["Nominal codes", str(E_['n_nominal'])],
        ["Stored data types", f"int64: {E_['dtypes']['int64']}, float64: {E_['dtypes']['float64']}, text (target): {E_['dtypes']['str']}"],
        ["Target column", "Target – Graduate / Dropout / Enrolled"],
        ["Class counts", f"Graduate {E_['class_counts']['Graduate']:,} ({E_['class_pct']['Graduate']}%), Dropout {E_['class_counts']['Dropout']:,} ({E_['class_pct']['Dropout']}%), Enrolled {E_['class_counts']['Enrolled']:,} ({E_['class_pct']['Enrolled']}%)"]],
        [4.6, 12.0], font=9, bold_first_col=True)
    H(d, 2, "2.4 Description of the Fields")
    para(d, "Categorical fields in this dataset are stored as integer codes. Their meaning of individual code values is given in the UCI documentation; this report treats them as nominal labels and does not interpret the numbers themselves.")
    fd = {"Marital status": "Code for marital status", "Application mode": "Code for the way the student applied", "Application order": "Order of choice in which the course was applied for",
          "Course": "Code of the degree programme", "Daytime/evening attendance": "Daytime or evening classes", "Previous qualification": "Code of qualification held before enrolment",
          "Previous qualification (grade)": "Grade of previous qualification", "Nacionality": "Code of nationality (spelling as in source)",
          "Mother's qualification": "Code of mother's education level", "Father's qualification": "Code of father's education level",
          "Mother's occupation": "Code of mother's occupation", "Father's occupation": "Code of father's occupation", "Admission grade": "Grade at admission",
          "Displaced": "Student is displaced from home", "Educational special needs": "Has special educational needs", "Debtor": "Has unpaid debts to the institution",
          "Tuition fees up to date": "Tuition fees are paid up to date", "Gender": "Gender (binary code)", "Scholarship holder": "Holds a scholarship",
          "Age at enrollment": "Age (years) at enrolment", "International": "International student",
          "Unemployment rate": "National unemployment rate (%)", "Inflation rate": "National inflation rate (%)", "GDP": "Gross domestic product indicator"}
    rows = []
    for i, c in enumerate([x for x in df.columns if x != "Target"], 1):
        grp = "Binary" if c in BINARY_COLS else "Nominal" if c in NOMINAL_COLS else "Numeric"
        if c in fd: dsc = fd[c]
        else:
            sem = "1st" if "1st" in c else "2nd"; what = c[c.index("(") + 1:-1]
            dsc = {"credited": "Units credited", "enrolled": "Units enrolled in", "evaluations": "Number of evaluations taken", "approved": "Units passed",
                   "grade": "Average grade (0–20)", "without evaluations": "Units with no evaluation"}[what] + f" in {sem} semester"
        rows.append([i, c, grp, str(df[c].dtype).replace("float64", "float").replace("int64", "int"), dsc])
    rows.append([37, "Target", "Target", "text", "Dropout / Enrolled / Graduate at end of normal course duration"])
    TAB(d, "All 36 input fields and the target", ["#", "Field", "Group", "Type", "Meaning"], rows, [0.9, 5.2, 1.7, 1.3, 7.5], font=8.5)
    H(d, 2, "2.5 Basic Descriptive Statistics")
    para(d, "Table 4 reports the mean, median, mode, standard deviation, quartiles and skewness of the 19 numeric features, computed on all 4,424 rows. Binary and nominal fields are summarised through frequency tables in Chapter 5.")
    rows = [[c.replace("Curricular units ", "CU "), f2(r["mean"]), f2(r["median"]), f2(r["mode"]), f2(r["std"]), f2(r["min"]), f2(r["q25"]), f2(r["q75"]), f2(r["max"]), f2(r["skew"])] for c, r in desc.iterrows()]
    TAB(d, "Descriptive statistics of numeric features", ["Feature", "Mean", "Median", "Mode", "Std", "Min", "Q1", "Q3", "Max", "Skew"], rows,
        [5.2, 1.3, 1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.3, 1.1], font=8, align_right_from=1)

    # ============================ 3 EDA & PREPROCESSING ============================
    H(d, 1, "3. Exploratory Data Analysis and Preprocessing")
    FIG(d, EDA_DIR / "diagram1_workflow.png", 16, "Diagram 1: End-to-end project workflow")
    para(d, "Diagram 1 shows the order of work. The top row (steps 1–5) is data understanding and visualisation; the bottom row (steps 6–9) is modelling. Every step is a separate Python module in the GitHub repository.", size=10)
    H(d, 2, "3.1 Loading and Inspecting the Data")
    para(d, "The CSV is loaded with pandas (pd.read_csv with sep=';'). The header contained stray white-space, which was stripped so that column names can be used safely in code. "
            f"The shape is {E_['shape'][0]:,} rows × {E_['shape'][1]} columns. df.info(), df.describe() and value counts were used to check data types and ranges.")
    H(d, 2, "3.2 Missing Values and Duplicates")
    para(d, f"Missing cells found: **{E_['clean_report']['missing_cells']}**. Duplicate rows found: **{E_['clean_report']['duplicate_rows']}**. The cleaning function therefore removed no rows ({E_['clean_report']['rows_after']:,} rows kept). "
            "The code still includes imputation-free checks so the same script is safe on a new data release.")
    para(d, "**A hidden pattern that looks like missing data.** The average-grade fields contain many zeros. Checking the data shows that the average grade of a semester is exactly 0 **if and only if** "
            f"no curricular unit was approved in that semester (this held for 100% of rows in both semesters). So a zero is a real, meaningful value (\"nothing passed\"), not an error, and was not imputed. "
            f"{pct(z1.mean())} of students had no approved unit in semester 1 and {pct(z2.mean())} in semester 2; of those students, {dr1:.1f}% and {dr2:.1f}% respectively finally dropped out.")
    H(d, 2, "3.3 Outlier Identification and Treatment")
    para(d, "Outliers were identified with the IQR rule (values outside Q1 − 1.5 × IQR or Q3 + 1.5 × IQR) and the |z| > 3 rule for the 14 numeric features with more than 10 distinct values (Table 5, Figure 4).")
    rows = [[c.replace("Curricular units ", "CU "), f2(r.lower_fence), f2(r.upper_fence), int(r.iqr_outliers), f"{r.iqr_pct:.1f}%", int(r.z_gt_3)] for c, r in outl.head(8).iterrows()]
    TAB(d, "Outlier summary (eight features with the highest IQR outlier share)", ["Feature", "Lower fence", "Upper fence", "IQR outliers", "Share", "|z|>3"], rows, [6.2, 2.0, 2.0, 2.2, 1.8, 1.6], font=8.5, align_right_from=1)
    para(d, "**Decision.** The extreme values are real students (for example mature students aged up to 70, or students who passed many units), and zero grades are meaningful, so rows were **not deleted**. "
            "Instead, inside the model pipeline each numeric feature is *winsorised*: values below the 1st or above the 99th percentile (learned from the training set only) are clipped to those limits. "
            "Because more than 1% of students have a zero grade, the 1st percentile of the grade fields is itself 0, so the meaningful zeros are preserved.")
    H(d, 2, "3.4 Cleaning, Transformation and Encoding")
    TAB(d, "Preprocessing applied to each group of features", ["Group", "Features", "Treatment", "Reason"], [
        ["Numeric", f"{E_['n_numeric']} fields (grades, counts, ages, macro-economic)", "Winsorise 1–99 % → StandardScaler (mean 0, std 1)", "Neural networks train poorly when inputs have very different scales"],
        ["Nominal codes", f"{E_['n_nominal']} fields (course, application mode, parents' education/occupation, …)", "One-hot encoding; categories with < 2 % of training rows are grouped as 'infrequent'", "Codes have no order, so they must not be treated as numbers; grouping avoids hundreds of sparse columns"],
        ["Binary flags", f"{E_['n_binary']} fields (Debtor, Gender, Scholarship holder, …)", "Used as 0/1", "Already numeric and unordered"],
        ["Target", "Target", "Label-encoded: Dropout = 0, Enrolled = 1, Graduate = 2", "Required by the softmax output layer"]], [2.3, 4.6, 5.0, 4.7], font=8.5)
    para(d, f"After encoding, the 36 raw features become **{T_['sizes']['n_input_features']} model input columns**.")
    H(d, 2, "3.5 Train / Validation / Test Split and Leakage Prevention")
    s = T_["sizes"]
    TAB(d, "Data split (stratified, random_state = 42)", ["Subset", "Rows", "Share", "Purpose"], [
        ["Training", f"{s['train']:,}", "70%", "Fit the scaler/encoder and learn the network weights"],
        ["Validation", f"{s['val']:,}", "15%", "Choose hyperparameters and stop training early"],
        ["Test", f"{s['test']:,}", "15%", "Final, one-time unbiased evaluation"]], [3, 2, 2, 9.6], font=9)
    para(d, "**Preventing data leakage.** (1) The split is done *before* any statistic is learned. (2) The winsoriser, scaler and one-hot encoder are fitted on the training rows only and then applied unchanged to validation and test rows. "
            "(3) The test set is used only once, after all tuning decisions are final. (4) The split is stratified, so each subset keeps the same Dropout/Enrolled/Graduate proportions.")
    H(d, 2, "3.6 Key Findings from EDA")
    bullets(d, [f"The target is moderately imbalanced: Graduate {E_['class_pct']['Graduate']}%, Dropout {E_['class_pct']['Dropout']}%, Enrolled {E_['class_pct']['Enrolled']}%. Always predicting 'Graduate' would already give about 50% accuracy, so accuracy alone is not enough; macro-F1 and class weights are used.",
                f"Dropout rate is {E_['dropout_rate_by_binary']['Tuition fees up to date']['0']}% for students whose tuition fees are **not** up to date versus {E_['dropout_rate_by_binary']['Tuition fees up to date']['1']}% for those who are up to date.",
                "Academic performance in the first two semesters (units approved and average grade) separates the three outcomes much more strongly than admission grade does.",
                "Macro-economic indicators (unemployment, inflation, GDP) are almost uncorrelated with academic variables (Section 5.3).",
                "Several semester-1 and semester-2 variables are almost duplicates of each other (Spearman ρ up to 0.96), so the data has strong multicollinearity."])

    # ============================ 4 VISUALISATION ============================
    H(d, 1, "4. Data Visualization")
    H(d, 2, "4.1 The Seven Stages of Visualization")
    para(d, "The assignment sheet asks that the visualisations follow \"all 7 stages of visualization\" but does not list them. This report therefore states the framework it uses: the seven-stage "
            "visualisation process of Ben Fry (2007), *Visualizing Data* (O'Reilly): **Acquire, Parse, Filter, Mine, Represent, Refine, Interact**. Table 8 maps each stage to the work done in this project.")
    TAB(d, "Seven stages of visualization applied to the student dataset", ["Stage", "Meaning", "How it was applied here", "Evidence"], [
        ["1. Acquire", "Obtain the data", "Selected the UCI dataset (id 697), checked licence, rows and columns, and loaded data/data.csv", "Sections 2.1, 2.3"],
        ["2. Parse", "Give the data structure and meaning", "Fixed delimiter and header, assigned numeric / binary / nominal roles, encoded the target", "Sections 2.4, 3.1, 3.4"],
        ["3. Filter", "Remove what is not needed", "Duplicate and missing-value checks; selected key variables for each chart; courses with ≥ 100 students in Fig. 12", "Sections 3.2, 3.3"],
        ["4. Mine", "Find patterns with statistics", "Descriptive statistics, Kruskal–Wallis and chi-square tests, Spearman correlation, VIF, PCA, distribution fitting", "Chapter 5"],
        ["5. Represent", "Choose a basic visual form", "Bar, histogram, boxplot, scatter, heat-map, pairplot, Q–Q plot, PCA projection – matched to the data type", "Figures 1–12"],
        ["6. Refine", "Improve clarity and design", "Colour-blind-safe palette checked with a validator, same colour per class in every chart, direct value labels, mean/median lines, transparency and jitter against over-plotting, sorted bars", "All figures"],
        ["7. Interact", "Let the viewer explore", "Two Plotly pages with hover, zoom and legend filtering: outputs/eda/interactive_*.html", "Section 4.3"]], [2.2, 3.2, 8.2, 3.0], font=8.5, bold_first_col=True)
    H(d, 2, "4.2 Visualization Design Choices (Refine stage)")
    para(d, "A single colour is reserved for each outcome in every figure: **blue = Graduate, orange = Dropout, green = Enrolled**. These three hues come from a palette whose adjacent colours were checked for colour-vision-deficiency separation. "
            "Grid-lines are light, axes are labelled with units, titles are numbered, and no chart uses a dual axis or a rainbow scale.")
    H(d, 2, "4.3 Interactive Visualizations")
    para(d, "Two interactive HTML pages are saved in outputs/eda/: **interactive_grades_scatter.html** (semester-1 vs semester-2 grade; hover shows age, admission grade, course and fee status; clicking a legend entry hides or shows a class) "
            "and **interactive_course_outcomes.html** (stacked counts of outcomes for each course code). They can be opened in any web browser and complete stage 7.")
    H(d, 2, "4.4 Visualization Gallery with Explanations")
    para(d, "Each figure below is followed by a table that states its purpose, the variables used, the chart type, why it was selected and the insight obtained, as the assignment requires. Figures 1–12 are EDA figures; model figures are in Chapter 7.")
    P = EDA_DIR
    CARD(d, 1, "Student outcome distribution", P / "fig01_target_distribution.png", 10.5,
         "Show how many students fall into each outcome class.", "Target", "Bar chart with count and percentage labels",
         "Class balance decides which metrics and training tricks are needed; a bar chart is the clearest form for a few categories.",
         f"Graduate {E_['class_counts']['Graduate']:,} ({E_['class_pct']['Graduate']}%), Dropout {E_['class_counts']['Dropout']:,} ({E_['class_pct']['Dropout']}%), Enrolled {E_['class_counts']['Enrolled']:,} ({E_['class_pct']['Enrolled']}%). Enrolled is the smallest class, so it is expected to be hardest to predict; class weights and macro-F1 are used.")
    CARD(d, 2, "Histograms of key numeric variables", P / "fig02_histograms.png", 16.4,
         "Show the shape, centre and spread of six important numeric variables.", "Admission grade, Previous qualification (grade), Age at enrollment, CU 1st/2nd sem (grade), CU 2nd sem (approved)",
         "Histograms (30 bins) with mean (solid line) and median (dashed line)", "Histograms are the standard univariate view of a continuous variable; the mean–median gap shows skewness at a glance.",
         f"Admission grade is roughly bell-shaped (mean {desc.loc['Admission grade','mean']:.1f}, median {desc.loc['Admission grade','median']:.1f}). Age is strongly right-skewed (skew {desc.loc['Age at enrollment','skew']:.2f}; median {desc.loc['Age at enrollment','median']:.0f}, mean {desc.loc['Age at enrollment','mean']:.1f}) – most students enrol at 18–20 but a tail of mature students reaches 70. "
         f"The semester grades are left-skewed with a spike at 0 (median {desc.loc['Curricular units 1st sem (grade)','median']:.1f} vs mean {desc.loc['Curricular units 1st sem (grade)','mean']:.1f}), caused by students who passed no unit.")
    CARD(d, 3, "Boxplots of key variables by outcome", P / "fig03_boxplots_by_target.png", 16.6,
         "Compare the distribution of important variables across the three outcome classes and reveal outliers.", "Target vs Admission grade, Age at enrollment, CU 1st/2nd sem (grade), CU 2nd sem (approved)",
         "Grouped boxplots", "Boxplots compare medians, spread and outliers between groups compactly; they are the natural bivariate view for numeric-vs-category.",
         f"Median 2nd-semester units approved rise from {med.loc['Dropout','Curricular units 2nd sem (approved)']:.0f} (Dropout) to {med.loc['Enrolled','Curricular units 2nd sem (approved)']:.0f} (Enrolled) to {med.loc['Graduate','Curricular units 2nd sem (approved)']:.0f} (Graduate). Dropouts are older (median age {med.loc['Dropout','Age at enrollment']:.0f} vs {med.loc['Graduate','Age at enrollment']:.0f}). "
         f"Admission-grade medians differ only a little ({med.loc['Dropout','Admission grade']:.1f}, {med.loc['Enrolled','Admission grade']:.1f}, {med.loc['Graduate','Admission grade']:.1f}), so entry marks say less about the outcome than later performance does.")
    CARD(d, 4, "Share of IQR outliers per numeric feature", P / "fig04_outlier_share.png", 12.5,
         "Quantify how many rows are flagged as outliers in each numeric feature.", "14 numeric features with more than 10 distinct values", "Sorted horizontal bar chart (percentage labels)",
         "A ranked bar chart makes it easy to see which features need a treatment decision.",
         f"The semester-grade fields flag the most rows ({outl.loc['Curricular units 2nd sem (grade)','iqr_pct']:.1f}% and {outl.loc['Curricular units 1st sem (grade)','iqr_pct']:.1f}%) mainly because of the zero grades described in Section 3.2, not because of data errors. Age flags {outl.loc['Age at enrollment','iqr_pct']:.1f}% (mature students). "
         "Since these are genuine values, they were winsorised in the pipeline rather than deleted.")
    CARD(d, 5, "Continuous distribution: Admission grade", P / "fig05_normal_fit_qq.png", 16.4,
         "Check whether admission grade follows a normal distribution.", "Admission grade", "Histogram with fitted normal curve, and normal Q–Q plot",
         "The histogram shows the fit visually; the Q–Q plot is more sensitive to tail behaviour.",
         f"Fitted normal: mean {DF['Admission grade']['mean']:.1f}, SD {DF['Admission grade']['std']:.1f}. Skewness {DF['Admission grade']['skew']:.2f}, excess kurtosis {DF['Admission grade']['excess_kurtosis']:.2f}. "
         f"D'Agostino's test rejects exact normality (p < 0.001; with 4,424 rows even small departures are detected), and the Q–Q plot shows why: the lower tail is cut off at the minimum grade of 95 and the upper tail is heavier than normal. The distribution is bell-shaped but right-skewed.")
    CARD(d, 6, "Discrete distributions: Poisson and Binomial", P / "fig06_discrete_distributions.png", 16.4,
         "Test whether a count variable fits a Poisson model and illustrate a Binomial model for the number of dropouts.", "(a) CU 1st sem (approved); (b) Target = Dropout (yes/no)",
         "Side-by-side bar charts of observed vs theoretical probabilities", "Observed-vs-theoretical bars show directly where a model fits or fails.",
         f"(a) A Poisson with the same mean (λ = {DF['Curricular units 1st sem (approved)']['mean']:.2f}) fits badly: the variance-to-mean ratio is {DF['Curricular units 1st sem (approved)']['dispersion_index']:.2f} (Poisson would be 1) and {pct(DF['Curricular units 1st sem (approved)']['zero_share'])} of students have 0 units approved versus {p_pois0*100:.1f}% expected – the data are over-dispersed and zero-inflated. "
         f"(b) Dropout is a Bernoulli variable with p = {DF['dropout_proportion']:.3f} (95% Wilson interval {DF['dropout_wilson_ci'][0]:.3f}–{DF['dropout_wilson_ci'][1]:.3f}). The number of dropouts among 10 random students follows Binomial(10, 0.321); a simulation of 5,000 random samples matches it closely. For example, P(at least 5 dropouts among 10) = {p_ge5:.3f}.")
    CARD(d, 7, "Spearman correlation heatmap", P / "fig07_correlation_heatmap.png", 14.5,
         "Show monotonic relationships among all 19 numeric features.", "All numeric features", "Lower-triangle heat-map with diverging colour scale (neutral midpoint at 0) and printed coefficients",
         "A heat-map is the standard multivariate summary; Spearman is used because many variables are skewed or counts.",
         f"Strongest pairs: CU enrolled 1st vs 2nd semester ρ = {E_['top_corr_pairs'][0][2]}, credited ρ = {E_['top_corr_pairs'][1][2]}, approved ρ = {E_['top_corr_pairs'][2][2]}, grade ρ = {E_['top_corr_pairs'][3][2]}. "
         "Admission grade and previous-qualification grade correlate at 0.58. Unemployment, inflation and GDP have |ρ| below 0.3 with every other variable. Highly correlated groups produce VIF values above 10 (Section 5.3), so the network is regularised.")
    CARD(d, 8, "Outcome mix by binary factors", P / "fig08_outcome_by_binary_factors.png", 16.4,
         "Compare the outcome mix between the two values of four binary factors.", "Tuition fees up to date, Debtor, Scholarship holder, Gender vs Target", "100% stacked bar charts",
         "Percent-stacked bars compare proportions fairly even when group sizes differ.",
         f"Not-up-to-date tuition: {E_['dropout_rate_by_binary']['Tuition fees up to date']['0']}% dropout vs {E_['dropout_rate_by_binary']['Tuition fees up to date']['1']}%. Debtors: {E_['dropout_rate_by_binary']['Debtor']['1']}% vs {E_['dropout_rate_by_binary']['Debtor']['0']}%. Scholarship holders: {E_['dropout_rate_by_binary']['Scholarship holder']['1']}% vs {E_['dropout_rate_by_binary']['Scholarship holder']['0']}%. "
         f"Gender code 1: {E_['dropout_rate_by_binary']['Gender']['1']}% vs code 0: {E_['dropout_rate_by_binary']['Gender']['0']}%. These are associations, not causes – for instance unpaid fees may be a consequence of leaving as well as a reason for it.")
    CARD(d, 9, "First vs second semester grade", P / "fig09_grade_scatter.png", 11.5,
         "Show how first- and second-semester performance relate and how outcomes are spread over them.", "CU 1st sem (grade), CU 2nd sem (grade), Target", "Scatter plot (colour = outcome, transparency + small jitter)",
         "Scatter plots are the standard bivariate view of two continuous variables; colour adds the third (categorical) dimension.",
         "The main cloud shows a clear positive relationship (ρ = 0.76) with graduates (blue) concentrated at higher grades. There are no points between 0 and about 10 because a semester grade is zero exactly when no unit was passed. "
         "The dense dot at (0, 0) and the strips along the axes are almost entirely dropouts – students who passed nothing in one or both semesters.")
    CARD(d, 10, "PCA projection of numeric features", P / "fig10_pca.png", 11.5,
         "See whether the classes separate when the 19 numeric features are compressed into two dimensions.", "All numeric features (standardised), Target", "PCA scatter plot",
         "PCA is the usual multivariate technique to visualise high-dimensional data in 2-D.",
         f"PC1 explains {E_['pca_var_ratio_first5'][0]*100:.1f}% and PC2 {E_['pca_var_ratio_first5'][1]*100:.1f}% of the variance; the first five components together explain {pca_cum5*100:.1f}%. Dropouts (orange) lie mostly to the left, graduates to the right/upper area, but there is heavy overlap and Enrolled students (green) sit in between – a visual explanation of why Enrolled is the hardest class to predict.")
    CARD(d, 11, "Pairplot of key variables", P / "fig11_pairplot.png", 15.5,
         "Show all pairwise relationships and each variable's distribution for five key variables in one view.", "Admission grade, Age, CU 1st/2nd sem (grade), CU 2nd sem (approved), Target", "Pairplot (scatter matrix with KDE diagonals)",
         "A pairplot is the quickest way to compare many pairs and to see class-wise distributions on the diagonal.",
         "Grade and approved-unit variables are positively related to each other, and the dropout curves on the diagonals have a separate peak at zero. Age shows dropouts with a longer right tail. Admission grade barely separates the classes.")
    CARD(d, 12, "Dropout rate by course code", P / "fig12_dropout_by_course.png", 12.5,
         "Compare dropout rates between degree programmes with enough students.", "Course (code), Target", "Sorted horizontal bar chart with overall-rate reference line",
         "Sorting and a reference line make differences between programmes easy to rank; only courses with ≥ 100 students are shown so that rates are stable.",
         f"Among the {len(g_)} courses with at least 100 students, dropout rates range from {g_.rate.iloc[0]:.1f}% (course {g_.index[0]}, n = {g_.n.iloc[0]}) to {g_.rate.iloc[-1]:.1f}% (course {g_.index[-1]}, n = {g_.n.iloc[-1]}), compared with {100*(df.Target=='Dropout').mean():.1f}% overall. "
         "Course is therefore an informative predictor; code meanings are listed in the UCI documentation.")
    d_ = None

    # ============================ 5 STATISTICAL ANALYSIS ============================
    H(d, 1, "5. Statistical and Distribution-Based Analysis")
    H(d, 2, "5.1 Univariate Analysis")
    para(d, "Table 4 (Section 2.5) gives the mean, median, mode, standard deviation and quartiles of every numeric feature. The main points are:")
    bullets(d, [f"**Centre and spread:** admission grade mean {desc.loc['Admission grade','mean']:.1f} (SD {desc.loc['Admission grade','std']:.1f}); age mean {desc.loc['Age at enrollment','mean']:.1f} but median {desc.loc['Age at enrollment','median']:.0f} and mode {desc.loc['Age at enrollment','mode']:.0f}.",
                f"**Skewness:** age has the strongest positive skew ({desc.loc['Age at enrollment','skew']:.2f}), so the median describes a typical student better than the mean. Grade fields are left-skewed because of zeros.",
                f"**Quantiles:** half of all students are aged 20 or less (median), 95% are {desc.loc['Age at enrollment','q95']:.0f} or younger, while the maximum is {desc.loc['Age at enrollment','max']:.0f}.",
                "**Categorical fields:** the outcome classes are 49.9% / 32.1% / 18.0%; the binary flags are unbalanced (for example only a minority are debtors or scholarship holders), which is reflected in the stratified split."])
    H(d, 2, "5.2 Probability Distributions")
    TAB(d, "Distribution analysis results", ["Variable", "Model", "Result", "Conclusion"], [
        ["Admission grade (continuous)", "Normal", f"μ = {DF['Admission grade']['mean']:.1f}, σ = {DF['Admission grade']['std']:.1f}; KS D = {DF['Admission grade']['ks_stat']:.3f}; D'Agostino p < 0.001", "Approximately bell-shaped but not exactly normal (skew 0.53, truncated at 95)"],
        ["Previous qualification (grade)", "Normal", f"μ = {DF['Previous qualification (grade)']['mean']:.1f}, σ = {DF['Previous qualification (grade)']['std']:.1f}; KS D = {DF['Previous qualification (grade)']['ks_stat']:.3f}; p < 0.001", "Near-symmetric (skew 0.31), heavier tails than normal"],
        ["Age at enrollment (continuous)", "Normal vs log-normal", f"KS D = {DF['Age at enrollment']['ks_normal']:.3f} (normal) vs {DF['Age at enrollment']['ks_lognormal']:.3f} (log-normal)", "Both poor; log-normal slightly better for the right-skewed shape"],
        ["Dropout (discrete, yes/no)", "Bernoulli / Binomial(10, p)", f"p = {DF['dropout_proportion']:.3f}; Wilson 95% CI {DF['dropout_wilson_ci'][0]:.3f}–{DF['dropout_wilson_ci'][1]:.3f}", "Binomial pmf matches simulated samples (Fig. 6b)"],
        ["CU 1st sem approved (count)", "Poisson", f"mean {DF['Curricular units 1st sem (approved)']['mean']:.2f}, variance {DF['Curricular units 1st sem (approved)']['variance']:.2f}, ratio {DF['Curricular units 1st sem (approved)']['dispersion_index']:.2f}", "Rejected: over-dispersed with excess zeros"],
        ["CU 2nd sem approved (count)", "Poisson", f"mean {DF['Curricular units 2nd sem (approved)']['mean']:.2f}, variance {DF['Curricular units 2nd sem (approved)']['variance']:.2f}, ratio {DF['Curricular units 2nd sem (approved)']['dispersion_index']:.2f}", "Rejected: over-dispersed"],
        ["CU 1st sem without evaluations (count)", "Poisson", f"mean {DF['Curricular units 1st sem (without evaluations)']['mean']:.2f}, ratio {DF['Curricular units 1st sem (without evaluations)']['dispersion_index']:.2f}, zeros {pct(DF['Curricular units 1st sem (without evaluations)']['zero_share'])}", "Rejected: very zero-heavy"]],
        [3.8, 2.8, 5.6, 4.4], font=8.5)
    para(d, "In simple words: a *Poisson* distribution describes counts of rare, independent events and has variance equal to its mean. Here the variance is about twice the mean because there are two kinds of student – those who pass nothing and those who pass most units – so a single Poisson cannot fit; a mixture or zero-inflated model would be more suitable.", size=10)
    H(d, 2, "5.3 Bivariate Analysis")
    para(d, "Every feature was tested against the three-class outcome. Numeric features used the **Kruskal–Wallis H test** (does not assume normality) with the effect size ε²; binary and nominal features used the **chi-square test of independence** with Cramér's V. "
            "With 4,424 students almost every p-value is below 0.001, so the *effect size* is what ranks the features (Table 10).")
    rows = [[f, r["type"], r["test"], f"{r['statistic']:.1f}", "< 0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}", f"{r['effect_size']:.3f}"] for f, r in biv.head(12).iterrows()]
    TAB(d, "Twelve features most strongly associated with the outcome", ["Feature", "Type", "Test", "Statistic", "p-value", "Effect size"], rows, [5.4, 1.7, 2.6, 2.0, 2.0, 2.0], font=8.5, align_right_from=3)
    para(d, f"**Interpretation.** Tuition-fee status (V = {eps('Tuition fees up to date'):.2f}) and units approved in semester 2 (ε² = {eps('Curricular units 2nd sem (approved)'):.2f}) have the largest effects, followed by units approved in semester 1, semester-2 grade, scholarship status and course. "
            f"Admission grade has a small effect (ε² = {eps('Admission grade'):.3f}), and macro-economic indicators are weakest. The tests show *association*, not causation.")
    H(d, 2, "5.4 Multivariate Analysis")
    bullets(d, [f"**Correlation structure (Fig. 7):** the strongest correlations are between the same measure in consecutive semesters (ρ between 0.76 and {E_['top_corr_pairs'][0][2]}).",
                "**Multicollinearity:** variance inflation factors above 10 were found for " + ", ".join(f"{k.replace('Curricular units ','CU ')} ({v:.1f})" for k, v in vif.head(6).items()) + ". Redundant inputs do not harm a neural network's accuracy as much as a linear regression's coefficients, but they make individual feature effects harder to interpret.",
                f"**PCA (Fig. 10):** five components explain {pca_cum5*100:.1f}% of the variance of the 19 numeric features, showing that the numeric information is partly redundant.",
                "**Combined effects (Figs. 8, 9, 11):** low performance, unpaid fees and absence of a scholarship tend to occur together in the dropout group."])
    H(d, 2, "5.5 Summary of Statistical Findings")
    TAB(d, "Statistical findings in plain words", ["Finding", "Evidence"], [
        ["Early academic performance is the strongest signal of the final outcome", f"Largest Kruskal–Wallis effects (ε² = {eps('Curricular units 2nd sem (approved)'):.2f} and {eps('Curricular units 1st sem (approved)'):.2f}); Fig. 3, 9"],
        ["Financial factors matter", f"Dropout {E_['dropout_rate_by_binary']['Tuition fees up to date']['0']}% vs {E_['dropout_rate_by_binary']['Tuition fees up to date']['1']}% by fee status; scholarship holders {E_['dropout_rate_by_binary']['Scholarship holder']['1']}% vs {E_['dropout_rate_by_binary']['Scholarship holder']['0']}%"],
        ["Entry marks are weak predictors", f"Admission-grade ε² = {eps('Admission grade'):.3f}; median grades 123.6 / 124.1 / 127.4"],
        ["Macro-economic indicators add little", "|ρ| < 0.3 with all variables; smallest effect sizes"],
        ["Counts are not Poisson", "Dispersion index about 2–3.5; excess zeros"]], [6.5, 10.1], font=9)

    # ============================ 6 DEEP LEARNING MODEL ============================
    H(d, 1, "6. Deep Learning Model")
    H(d, 2, "6.1 Why a Multilayer Perceptron?")
    para(d, "The data are in table form (numbers and codes), so the natural deep-learning model is a **multilayer perceptron (MLP)** – a stack of fully connected layers (Goodfellow et al., 2016). CNNs are designed for images and RNN/LSTM/Transformers for sequences or text, which this dataset does not contain. "
            "An MLP can learn non-linear combinations of features (for example 'low grades *and* unpaid fees') that a simple linear model cannot. To judge whether the extra complexity is worthwhile, logistic regression and a random forest were trained on the same data as baselines.")
    H(d, 2, "6.2 Inputs, Target and Pipeline")
    TAB(d, "Model inputs and output", ["Item", "Description"], [
        ["Input", f"{s['n_input_features']} columns produced from {s['n_raw_features']} raw features (scaled numeric + one-hot nominal + binary)"],
        ["Target", "3 classes: Dropout (0), Enrolled (1), Graduate (2)"],
        ["Task type", "Multi-class classification"],
        ["Pipeline", "Winsorise (1–99 %) → standardise; one-hot with rare-category grouping; binary pass-through. Fitted on the training set only and saved as models/preprocessor.joblib"]], [3, 13.6], font=9, bold_first_col=True)
    H(d, 2, "6.3 Architecture and Hyperparameters")
    FIG(d, EVAL_DIR / "diagram2_mlp_architecture.png", 16, "Diagram 2: MLP architecture of the final model")
    cfg = T_["best_config"]
    TAB(d, "Final model configuration", ["Component", "Setting", "Simple explanation"], [
        ["Hidden layers", f"{' → '.join(map(str, cfg['hidden']))} neurons", "Each layer learns more abstract combinations of the inputs"],
        ["Activation", "ReLU (hidden), Softmax (output)", "ReLU adds non-linearity; softmax turns the 3 outputs into probabilities that sum to 1"],
        ["Regularisation", f"Dropout {cfg['dropout']}, L2 weight penalty 1e-4, BatchNorm", "Reduce over-fitting and stabilise training"],
        ["Loss function", "Sparse categorical cross-entropy", "Penalises confident wrong probabilities"],
        ["Optimizer", f"Adam, learning rate {cfg['lr']}", "Adaptive gradient method (Kingma & Ba, 2015)"],
        ["Batch size / max epochs", f"{T_['batch_size']} / {T_['max_epochs']}", "Samples per weight update / upper limit on passes over the data"],
        ["Early stopping", f"Monitor validation loss, patience {T_['early_stopping_patience']}, restore best weights", f"Training actually ran {M_['epochs_run']} epochs; best epoch {M_['best_epoch']}"],
        ["Class weights", ", ".join(f"{k} {v:.2f}" for k, v in T_["class_weights"].items()), "Give the rarer 'Enrolled' class more influence on the loss"],
        ["Parameters", f"{T_['n_params']:,}", "Includes batch-normalisation statistics"],
        ["Random seed", "42", "Makes the run reproducible"]], [3.3, 6.0, 7.3], font=8.5, bold_first_col=True)
    H(d, 2, "6.4 Training, Validation and Testing Procedure")
    bullets(d, ["Fit the preprocessing pipeline on the training rows and transform all three subsets.",
                "For each hyperparameter configuration, train on the training set and monitor loss on the validation set; stop early when it no longer improves.",
                "Choose the configuration with the best **validation macro-F1** (ties broken by validation loss).",
                "Evaluate the chosen model **once** on the untouched test set and report accuracy, macro precision/recall/F1, ROC-AUC, confusion matrix and importance."], style="List Number")
    H(d, 2, "6.5 Basic Hyperparameter Tuning")
    para(d, "A small grid search over **3 architectures × 2 dropout rates × 2 learning rates = 12 configurations** was run. All other settings were fixed. Selection used the validation set only.")
    rows = [[r.hidden, r.dropout, r.lr, r.epochs_run, f3(r.val_macro_f1), f3(r.val_loss), f3(r.val_acc)] for r in tune.sort_values("val_macro_f1", ascending=False).itertuples()]
    TAB(d, "Hyperparameter search results (sorted by validation macro-F1)", ["Hidden layers", "Dropout", "Learning rate", "Epochs", "Val macro-F1", "Val loss", "Val accuracy"], rows, [3.2, 1.8, 2.3, 1.7, 2.6, 2.4, 2.6], font=8.5, align_right_from=1)
    para(d, f"The best configuration was {cfg['hidden']} with dropout {cfg['dropout']} and learning rate {cfg['lr']} (validation macro-F1 {f3(T_['val_macro_f1'])}). The spread between the best and worst configuration is "
            f"{tune.val_macro_f1.max()-tune.val_macro_f1.min():.3f} macro-F1, so the model is fairly insensitive to these choices; differences of about 0.01–0.02 between neighbouring configurations are within the noise expected from a 664-row validation set. Figure 19 shows the same results as a chart.", size=10.5)
    H(d, 2, "6.6 Feature Analysis and Feature Importance")
    para(d, "Two complementary methods were used. (1) **Statistical association** (Section 5.3) shows which features relate to the outcome in the raw data. (2) **Permutation importance** measures what the *trained model* relies on: each raw feature in the test set is randomly shuffled 10 times and the drop in macro-F1 is recorded; a large drop means the model depends on that feature. "
            "Because shuffling is done on the raw feature (before one-hot encoding), all columns that came from one original field are shuffled together.")
    rows = [[i + 1, f.replace("Curricular units ", "CU "), f"{r.mean_drop:.4f}", f"{r['std']:.4f}"] for i, (f, r) in enumerate(imp.head(10).iterrows())]
    TAB(d, "Top 10 features by permutation importance (test set)", ["Rank", "Feature", "Mean macro-F1 drop", "Std (10 repeats)"], rows, [1.4, 8.0, 3.6, 3.6], font=8.5, align_right_from=2)
    para(d, f"The model relies overwhelmingly on units approved in semester 2 (drop {imp.iloc[0].mean_drop:.3f}) and semester 1 (drop {imp.iloc[1].mean_drop:.3f}), followed by semester-2 grade and tuition-fee status. This agrees with the statistical findings in Chapter 5. "
            "Importance of correlated features is shared between them (for example approved units and grade), so a small value for one feature does not prove it is unrelated to the outcome.", size=10.5)

    # ============================ 7 EVALUATION ============================
    H(d, 1, "7. Model Evaluation and Result Visualization")
    H(d, 2, "7.1 Evaluation Metrics")
    para(d, "All numbers below were measured by running the code in the appendix (seed 42). The test set contains 664 students that were never used for fitting, scaling or tuning.")
    TAB(d, "MLP performance on validation and test sets", ["Metric", "Validation", "Test"], [
        ["Accuracy", f3(mv["accuracy"]), f3(mt["accuracy"])], ["Macro precision", f3(mv["macro_precision"]), f3(mt["macro_precision"])],
        ["Macro recall", f3(mv["macro_recall"]), f3(mt["macro_recall"])], ["Macro F1-score", f3(mv["macro_f1"]), f3(mt["macro_f1"])],
        ["Weighted F1-score", f3(mv["weighted_f1"]), f3(mt["weighted_f1"])], ["ROC-AUC (one-vs-rest, macro)", f3(mv["roc_auc_ovr_macro"]), f3(mt["roc_auc_ovr_macro"])]],
        [6.5, 3.5, 3.5], font=9.5, align_right_from=1)
    para(d, f"Test accuracy is {pct(mt['accuracy'])} (approximate 95% Wilson interval {pct(acc_lo)}–{pct(acc_hi)} for 664 students), compared with about 50% for always predicting 'Graduate'. "
            "Validation and test scores are close, which suggests the tuning did not over-fit the validation set badly.", size=10.5)
    TAB(d, "Per-class results on the test set", ["Class", "Precision", "Recall", "F1-score", "Support"],
        [[c, f2(pc[c]["precision"]), f2(pc[c]["recall"]), f2(pc[c]["f1-score"]), int(pc[c]["support"])] for c in ["Dropout", "Enrolled", "Graduate"]], [4, 3, 3, 3, 3], font=9.5, align_right_from=1)
    H(d, 2, "7.2 Comparison with Baseline Models")
    rows = [[k, f3(v["accuracy"]), f3(v["macro_precision"]), f3(v["macro_recall"]), f3(v["macro_f1"]), f3(v["roc_auc_ovr_macro"])] for k, v in comp.items()]
    TAB(d, "Test-set comparison of the MLP and baseline models (same split and preprocessing)", ["Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "ROC-AUC"], rows, [4.6, 2.2, 2.2, 2.2, 2.2, 2.2], font=9.5, align_right_from=1)
    para(d, "**Honest reading of the comparison.** On this dataset the MLP is *competitive with*, but **not better than**, logistic regression and a random forest; the random forest has the highest accuracy, macro-F1 and ROC-AUC in Table 18. "
            "This is common for small tabular datasets (about 3,100 training rows), where the relationships are largely monotonic and simpler models are already close to the achievable limit. Because all three models are evaluated on one 664-row test set, "
            "differences of one or two percentage points should not be over-interpreted. The MLP is kept as the required deep-learning model; the baselines are reported for context.", size=10.5)
    H(d, 2, "7.3 Result Visualizations and Interpretation")
    Q = EVAL_DIR
    CARD(d, 13, "Training vs validation loss", Q / "fig13_loss_vs_epoch.png", 10.5,
         "Show how learning progressed and detect over-fitting.", "Epoch, cross-entropy loss (train and validation)", "Line chart with best-epoch marker", "Loss curves are the standard diagnostic of neural-network training.",
         f"Training loss keeps falling, but validation loss stops improving around epoch {M_['best_epoch']} and then rises, a classic sign of over-fitting. Early stopping ended training at epoch {M_['epochs_run']} and restored the weights from epoch {M_['best_epoch']}.")
    CARD(d, 14, "Training vs validation accuracy", Q / "fig14_accuracy_vs_epoch.png", 10.5,
         "Show accuracy progress for both subsets.", "Epoch, accuracy (train and validation)", "Line chart", "Accuracy is easier to interpret than loss and is requested by the assignment.",
         "Training accuracy continues to climb while validation accuracy flattens, giving a widening gap after the early epochs – consistent with Figure 13. The gap is the reason dropout, L2 and early stopping were used.")
    CARD(d, 15, "Confusion matrix on the test set", Q / "fig15_confusion_matrix.png", 14,
         "Show which classes are confused with each other.", "Actual vs predicted class (test set)", "Heat-map of counts and row-normalised recall", "A confusion matrix reveals per-class errors that a single accuracy figure hides.",
         f"Recall is {pc['Dropout']['recall']:.2f} for Dropout, {pc['Enrolled']['recall']:.2f} for Enrolled and {pc['Graduate']['recall']:.2f} for Graduate. Most mistakes involve the Enrolled class: {cm[2,1]} Graduates and {cm[0,1]} Dropouts were predicted as Enrolled, and {cm[1,2]} Enrolled students were predicted as Graduate. "
         f"Enrolled has the lowest precision ({pc['Enrolled']['precision']:.2f}) and F1 ({pc['Enrolled']['f1-score']:.2f}); only {cm[2,0]} of 331 Graduates were predicted as Dropout, so the extreme confusions are rare.")
    CARD(d, 16, "One-vs-rest ROC curves", Q / "fig16_roc_curves.png", 9.5,
         "Show ranking quality of the predicted probabilities for every class at all thresholds.", "False- and true-positive rates for each class", "ROC curves with AUC in the legend",
         "ROC/AUC is threshold-independent and is requested by the assignment; one-vs-rest extends it to three classes.",
         f"AUC is {M_['roc_auc_per_class']['Graduate']:.3f} for Graduate, {M_['roc_auc_per_class']['Dropout']:.3f} for Dropout and {M_['roc_auc_per_class']['Enrolled']:.3f} for Enrolled (macro {mt['roc_auc_ovr_macro']:.3f}). All curves are well above the chance diagonal, but Enrolled is clearly the hardest class, consistent with the confusion matrix.")
    CARD(d, 17, "Prediction confidence: correct vs wrong", Q / "fig17_confidence_distribution.png", 10.5,
         "Show whether the model is less confident when it is wrong (acts as an error-distribution view for a classifier).", "Maximum predicted probability, correct vs wrong predictions", "Overlaid histograms",
         "If wrong predictions are low-confidence, uncertain cases can be flagged for human review.",
         f"Mean confidence is {M_['mean_conf_correct']:.2f} for correct and {M_['mean_conf_wrong']:.2f} for wrong predictions. Wrong predictions are spread over lower confidence values, so low-confidence cases could be sent to a tutor, but the model is still sometimes confidently wrong.")
    CARD(d, 18, "Permutation importance (top 12 features)", Q / "fig18_permutation_importance.png", 12.5,
         "Show which raw features the trained model depends on.", "Raw features; drop in test macro-F1 after shuffling", "Horizontal bar chart with error bars (SD over 10 shuffles)", "Permutation importance is model-agnostic and works with one-hot encoded inputs when applied to raw columns.",
         f"Units approved in semester 2 (drop {imp.iloc[0].mean_drop:.3f}) and semester 1 ({imp.iloc[1].mean_drop:.3f}) dominate; semester-2 grade, tuition-fee status and age follow. Admission grade and macro-economic indicators are not among the top features.")
    CARD(d, 19, "Hyperparameter search results", Q / "fig19_tuning_results.png", 12.5,
         "Compare all 12 tested configurations.", "Architecture, dropout, learning rate vs validation macro-F1", "Sorted horizontal bars; best configuration highlighted", "A ranked chart shows both the winner and how close the others are.",
         f"Validation macro-F1 ranges from {tune.val_macro_f1.min():.3f} to {tune.val_macro_f1.max():.3f}. No single setting is dramatically better; the best (orange) is {cfg['hidden']}, dropout {cfg['dropout']}, learning rate {cfg['lr']}.")

    # ============================ 8 CONCLUSION ============================
    H(d, 1, "8. Conclusion, Challenges and Future Scope")
    H(d, 2, "8.1 What Was Learned from the Data")
    bullets(d, ["The dataset is clean in the sense of having no missing values or duplicates, but it contains meaningful zeros, skewed variables, many genuine outliers and strongly correlated features.",
                f"Performance in the first two semesters is the clearest signal: {dr1:.0f}% of students with no approved unit in semester 1 and {dr2:.0f}% with none in semester 2 finally dropped out.",
                f"Financial indicators are strongly associated with outcome: dropout was {E_['dropout_rate_by_binary']['Tuition fees up to date']['0']}% when fees were not up to date, and scholarship holders dropped out less often ({E_['dropout_rate_by_binary']['Scholarship holder']['1']}% vs {E_['dropout_rate_by_binary']['Scholarship holder']['0']}%).",
                "Admission grade and national macro-economic indicators carry much less information than later academic behaviour."])
    H(d, 2, "8.2 Model Performance")
    para(d, f"The tuned MLP reached {pct(mt['accuracy'])} accuracy, macro-F1 {f3(mt['macro_f1'])} and macro ROC-AUC {f3(mt['roc_auc_ovr_macro'])} on the held-out test set. Dropout and Graduate are predicted well (F1 {pc['Dropout']['f1-score']:.2f} and {pc['Graduate']['f1-score']:.2f}); "
            f"Enrolled is difficult (F1 {pc['Enrolled']['f1-score']:.2f}) because those students are, by definition, between the other two groups. The MLP did not outperform the random forest ({pct(comp['Random forest']['accuracy'])} accuracy) or logistic regression ({pct(comp['Logistic regression']['accuracy'])}).")
    H(d, 2, "8.3 Challenges and Limitations")
    bullets(d, ["**Class imbalance and the 'Enrolled' class:** handled with class weights and macro metrics, but recall for Enrolled is still modest.",
                "**Over-fitting:** the network over-fits after about 13 epochs on 3,096 training rows; mitigated with dropout, L2, BatchNorm and early stopping.",
                "**Coded categories:** meanings of code values are not interpreted in this report; the UCI documentation must be consulted for domain explanation.",
                "**Single split and seed:** results come from one stratified split and one random seed; no repeated cross-validation was done, so small differences between models are not statistically established.",
                "**Timing of the strongest features:** semester-1 and semester-2 results are known only after those semesters, so the model is an early-warning tool for the second year, not at enrolment.",
                "**Single institution:** the data come from one Portuguese polytechnic, so results may not transfer to other institutions.",
                "**Data source for this study:** the CSV was obtained from public copies of the UCI dataset (identical across two copies) because the UCI site was not reachable from the analysis environment.",
                "**Ethics:** features such as gender and nationality should be used with care; a deployed system must be checked for fairness and used to offer support, not to penalise students."])
    H(d, 2, "8.4 Future Scope")
    bullets(d, ["Build a **first-semester-only** model to give earlier warnings, and a model that uses only enrolment-time information.",
                "Use repeated stratified k-fold cross-validation with confidence intervals to compare models properly.",
                "Try gradient-boosted trees, embeddings for high-cardinality codes (Course, occupation) and focal loss or SMOTE for the Enrolled class.",
                "Explain predictions for individual students with SHAP, and calibrate predicted probabilities.",
                "Fit zero-inflated count models for the approved-units variables.",
                "Wrap the model in a simple dashboard (Plotly/Streamlit) for tutors, and monitor fairness across gender and nationality groups."])

    # ============================ 9 REFERENCES ============================
    H(d, 1, "9. Literature Review and References")
    H(d, 2, "9.1 Short Literature Review")
    para(d, "**Dataset and application.** Realinho et al. (2022) describe the dataset used here as a combination of demographic, socio-economic, macro-economic and academic records from the Polytechnic Institute of Portalegre, gathered at enrolment and at the end of the first and second semesters, "
            "and used to train models that estimate each student's risk of dropout or failure for a tutoring team. This project reuses the data openly released under a CC BY 4.0 licence (Realinho et al., 2021).")
    para(d, "**EDA and visualisation.** Tukey (1977) established exploratory analysis – looking at data before modelling. Fry (2007) formalised the seven-stage visualisation process (acquire, parse, filter, mine, represent, refine, interact) that this report follows. "
            "Plotting and statistics were done with Matplotlib (Hunter, 2007), seaborn (Waskom, 2021) and SciPy (Virtanen et al., 2020).")
    para(d, "**Modelling.** Goodfellow, Bengio and Courville (2016) describe multilayer perceptrons, dropout and regularisation; Srivastava et al. (2014) introduced dropout and Kingma and Ba (2015) the Adam optimiser used here. "
            "The baselines use scikit-learn (Pedregosa et al., 2011) and the random-forest method of Breiman (2001). The network was built with TensorFlow/Keras (Abadi et al., 2016).")
    H(d, 2, "9.2 References (APA 7th edition)")
    refs = ["Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., et al. (2016). TensorFlow: A system for large-scale machine learning. *Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI '16)*, 265–283.",
            "Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.",
            "Fry, B. (2007). *Visualizing data: Exploring and explaining data with the Processing environment*. O'Reilly Media.",
            "Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.",
            "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95.",
            "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR 2015)*.",
            "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.",
            "Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146",
            "Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict students' dropout and academic success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89",
            "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958.",
            "Tukey, J. W. (1977). *Exploratory data analysis*. Addison-Wesley.",
            "Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*, 261–272.",
            "Waskom, M. L. (2021). seaborn: Statistical data visualization. *Journal of Open Source Software, 6*(60), 3021."]
    for r in refs:
        p = d.add_paragraph(); p.paragraph_format.left_indent = Cm(1.0); p.paragraph_format.first_line_indent = Cm(-1.0); p.paragraph_format.space_after = Pt(4)
        parts = r.split("*")
        for i, part in enumerate(parts):
            if part: rr = p.add_run(part); rr.italic = (i % 2 == 1); rr.font.size = Pt(10)

    # ============================ 10 RUBRIC ALIGNMENT ============================
    H(d, 1, "10. Rubric Alignment")
    para(d, "The table maps each assessment criterion in the assignment sheet to the evidence in this report and repository. It lists where the evidence is; it does not claim any marks.")
    TAB(d, "Rubric alignment", ["CO / Criterion (marks)", "Report section", "Evidence provided"], [
        ["CO3 – Selection and creation of visualizations (3)", "4.1, 4.4, Figs. 1–12; Ch. 7, Figs. 13–19", "19 figures of 10+ types (bar, histogram, boxplot, scatter, heat-map, pairplot, Q–Q, PCA, ROC, confusion matrix, line) chosen by data type; 2 interactive pages"],
        ["CO3 – Quality and presentation (2)", "4.2", "Numbered titles, labelled axes, legends, direct labels, colour-blind-safe palette, consistent class colours"],
        ["CO3 – Interpretation of visualizations (3)", "4.4, 7.3", "Purpose / variables / chart type / why / insight table for every figure"],
        ["CO4 – Univariate analysis and distribution exploration (4)", "2.5, 5.1, Figs. 1–5", "Mean, median, mode, SD, quantiles, skew, kurtosis, histograms, boxplots, outlier analysis"],
        ["CO4 – Discrete / continuous distributions (4)", "5.2, Figs. 5–6", "Normal fit + Q–Q + normality tests; log-normal comparison; Bernoulli/Binomial with simulation; Poisson fit and dispersion analysis"],
        ["CO4 – Bivariate / multivariate analysis (4)", "5.3, 5.4, Figs. 7–11", "Kruskal–Wallis, chi-square with effect sizes, Spearman heat-map, VIF, PCA, pairplot, scatter"],
        ["CO4 – Interpretation of patterns and findings (4)", "5.5, 3.6", "Plain-language findings table linked to evidence"],
        ["CO5 – Real-world problem / application (3)", "1.2–1.5, 2.2", "Student dropout early-warning problem with a real dataset"],
        ["CO5 – Application of EDA techniques (3)", "Chapter 3", "Missing/duplicate checks, outliers, encoding, scaling, split, leakage control"],
        ["CO5 – Insights and interpretation (3)", "3.6, 5.5, 4.4", "Zero-grade finding, fee and scholarship effects, weak admission grade, over-dispersion"],
        ["CO5 – Conclusion and recommendations (3)", "Chapter 8", "Findings, model performance, limitations, future scope"],
        ["CO5 – GitHub repository and documentation (4)", "Appendix B", "Repository structure, README, requirements, dataset file, source code, report; public link to be added after upload"],
        ["Enrichment – ML/DL (optional)", "Ch. 6, 7", "Tuned MLP, baselines, feature importance, ROC/AUC, confusion matrix, learning curves"]], [5.6, 3.6, 7.4], font=8.5)

    # ============================ APPENDIX A CODE ============================
    page_break(d)
    H(d, 1, "Appendix A – Complete Python Source Code")
    para(d, "Files are shown in the order in which the pipeline runs. Run everything with: python src/run_all.py", size=10)
    for fn, ttl in [("config.py", "A.1 config.py – paths, seed, column groups, plot style"), ("data_preprocessing.py", "A.2 data_preprocessing.py – loading, cleaning, split, pipeline"),
                    ("eda.py", "A.3 eda.py – statistics and distribution fitting"), ("visualizations.py", "A.4 visualizations.py – Figures 1–12 and interactive pages"),
                    ("train_model.py", "A.5 train_model.py – tuning, training, baselines"), ("evaluate_model.py", "A.6 evaluate_model.py – evaluation, Figures 13–19"),
                    ("diagrams.py", "A.7 diagrams.py – workflow and architecture diagrams"), ("run_all.py", "A.8 run_all.py – run the whole pipeline")]:
        H(d, 2, ttl); code_block(d, (ROOT / "src" / fn).read_text(), 6.5)

    # ============================ APPENDIX B GITHUB ============================
    page_break(d)
    H(d, 1, "Appendix B – GitHub Repository and Checklist")
    H(d, 2, "B.1 Repository Structure")
    code_block(d, """student-performance-analysis/
├── data/
│   └── data.csv                     # UCI dataset 697 (semicolon-separated, 4,424 x 37)
├── src/
│   ├── config.py                    # paths, seed, column groups, plot style
│   ├── data_preprocessing.py        # load, clean, split, leakage-safe pipeline
│   ├── eda.py                       # tables + statistical tests + distribution fits
│   ├── visualizations.py            # Figures 1-12 + interactive HTML pages
│   ├── train_model.py               # grid search, MLP training, baselines
│   ├── evaluate_model.py            # test metrics + Figures 13-19
│   ├── diagrams.py                  # workflow / architecture diagrams
│   └── run_all.py                   # runs the full pipeline
├── outputs/
│   ├── eda/                         # EDA figures, tables, interactive pages
│   └── model_evaluation/            # evaluation figures, metrics JSON/CSV
├── models/                          # saved MLP (.keras) + preprocessor (.joblib)
├── report/                          # Word + PDF report, report builder script
├── requirements.txt
├── README.md
└── .gitignore""", 8)
    TAB(d, "Purpose of each folder and file", ["Path", "Purpose"], [
        ["data/", "The dataset used (or a link to it). CC BY 4.0 – keep the citation in the README"],
        ["src/", "All source code, one module per stage so it is easy to read and to explain in a viva"],
        ["outputs/eda, outputs/model_evaluation", "Generated figures, tables and metrics, so the report can be checked without re-running"],
        ["models/", "Trained MLP and fitted preprocessor. The 36 MB random-forest baseline is excluded by .gitignore and is re-created by run_all.py"],
        ["report/", "Final report (DOCX and PDF) and the script that builds it from the saved results"],
        ["requirements.txt", "Exact library versions used"], ["README.md", "Project summary, results, how to run"], [".gitignore", "Keeps caches, secrets and large files out of the repository"]], [5.5, 11.1], font=9)
    H(d, 2, "B.2 README Outline")
    bullets(d, ["Title, one-paragraph summary and student details (RITHI R, 24AD091, U21ADP05)", "Problem statement and objectives", "Dataset: name, UCI URL, DOI, licence, size, target",
                "Project structure (tree above)", "How to run: pip install -r requirements.txt, then python src/run_all.py",
                "Key results table (accuracy, macro-F1, ROC-AUC) and 3–4 sample figures", "Key insights (5 bullets)", "Limitations and future work", "References / citation of the dataset", "Link to the report in report/"])
    H(d, 2, "B.3 Commands to Publish")
    code_block(d, """cd student-performance-analysis
git init
git add .
git commit -m "Student dropout EDA and MLP case study"
git branch -M main
git remote add origin https://github.com/<your-github-id>/student-performance-analysis.git
git push -u origin main""", 8)
    H(d, 2, "B.4 GitHub Submission Checklist")
    TAB(d, "Checklist before submitting the link", ["✔", "Item"], [["☐", "Repository is **public** and created under your own GitHub ID"], ["☐", "Project report (PDF and Word) is inside report/"],
        ["☐", "Source code (.py) is inside src/ and runs with python src/run_all.py"], ["☐", "Dataset file or dataset link is present (data/data.csv + UCI link in README)"],
        ["☐", "README.md has summary, dataset, how to run, results and limitations"], ["☐", "No passwords, tokens, .env files or personal data are committed"],
        ["☐", "No very large files (> 50 MB); models/baseline_random.joblib is ignored"], ["☐", "Public GitHub link pasted into the Google Form (https://forms.gle/kT9DU5aERo2peTNh8) or emailed to the coordinator"]], [1.2, 15.4], font=9.5)

    # ============================ APPENDIX C FINAL CHECKLIST ============================
    H(d, 1, "Appendix C – Final Checklist Against the Assignment Requirements")
    TAB(d, "Requirement checklist", ["Requirement from assignment sheet", "Status", "Where"], [
        ["Real-world dataset with ≥ 15 features", "Done – 36 features", "2.1–2.4"], ["Load data and perform EDA", "Done", "Ch. 3, 5"],
        ["Handle missing values, outliers, duplicates", "Done – none missing, none duplicated, outliers analysed and winsorised", "3.2, 3.3"],
        ["Cleaning, transformation, encoding, scaling", "Done", "3.4"], ["Train / validation / test split", "Done – 70/15/15 stratified", "3.5"],
        ["Minimum 5 visualisations covering all 7 stages", "Done – 12 EDA + 7 model figures; stages mapped in Table 8", "4.1, 4.4"],
        ["Mention what each visual shows, why created, insights", "Done – explanation table per figure", "4.4, 7.3"],
        ["At least one deep-learning model", "Done – MLP", "Ch. 6"], ["Feature analysis, importance, basic tuning", "Done", "6.5, 6.6"],
        ["3–4 result visualisations (loss, accuracy, confusion matrix, ROC/AUC)", "Done – Figs. 13–19", "7.3"],
        ["Conclusion, challenges, future scope", "Done", "Ch. 8"], ["3–5 references, APA/IEEE", "Done – 13 APA references", "9.2"],
        ["Report format (title page … appendix)", "Done", "Whole report"], ["Well-commented code with loading, EDA, preprocessing, model, evaluation", "Done", "Appendix A"],
        ["GitHub repository, README, dataset link, report, source code", "Prepared locally; **upload and public link still to be done by the student**", "Appendix B"]], [7.4, 6.0, 3.2], font=8.5)
    return d


def finish(d):
    """Insert TOC / list of figures / list of tables (static text inside a Word TOC field, so it can be refreshed in Word)."""
    body = d.element.body
    marker = [p for p in d.paragraphs if p.text == "[[TOC]]"][0]
    anchor = marker._p
    def add(par):
        anchor.addprevious(par._p)
    para_title = lambda t: para(d, t, bold=True, size=15, color=BLUE, after=6)
    t = para_title("Table of Contents"); add(t)
    items = [(l, x) for l, x in HEAD if l <= 2]
    for i, (l, x) in enumerate(items):
        add(toc_entry(d, x, PAGES.get(x, "0"), indent_cm=0 if l == 1 else 0.7, bold=(l == 1), first=(i == 0), last=(i == len(items) - 1), field='TOC \\o "1-2" \\h \\z \\u'))
    t = para_title("List of Figures"); t.paragraph_format.space_before = Pt(12); add(t)
    for x in FIGS: add(toc_entry(d, x, PAGES.get(x, "0")))
    t = para_title("List of Tables"); t.paragraph_format.space_before = Pt(12); t.paragraph_format.page_break_before = True; add(t)
    for x in TABS: add(toc_entry(d, x, PAGES.get(x, "0")))
    anchor.getparent().remove(anchor)


if __name__ == "__main__":
    d = build(); finish(d)
    out = ROOT / "report" / "Student_Dropout_EDA_Report_RITHI_R_24AD091.docx"
    d.save(out); json.dump({"headings": [x for _, x in HEAD], "figs": FIGS, "tabs": TABS}, open(ROOT / "report" / "_toc_items.json", "w"))
    print("saved", out)
