"""Run the whole pipeline end-to-end:  python src/run_all.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import eda, visualizations, train_model, evaluate_model

if __name__ == "__main__":
    eda.run()                 # tables + statistics
    visualizations.run()      # Figures 1-12 + interactive pages
    train_model.run()         # preprocessing, tuning, training, baselines
    evaluate_model.run()      # test metrics + Figures 13-19
