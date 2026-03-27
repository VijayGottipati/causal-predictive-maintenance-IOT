import numpy as np
import pandas as pd
import os
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq

FEATURE_NAMES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Temp_Diff",
    "Power",
    "Temp_Rate",
    "Wear_Stress",
]

CAUSAL_VARIABLES = FEATURE_NAMES + ["Machine failure"]


class CausalDiscoveryEngine:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.causal_graph = None
        self.causal_matrix = None

    def discover(self, data, target_col="Machine failure"):
        """
        Discover causal relationships from observational data

        Args:
            data: numpy array or DataFrame
            target_col: name of target variable (failure)

        Returns:
            Causal graph and ranked features
        """
        if isinstance(data, pd.DataFrame):
            data_matrix = data.values
        else:
            data_matrix = data

        print(f"Running PC algorithm with alpha={self.alpha}...")

        cg = pc(data_matrix, alpha=self.alpha, indep_test="fisherz")

        self.causal_graph = cg

        adj_matrix = cg.G.graph
        self.causal_matrix = adj_matrix

        print("Causal discovery complete")

        return self._analyze_causal_relationships(target_col)

    def _analyze_causal_relationships(self, target_col="Machine failure"):
        """
        Analyze the causal graph to find:
        - Direct causes of target
        - Effects of target
        - Confounders
        """
        if self.causal_matrix is None:
            return None

        target_idx = CAUSAL_VARIABLES.index(target_col)
        n_vars = len(CAUSAL_VARIABLES)

        causes = []
        effects = []

        for i in range(n_vars):
            if i == target_idx:
                continue

            edge = self.causal_matrix[i, target_idx]

            if edge == -1:
                causes.append(CAUSAL_VARIABLES[i])
            elif edge == 1:
                effects.append(CAUSAL_VARIABLES[i])

        causal_ranks = self._rank_by_causal_importance(causes)

        return {
            "direct_causes": causes,
            "effects": effects,
            "causal_ranks": causal_ranks,
            "graph": self.causal_graph,
        }

    def _rank_by_causal_importance(self, direct_causes):
        """
        Rank features by their causal proximity to failure
        Direct causes get highest rank
        """
        ranks = {}

        for i, feature in enumerate(FEATURE_NAMES):
            if feature in direct_causes:
                ranks[feature] = 1
            else:
                ranks[feature] = 3

        for feature in FEATURE_NAMES:
            if feature in direct_causes:
                ranks[feature] = 1
            else:
                ranks[feature] = 2

        return ranks

    def get_precision_map(self):
        """
        Get precision mapping based on causal importance

        Returns:
            dict: feature -> precision level
                  0 = FP32, 1 = FP16, 2 = INT8, 3 = INT4
        """
        if not hasattr(self, "_causal_ranks"):
            return {f: 2 for f in FEATURE_NAMES}

        precision_map = {}

        for feature, rank in self._causal_ranks.items():
            if rank == 1:
                precision_map[feature] = 0
            elif rank == 2:
                precision_map[feature] = 2
            else:
                precision_map[feature] = 3

        return precision_map

    def visualize_graph(self, save_path="causal_graph.png"):
        try:
            from causallearn.utils.GraphViz import GraphViz

            gv = GraphViz()
            gv.draw(self.causal_graph, save_path)
            print(f"Causal graph saved to {save_path}")
        except Exception as e:
            print(f"Could not visualize graph: {e}")


def run_causal_discovery():
    data = pd.read_csv("data/ai4i_train.csv")

    data["Temp_Diff"] = data["Process temperature"] - data["Air temperature"]
    data["Power"] = data["Rotational speed"] * data["Torque"]
    data["Temp_Rate"] = data["Temp_Diff"] / (data["Air temperature"] + 1)
    data["Wear_Stress"] = data["Tool wear"] * data["Torque"]

    data["Type"] = data["Type"].map({"L": 0, "M": 1, "H": 2})

    feature_cols = FEATURE_NAMES + ["Machine failure"]
    data_subset = data[feature_cols].dropna()

    engine = CausalDiscoveryEngine(alpha=0.05)
    results = engine.discover(data_subset)

    print("\n=== CAUSAL DISCOVERY RESULTS ===")
    print(f"Direct causes of failure: {results['direct_causes']}")
    print(f"Effects of failure: {results['effects']}")
    print(f"\nCausal ranks: {results['causal_ranks']}")

    os.makedirs("models", exist_ok=True)
    import joblib

    joblib.dump(engine, "models/causal_engine.pkl")
    print("\nCausal engine saved to models/causal_engine.pkl")

    return engine, results


if __name__ == "__main__":
    engine, results = run_causal_discovery()
