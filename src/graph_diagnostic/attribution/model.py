import pandas as pd
import lightgbm as lgb
import shap
import logging
import json
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AttributionModel:
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
            
        self.df = pd.read_csv(self.dataset_path)
        
        # Expanded features list based on FEATURE_GUIDE.md
        self.feature_cols = [
            "node_count", "edge_count", "seed_count", 
            "density", "avg_degree", "component_count", 
            "clustering_coeff", "betweenness_mean", "diameter",
            "seed_confidence_mean", "seed_ambiguity",
            "property_fill_rate", "entity_diversity", "relation_diversity",
            "property_diversity"
        ]
        self.target_col = "f1_score"

    def train_and_explain(self, output_dir: str | Path):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter for rows where features exist (important for the transition to new extractor)
        available_features = [c for c in self.feature_cols if c in self.df.columns]
        X = self.df[available_features]
        y = self.df[self.target_col]
        
        logger.info(f"Training on {len(X)} samples with {len(available_features)} features...")
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Parameters optimized for small datasets
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'verbose': -1,
            'seed': 42,
            'min_data_in_leaf': 5 if len(X) < 100 else 20
        }
        
        logger.info("Training LightGBM model...")
        model = lgb.train(params, train_data, num_boost_round=100)
        
        logger.info("Running SHAP Analysis...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Handle SHAP output structure (array for regression)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        logger.info("\n--- Global Feature Importance (SHAP) ---")
        for idx, row in feature_importance.iterrows():
            logger.info(f"{row['feature'].ljust(20)}: {row['importance']:.4f}")
            
        # Save report
        report = {
            "dataset_size": len(X),
            "features_analyzed": available_features,
            "feature_importance": feature_importance.to_dict('records')
        }
        
        report_path = out_dir / "attribution_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"\nSaved attribution report to {report_path}")

if __name__ == "__main__":
    try:
        model = AttributionModel("experiments/runs/phase3_dataset.csv")
        model.train_and_explain("experiments/runs/reports")
    except Exception as e:
        logger.error(f"Failed to run attribution: {e}")
