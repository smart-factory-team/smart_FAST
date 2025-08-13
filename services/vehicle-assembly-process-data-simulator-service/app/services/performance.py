from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(results: List[dict]) -> Dict:
        """성능 지표 계산"""
        valid_results = [r for r in results if r['predicted_label'] is not None]

        if not valid_results:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}

        y_true = [r['ground_truth'] for r in valid_results]
        y_pred = [r['predicted_label'] for r in valid_results]

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_images": len(results),
            "successful_predictions": len(valid_results),
            "failed_predictions": len(results) - len(valid_results)
        }