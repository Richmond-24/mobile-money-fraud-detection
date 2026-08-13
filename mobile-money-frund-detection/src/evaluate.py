from __future__ import annotations

import json
from pathlib import Path

from src.train import train_model


def write_evaluation_report():
    model, metrics = train_model()
    report = {
        "model": model.__class__.__name__,
        "metrics": metrics,
    }
    report_path = Path(__file__).resolve().parents[1] / "reports" / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    write_evaluation_report()
