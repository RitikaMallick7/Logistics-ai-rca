import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "tms_shipments.csv")


def generate_dataset(n_rows: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    carriers = ["Carrier_A", "Carrier_B", "Carrier_C", "Carrier_D"]
    regions = ["North", "South", "East", "West"]
    modes = ["Truck", "Air", "Ocean"]
    shipment_types = ["Express", "Standard", "Cold Chain"]

    data = {
        "shipment_id": [f"SHP-{i:04d}" for i in range(1, n_rows + 1)],
        "carrier": rng.choice(carriers, size=n_rows),
        "region": rng.choice(regions, size=n_rows),
        "mode": rng.choice(modes, size=n_rows),
        "shipment_type": rng.choice(shipment_types, size=n_rows),
        "distance_km": rng.normal(750, 250, n_rows).round(1),
        "weight_tons": rng.normal(6.5, 2.5, n_rows).clip(1, 20).round(2),
        "priority": rng.choice(["Low", "Medium", "High", "Critical"], p=[0.2, 0.3, 0.35, 0.15], size=n_rows),
        "weather_delay": rng.poisson(1.5, n_rows),
        "customs_delay": rng.poisson(0.6, n_rows),
        "dock_congestion": rng.poisson(1.2, n_rows),
        "driver_shortage": rng.poisson(0.8, n_rows),
        "late_pickup": rng.poisson(0.7, n_rows),
        "handling_error": rng.poisson(0.4, n_rows),
        "on_time": np.zeros(n_rows, dtype=int),
    }

    df = pd.DataFrame(data)
    df["service_level"] = rng.choice(["Gold", "Silver", "Bronze"], p=[0.5, 0.35, 0.15], size=n_rows)

    # Create an outcome that simulates late shipments with operational causes
    late_prob = (
        0.08
        + 0.03 * (df["weather_delay"] > 0)
        + 0.04 * (df["customs_delay"] > 0)
        + 0.05 * (df["dock_congestion"] > 0)
        + 0.04 * (df["driver_shortage"] > 0)
        + 0.035 * (df["late_pickup"] > 0)
        + 0.03 * (df["handling_error"] > 0)
        + 0.02 * (df["priority"].isin(["High", "Critical"]))
        + 0.015 * (df["mode"] == "Air")
    )
    lateness = rng.random(n_rows) < late_prob
    df["on_time"] = (~lateness).astype(int)

    # Add a few synthetic risk clusters for more realistic RCA patterns
    high_risk_idx = rng.choice(df.index, size=max(80, int(n_rows * 0.08)), replace=False)
    df.loc[high_risk_idx, "dock_congestion"] += rng.integers(2, 5, size=len(high_risk_idx))
    df.loc[high_risk_idx, "driver_shortage"] += rng.integers(1, 4, size=len(high_risk_idx))
    df.loc[high_risk_idx, "weather_delay"] += rng.integers(1, 3, size=len(high_risk_idx))
    df.loc[high_risk_idx, "on_time"] = 0

    df["delay_hours"] = (
        df["weather_delay"] * 3
        + df["customs_delay"] * 4
        + df["dock_congestion"] * 2.5
        + df["driver_shortage"] * 3.5
        + df["late_pickup"] * 2
        + df["handling_error"] * 1.5
    ).round(1)

    return df


if __name__ == "__main__":
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(df)} shipments at {DATA_PATH}")
