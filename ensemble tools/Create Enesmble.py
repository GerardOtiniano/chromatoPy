import json
import numpy as np
from pathlib import Path

def ensemble_areas(json_path, n_samples=1000, seed=None):
    """
    Build per-GDGT area ensembles from an exported integration JSON and summarize
    (mean, median, 95% CI) into:
      { "<Sample Name>": { "<GDGT Type>": { "<GDGT>": {"mean": ..., "median": ..., "ci95": [lo, hi]} } } }

    Notes
    -----
    - Uses Normal sampling around stored parameters (Amplitude, Center, Width)
      with their 1σ uncertainties. Clips amplitude >= 0, width > 0.
    - Evaluates on the saved Fit["x"] grid and integrates with np.trapz.
    - If a GDGT has multiple rows (rare), each row becomes a separate entry
      with a suffix " (1)", " (2)", etc.

    Parameters
    ----------
    json_path : str | Path
        Path to the exported JSON file.
    n_samples : int
        Number of Monte Carlo draws per GDGT.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    dict
        { "<Sample Name>": { "<GDGT Type>": { "<GDGT>": {"mean": float, "median": float, "ci95": [float, float]} } } }
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = np.random.default_rng(seed)
    sample_name = data.get("Sample Name", "Unknown Sample")

    # Identify a result bucket
    def is_bucket(d):
        return (
            isinstance(d, dict)
            and "Area" in d
            and "Retention Time" in d
            and "Model Type" in d
            and "Model Parameters" in d
            and "Fit" in d
        )

    # Walk nested dict and yield (gdgt_type, gdgt_name, bucket)
    def walk(d, path=None):
        path = [] if path is None else path
        for k, v in d.items():
            if k == "Sample Name":
                continue
            if isinstance(v, dict):
                if is_bucket(v):
                    gdgt_type = path[-1] if len(path) >= 1 else "Unknown Type"
                    gdgt_name = k
                    yield gdgt_type, gdgt_name, v
                else:
                    yield from walk(v, path + [k])

    # Gaussian on grid
    def gaussian_on_grid(x, a, mu, sig):
        sig = np.maximum(sig, 1e-12)  # guard tiny/negative
        z = (x - mu) / sig
        return a * np.exp(-0.5 * z * z)

    out = {sample_name: {}}

    for gdgt_type, gdgt_name, bucket in walk(data):
        # Pull arrays
        mp = bucket.get("Model Parameters", {})
        fit = bucket.get("Fit", {})

        amps   = mp.get("Amplitude", [])
        cens   = mp.get("Center", [])
        wids   = mp.get("Width", [])
        amps_u = mp.get("Amplitude Unc", [])
        cens_u = mp.get("Center Unc", [])
        wids_u = mp.get("Width Unc", [])
        xs     = fit.get("x", [])   # list of lists
        # y is not required for integration; we integrate the refit curve
        # ys  = fit.get("y", [])

        n_rows = max(len(amps), len(cens), len(wids), len(xs))
        if n_rows == 0:
            # nothing here
            continue

        # Ensure type bucket
        out[sample_name].setdefault(gdgt_type, {})

        # For each row (usually 1)
        name_counts = {}
        for i in range(n_rows):
            a   = float(amps[i])   if i < len(amps)   else np.nan
            c   = float(cens[i])   if i < len(cens)   else np.nan
            s   = float(wids[i])   if i < len(wids)   else np.nan
            a_u = float(amps_u[i]) if i < len(amps_u) else 0.0
            c_u = float(cens_u[i]) if i < len(cens_u) else 0.0
            s_u = float(wids_u[i]) if i < len(wids_u) else 0.0

            # Fit grid
            x_list = xs[i] if i < len(xs) else []
            x = np.asarray(x_list, dtype=float) if x_list is not None else np.array([], dtype=float)

            # Generate ensemble
            if x.size == 0 or not np.isfinite([a, c, s]).all():
                # Absent/missing: area is zero by construction
                areas = np.zeros(n_samples, dtype=float)
            else:
                # Draw
                a_draws = a + a_u * rng.standard_normal(n_samples)
                c_draws = c + c_u * rng.standard_normal(n_samples)
                s_draws = s + s_u * rng.standard_normal(n_samples)
                a_draws = np.clip(a_draws, 0.0, None)
                s_draws = np.clip(s_draws, 1e-9, None)

                # Evaluate and integrate in chunks
                areas = np.empty(n_samples, dtype=float)
                chunk = 256
                for start in range(0, n_samples, chunk):
                    stop = min(start + chunk, n_samples)
                    A  = a_draws[start:stop][:, None]
                    MU = c_draws[start:stop][:, None]
                    SG = s_draws[start:stop][:, None]
                    y_mc = A * np.exp(-0.5 * ((x[None, :] - MU) / SG) ** 2)
                    areas[start:stop] = np.trapezoid(y_mc, x, axis=1)

            # Summaries
            mean   = float(np.mean(areas))
            median = float(np.median(areas))
            lo, hi = np.percentile(areas, [2.5, 97.5])
            ci95   = [float(lo), float(hi)]

            # Handle duplicates by suffixing "(1)", "(2)", ...
            base_key = gdgt_name
            count = name_counts.get(base_key, 0) + 1
            name_counts[base_key] = count
            key = base_key if count == 1 else f"{base_key} ({count})"

            out[sample_name][gdgt_type][key] = {
                "mean": mean,
                "median": median,
                "ci95": ci95,
            }

    return out
# %%
jp = 'path/to/.json/file'
data = ensemble_areas(jp, n_samples=1000, seed=None)