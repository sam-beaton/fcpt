"""
Shared functions for hybrid-null SPATIOTEMPORAL contrast TFCE pipelines
========================================================================

Supports four paired-window contrasts:
  - Habituation:           hab1 (1-5)  vs hab3 (11-15)
  - Novelty:               hab3 (11-15) vs nov (16-20)
  - Delayed habituation:   hab1 (1-5)  vs nov (16-20)
  - Long-range habituation: hab1 (1-5)  vs post (21-25)

Each contrast uses the same machinery:
  - Welch t per timepoint between the two trial groups (vectorised)
  - Spatiotemporal TFCE over the (parcel × timepoint) graph, using the
    parcel adjacency for spatial connectivity and ±1 sample temporal
    connectivity. Connected components found via scipy.sparse.csgraph.
  - Max-stat FWER correction via combined null:
      (a) random circular shift per trial (preserves within-trial autocorr)
      (b) random relabel of trial-group membership (preserves H0 exchangeability)

The same shift offsets and label assignment are applied across all parcels
in a given permutation, preserving spatial dependence so the max-stat
distribution is properly calibrated.

Convention: every contrast computes (expected-larger group) − (expected-
smaller group) so positive t corresponds to the hypothesised direction.
"""

import hashlib
import warnings
from itertools import combinations
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix, eye as sparse_eye, kron, diags as sparse_diags
from scipy.sparse.csgraph import connected_components as sp_connected_components


# =============================================================================
# CORE STATISTICS
# =============================================================================

def compute_welch_tstat_groups_spatiotemporal(trial_data, test_idx,
                                                group_a_idx, group_b_idx):
    """
    Welch t per timepoint between two trial groups (one parcel's data).

    t[t] = (mean_a[t] − mean_b[t]) / sqrt(var_a[t]/n_a + var_b[t]/n_b)

    Sign convention is the caller's responsibility: arrange group_a / group_b
    so positive t means the hypothesised direction.

    Returns
    -------
    t_series : ndarray (n_test_samples,)
    """
    a = trial_data[group_a_idx][:, test_idx]   # (n_a, n_test)
    b = trial_data[group_b_idx][:, test_idx]   # (n_b, n_test)

    n_a = np.sum(~np.isnan(a), axis=0)
    n_b = np.sum(~np.isnan(b), axis=0)
    mean_a = np.nanmean(a, axis=0)
    mean_b = np.nanmean(b, axis=0)
    var_a = np.nanvar(a, axis=0, ddof=1)
    var_b = np.nanvar(b, axis=0, ddof=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        se = np.sqrt(var_a / n_a + var_b / n_b)
        t = (mean_a - mean_b) / se
    invalid = (n_a < 2) | (n_b < 2) | (se == 0) | ~np.isfinite(se)
    t[invalid] = np.nan
    return t


def enumerate_label_splits(n_trials, n_group_a):
    """
    All ways of choosing which trial indices form group A (no circular shift).

    Returns a list of (a_idx, b_idx) ndarray pairs, length C(n_trials, n_group_a).
    The OBSERVED split (0..n_a-1 -> A) is placed first, so the consuming loop
    includes the observed labelling among the permutations, giving the +1
    correction for free.
    """
    all_idx = np.arange(n_trials)
    observed_a = tuple(range(n_group_a))

    a_set = set(observed_a)
    b_obs = np.array([i for i in all_idx if i not in a_set])
    splits = [(np.array(observed_a), b_obs)]

    for combo in combinations(range(n_trials), n_group_a):
        if combo == observed_a:
            continue
        a_set = set(combo)
        a_arr = np.array(combo)
        b_arr = np.array([i for i in all_idx if i not in a_set])
        splits.append((a_arr, b_arr))

    return splits


# =============================================================================
# SPATIOTEMPORAL ADJACENCY (built once per task)
# =============================================================================

def build_spatiotemporal_adjacency(spatial_adj, n_time):
    """
    Sparse (n_parcels * n_time) x (n_parcels * n_time) adjacency.
    Node ordering: row-major over (parcel, timepoint), node_idx = p*n_time + t.

      - spatial: (p, t) ↔ (q, t) if spatial_adj[p, q]
      - temporal: (p, t) ↔ (p, t±1)
    """
    n_parcels = spatial_adj.shape[0]

    spatial_part = kron(
        csr_matrix(spatial_adj.astype(bool)),
        sparse_eye(n_time, dtype=bool, format='csr'),
        format='csr',
    )

    T_off = sparse_diags(
        [np.ones(n_time - 1, dtype=bool), np.ones(n_time - 1, dtype=bool)],
        offsets=[-1, 1], shape=(n_time, n_time), format='csr', dtype=bool,
    )
    temporal_part = kron(
        sparse_eye(n_parcels, dtype=bool, format='csr'),
        T_off, format='csr',
    )

    return (spatial_part + temporal_part).astype(bool).tocsr()


# =============================================================================
# SPATIOTEMPORAL TFCE (vectorised CC)
# =============================================================================

def compute_spatiotemporal_tfce_fast(t_map_2d, st_adj, e=1.0, h=2.0, dh=0.1):
    """Spatiotemporal TFCE using precomputed sparse spatiotemporal adjacency."""
    n_parcels, n_time = t_map_2d.shape
    n_nodes = n_parcels * n_time
    tfce_flat = np.zeros(n_nodes)
    t_flat = t_map_2d.ravel()
    valid = ~np.isnan(t_flat)
    if not valid.any():
        return tfce_flat.reshape(n_parcels, n_time)

    for sign in (+1, -1):
        signed_t = sign * t_flat
        max_val = np.nanmax(signed_t[valid])
        if max_val <= 0:
            continue

        thresholds = np.arange(dh, max_val + dh, dh)
        for thresh in thresholds:
            supra = (signed_t >= thresh) & valid
            if not supra.any():
                continue
            supra_idx = np.where(supra)[0]
            sub_adj = st_adj[supra_idx][:, supra_idx]
            n_comp, labels = sp_connected_components(sub_adj, directed=False)
            sizes = np.bincount(labels, minlength=n_comp)
            increments = (sizes ** e) * (thresh ** h) * dh
            tfce_flat[supra_idx] += sign * increments[labels]

    return tfce_flat.reshape(n_parcels, n_time)


# =============================================================================
# MAX-STAT SPATIOTEMPORAL TFCE PERMUTATION (HYBRID NULL)
# =============================================================================

def maxstat_tfce_hybrid(trial_data_by_parcel, adjacency, parcel_order,
                         test_idx, n_group_a, n_group_b,
                         direction='a_minus_b',
                         n_perms=None, e=1.0, h=2.0, dh=0.1,
                         rng=None, verbose=False):
    """
    EXHAUSTIVE LABEL-PERMUTATION max-stat TFCE (relabelling only, no circshift).

    n_perms and rng are accepted but IGNORED (kept so existing callers don't
    break). The number of permutations is fixed by the combinatorics:
    C(n_group_a + n_group_b, n_group_a).

    Returns
    -------
    obs_t, obs_tfce, perm_p, parcel_p_min, threshold, max_null
    (max_null has length = number of enumerated splits)
    """
    if direction not in ('a_minus_b', 'b_minus_a'):
        raise ValueError(f"Unknown direction: {direction}")

    parcels = [p for p in parcel_order if p in trial_data_by_parcel]
    if not parcels:
        return {}, {}, {}, {}, np.nan, np.array([])

    parcel_to_pos = {p: i for i, p in enumerate(parcel_order)}
    keep_idx = [parcel_to_pos[p] for p in parcels]
    adj_sub = adjacency[np.ix_(keep_idx, keep_idx)]

    data_3d = np.stack([trial_data_by_parcel[p] for p in parcels], axis=0)
    n_parcels, n_trials, n_samples = data_3d.shape

    if n_group_a + n_group_b != n_trials:
        raise ValueError(
            f"n_group_a ({n_group_a}) + n_group_b ({n_group_b}) must equal "
            f"n_trials in the data ({n_trials})"
        )

    n_test = len(range(*test_idx.indices(n_samples)))
    st_adj = build_spatiotemporal_adjacency(adj_sub, n_test)
    sign_factor = 1.0 if direction == 'a_minus_b' else -1.0

    def _signed_t_map(a_idx, b_idx):
        t_2d = np.array([
            compute_welch_tstat_groups_spatiotemporal(data_3d[i], test_idx, a_idx, b_idx)
            for i in range(n_parcels)
        ])
        return sign_factor * t_2d

    # Enumerate all label splits; observed split is first
    splits = enumerate_label_splits(n_trials, n_group_a)
    n_enumerated = len(splits)

    # Observed
    obs_a_idx, obs_b_idx = splits[0]
    obs_t_2d = _signed_t_map(obs_a_idx, obs_b_idx)
    obs_tfce_2d = compute_spatiotemporal_tfce_fast(obs_t_2d, st_adj, e=e, h=h, dh=dh)
    obs_abs = np.abs(obs_tfce_2d)
    obs_finite = ~np.isnan(obs_tfce_2d)

    obs_t    = {p: obs_t_2d[i]    for i, p in enumerate(parcels)}
    obs_tfce = {p: obs_tfce_2d[i] for i, p in enumerate(parcels)}

# Enumerate null (includes observed split as one entry)
    max_null = np.zeros(n_enumerated)
    ge_count = np.zeros_like(obs_tfce_2d, dtype=np.int64)

    # Track per-parcel SIGNED TFCE peak for every relabelling, so we can
    # compute each parcel's empirical null directional hit-rate (p0 for 3b).
    perm_signed_peak = np.full((n_enumerated, n_parcels), np.nan)

    for k, (a_idx, b_idx) in enumerate(splits):
        perm_t_2d = _signed_t_map(a_idx, b_idx)
        perm_tfce_2d = compute_spatiotemporal_tfce_fast(perm_t_2d, st_adj, e=e, h=h, dh=dh)
        perm_abs = np.abs(perm_tfce_2d)

        valid = ~np.isnan(perm_abs)
        max_null[k] = np.max(perm_abs[valid]) if valid.any() else 0.0

        cmp = obs_finite & valid & (perm_abs >= obs_abs)
        ge_count[cmp] += 1

        # signed TFCE at each parcel's peak |t| under this relabelling
        for pi in range(n_parcels):
            row_abs = perm_abs[pi]
            if np.any(~np.isnan(row_abs)):
                pk = int(np.nanargmax(row_abs))
                perm_signed_peak[k, pi] = perm_tfce_2d[pi, pk]

        if verbose and (k + 1) % 50 == 0:
            print(f"    Enumerated split {k+1}/{n_enumerated}")

    threshold = np.percentile(max_null, 95)

    # Per-(parcel, timepoint) p-values. The observed split is among the
    # enumerated ones, so ge_count >= 1 at the observed location; this carries
    # the +1 implicitly, with the full enumeration count as denominator.
    point_p_2d = np.full_like(obs_tfce_2d, np.nan, dtype=float)
    point_p_2d[obs_finite] = ge_count[obs_finite] / n_enumerated

    perm_p = {p: point_p_2d[i] for i, p in enumerate(parcels)}
    parcel_p_min = {
        p: (float(np.nanmin(point_p_2d[i])) if np.any(~np.isnan(point_p_2d[i])) else np.nan)
        for i, p in enumerate(parcels)
    }

    # Per-parcel empirical null directional hit-rate:
    # fraction of relabellings whose signed TFCE peak exceeds the 95th
    # percentile of that parcel's own signed-peak null distribution.
    # This is the honest p0 for method 3b at the group level.
    null_dir_hit_rate = {}
    for i, p in enumerate(parcels):
        col = perm_signed_peak[:, i]
        finite = ~np.isnan(col)
        if finite.sum() < 2:
            null_dir_hit_rate[p] = np.nan
            continue
        thr = np.percentile(col[finite], 95)   # positive tail
        null_dir_hit_rate[p] = float((col[finite] > thr).mean())

    return (obs_t, obs_tfce, perm_p, parcel_p_min, threshold, max_null,
            null_dir_hit_rate)


# =============================================================================
# SANITY CHECKS
# =============================================================================

def check_null_distribution(max_null):
    if len(max_null) == 0:
        return {}
    return {
        'mean':   float(np.mean(max_null)),
        'sd':     float(np.std(max_null)),
        'median': float(np.median(max_null)),
        'p95':    float(np.percentile(max_null, 95)),
        'p99':    float(np.percentile(max_null, 99)),
        'min':    float(np.min(max_null)),
        'max':    float(np.max(max_null)),
    }


# =============================================================================
# PER-INFANT PIPELINE
# =============================================================================

def run_infant_contrast_hybrid(trial_data_by_parcel, adjacency, parcel_order,
                                 stim_onset, test_start_offset, test_end_offset,
                                 n_group_a, n_group_b,
                                 direction='a_minus_b',
                                 n_perms=2500, verbose=False, rng_seed=42):
    """Per-infant spatiotemporal contrast pipeline using hybrid null."""
    rng = np.random.default_rng(rng_seed)
    parcels = list(trial_data_by_parcel.keys())
    if not parcels:
        return None

    # Enforce consistent shape across parcels
    shapes = [trial_data_by_parcel[p].shape for p in parcels]
    n_trials_counts = [s[0] for s in shapes]
    n_samples_counts = [s[1] for s in shapes]

    modal_n_trials = Counter(n_trials_counts).most_common(1)[0][0]
    modal_n_samples = Counter(n_samples_counts).most_common(1)[0][0]

    consistent_parcels = {
        p: trial_data_by_parcel[p]
        for p in parcels
        if trial_data_by_parcel[p].shape == (modal_n_trials, modal_n_samples)
    }
    n_dropped_shape = len(parcels) - len(consistent_parcels)

    clean_parcels = {
        p: trials for p, trials in consistent_parcels.items()
        if not np.isnan(trials).any()
    }
    if not clean_parcels:
        return None

    if modal_n_trials != n_group_a + n_group_b:
        warnings.warn(
            f"Modal n_trials ({modal_n_trials}) != n_group_a ({n_group_a}) + "
            f"n_group_b ({n_group_b}). Skipping infant."
        )
        return None

    # Test window
    test_start = stim_onset + test_start_offset
    test_end = modal_n_samples if test_end_offset is None else stim_onset + test_end_offset
    if test_start >= test_end:
        warnings.warn("Test window invalid. Skipping infant.")
        return None
    if test_end > modal_n_samples:
        warnings.warn(
            f"Test window extends beyond series (end={test_end}, "
            f"length={modal_n_samples}). Skipping infant."
        )
        return None
    test_idx = slice(test_start, test_end)

    if verbose:
        print(f"  Parcels after cleaning: {len(clean_parcels)} "
              f"(n_a={n_group_a}, n_b={n_group_b})")

    obs_t, obs_tfce, perm_p, parcel_p_min, threshold, max_null, null_dir_hit_rate = maxstat_tfce_hybrid(
        clean_parcels, adjacency, parcel_order,
        test_idx=test_idx,
        n_group_a=n_group_a, n_group_b=n_group_b,
        direction=direction,
        n_perms=n_perms, e=1.0, h=2.0, dh=0.1,
        rng=rng, verbose=verbose,
    )

    null_stats = check_null_distribution(max_null)
    if verbose:
        print(f"  Null distribution: {null_stats}")

    sig_mask_by_parcel = {
        p: (~np.isnan(pv)) & (pv < 0.05) for p, pv in perm_p.items()
    }
    significant_parcels = [p for p, m in sig_mask_by_parcel.items() if m.any()]

    return {
        'observed_t': obs_t,
        'observed_tfce': obs_tfce,
        'permutation_p': perm_p,
        'parcel_p_min': parcel_p_min,
        'sig_mask': sig_mask_by_parcel,
        'null_dir_hit_rate': null_dir_hit_rate,
        'fwer_threshold': threshold,
        'tfce_params': {'e': 1.0, 'h': 2.0, 'dh': 0.1},
        'null_method': 'label_permutation_exhaustive_spatiotemporal',
        'n_enumerated_permutations': int(len(max_null)),
        'min_achievable_p': (1.0 / len(max_null)) if len(max_null) else np.nan,
        'direction': direction,
        'max_null': max_null,
        'null_distribution_stats': null_stats,
        'test_window_samples': (test_start, test_end),
        'n_group_a': n_group_a,
        'n_group_b': n_group_b,
        'n_parcels_input': len(parcels),
        'n_parcels_tested': len(clean_parcels),
        'n_parcels_dropped_shape': n_dropped_shape,
        'n_significant_parcels': len(significant_parcels),
        'n_significant_pixels': int(sum(m.sum() for m in sig_mask_by_parcel.values())),
    }


# =============================================================================
# HELPERS FOR THE MAIN LOOP
# =============================================================================

def _deterministic_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def build_contrast_trials_hybrid(infant_chrom_df, tsDataset,
                                   group_a_trials, group_b_trials,
                                   min_per_group=3):
    """Unchanged from the original — groups trials into a/b for the contrast."""
    group_a_set = set(group_a_trials)
    group_b_set = set(group_b_trials)
    relevant_set = group_a_set | group_b_set

    trials_by_parcel = defaultdict(dict)
    for _, row in infant_chrom_df.iterrows():
        if row['Trial'] in relevant_set:
            trials_by_parcel[row['Parcel']][row['Trial']] = row['tsIDx']

    parcel_series = {}
    for parcel, trial_to_idx in trials_by_parcel.items():
        a_series = []
        for t in group_a_trials:
            if t not in trial_to_idx:
                continue
            ts = tsDataset[int(trial_to_idx[t])] * 1e6
            if np.isnan(ts).all():
                continue
            a_series.append(ts)

        b_series = []
        for t in group_b_trials:
            if t not in trial_to_idx:
                continue
            ts = tsDataset[int(trial_to_idx[t])] * 1e6
            if np.isnan(ts).all():
                continue
            b_series.append(ts)

        if len(a_series) >= min_per_group and len(b_series) >= min_per_group:
            parcel_series[parcel] = (a_series, b_series)

    if not parcel_series:
        return {}, 0, 0

    shape_counts = Counter(
        (len(a), len(b)) for (a, b) in parcel_series.values()
    )
    modal_n_a, modal_n_b = shape_counts.most_common(1)[0][0]

    matching = {
        parcel: (a, b)
        for parcel, (a, b) in parcel_series.items()
        if len(a) == modal_n_a and len(b) == modal_n_b
    }
    if not matching:
        return {}, 0, 0

    all_lengths = []
    for parcel, (a, b) in matching.items():
        all_series = a + b
        all_lengths.append(min(len(s) for s in all_series))
    modal_n_samples = Counter(all_lengths).most_common(1)[0][0]

    cleaned = {}
    for parcel, (a, b) in matching.items():
        all_series = a + b
        if min(len(s) for s in all_series) < modal_n_samples:
            continue
        truncated = np.array([s[:modal_n_samples] for s in all_series])
        if np.isnan(truncated).any():
            continue
        cleaned[parcel] = truncated

    if not cleaned:
        return {}, 0, 0

    return cleaned, modal_n_a, modal_n_b


def process_one_infant_chrom_hybrid(age, infant, chrom, cleaned,
                                      n_group_a, n_group_b,
                                      adjacency, parcel_order, cfg,
                                      seed_tag,
                                      group_a_label, group_b_label,
                                      direction):
    """Worker: spatiotemporal contrast for one infant×chrom (relabel-only null)."""
    seed = _deterministic_seed(seed_tag, age, infant, chrom)

    result = run_infant_contrast_hybrid(
        cleaned,
        adjacency=adjacency,
        parcel_order=parcel_order,
        stim_onset=cfg['stim_onset'],
        test_start_offset=cfg['test_start_offset'],
        test_end_offset=cfg.get('test_end_offset'),
        n_group_a=n_group_a,
        n_group_b=n_group_b,
        direction=direction,
        n_perms=cfg['n_perms'],          # ignored by exhaustive engine
        verbose=False,
        rng_seed=seed,
    )
    if result is None:
        return [], None

    test_start, test_end = result['test_window_samples']
    n_test_samples = test_end - test_start
    null_dir = result.get('null_dir_hit_rate', {})

    results_rows = []
    for parcel in result['observed_t'].keys():
        trials_arr = cleaned[parcel]

        a_means_per_trial = np.nanmean(trials_arr[:n_group_a, test_start:test_end], axis=1)
        b_means_per_trial = np.nanmean(trials_arr[n_group_a:, test_start:test_end], axis=1)

        t_vec    = result['observed_t'][parcel]
        tfce_vec = result['observed_tfce'][parcel]
        p_vec    = result['permutation_p'][parcel]
        sig_mask = result['sig_mask'][parcel]

        if np.any(~np.isnan(t_vec)):
            best_idx = int(np.nanargmax(np.abs(t_vec)))
            t_peak = float(t_vec[best_idx])
            tfce_peak = float(tfce_vec[best_idx])
            p_at_peak = float(p_vec[best_idx])
        else:
            best_idx = -1
            t_peak = tfce_peak = p_at_peak = np.nan

        p_min = result['parcel_p_min'][parcel]
        n_sig_samples = int(sig_mask.sum())
        any_sig = bool(n_sig_samples > 0)

        sig_idx = np.where(sig_mask)[0]
        sig_idx_str = ','.join(map(str, sig_idx.tolist()))

        results_rows.append({
            'Age': age,
            'Parcel': parcel,
            'Chrom': chrom,
            'Infant': infant,
            f'Mean {group_a_label}': float(np.mean(a_means_per_trial)),
            f'SD {group_a_label}':   float(np.std(a_means_per_trial, ddof=1))
                                       if n_group_a > 1 else np.nan,
            f'Mean {group_b_label}': float(np.mean(b_means_per_trial)),
            f'SD {group_b_label}':   float(np.std(b_means_per_trial, ddof=1))
                                       if n_group_b > 1 else np.nan,
            't peak': t_peak,                 # RAW signed (a_minus_b/b_minus_a applied)
            'TFCE peak': tfce_peak,           # RAW signed
            'p at peak': p_at_peak,
            'p min': p_min,
            'Peak sample idx (within test)': best_idx,
            'N sig samples': n_sig_samples,
            'Sig sample idx FDR': sig_idx_str,        # <-- NEW (temporal overlap)
            'Null dir hit rate': float(null_dir.get(parcel, np.nan)),  # <-- NEW (3b p0)
            'N test samples': n_test_samples,
            'Test start sample': test_start,          # <-- NEW (grid alignment)
            f'n_{group_a_label}': n_group_a,
            f'n_{group_b_label}': n_group_b,
            'Significant': any_sig,
        })

    diag_row = {
        'Age': age,
        'Infant': infant,
        'Chrom': chrom,
        'Contrast': seed_tag,
        'Null method': result['null_method'],
        'Direction': result['direction'],
        'N enumerated perms': result.get('n_enumerated_permutations'),
        'Min achievable p':   result.get('min_achievable_p'),
        'Null mean':              result['null_distribution_stats'].get('mean'),
        'Null sd':                result['null_distribution_stats'].get('sd'),
        'Null p95 threshold':     result['null_distribution_stats'].get('p95'),
        'Test window start sample': test_start,
        'Test window end sample':   test_end,
        f'N {group_a_label} trials': n_group_a,
        f'N {group_b_label} trials': n_group_b,
        'N parcels tested':       result['n_parcels_tested'],
        'N significant parcels':  result['n_significant_parcels'],
        'N significant (parcel,time) pixels': result['n_significant_pixels'],
    }
    return results_rows, diag_row