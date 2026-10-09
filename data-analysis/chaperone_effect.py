import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "chaperone"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COLLABS_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"
ARTISTS_CSV = DATA_DIR / "spotify_artists_2000_2022.csv"

OUTPUT_DYADS_CSV = RESULTS_DIR / "chaperone-dyads.csv"
OUTPUT_BY_CHAPERONE_CSV = RESULTS_DIR / "chaperone-by-artist.csv"
OUTPUT_SUMMARY_JSON = RESULTS_DIR / "chaperone-summary.json"
OUTPUT_PRE_POST_PLOT = RESULTS_DIR / "chaperone-pre-post.png"
OUTPUT_STRATA_PLOT = RESULTS_DIR / "chaperone-vs-others-by-career-stage.png"

# ---- settings ----

# Chaperones = the TOP_N_CHAPERONES artists with the most collaborators.
TOP_N_CHAPERONES = 20

# For a mentee X whose first collaboration with chaperone M happened in year t,
# new links are counted in the WINDOW_YEARS years before t and after t.
# Year t itself is excluded (other artists credited on the same release as M
# would otherwise count as links "caused" by M).
WINDOW_YEARS = 3

# A mentee's later links are only fully observed if the mentee's own discography
# was read. True = only use mentees that have a row (with >= 1 album) in the
# artists CSV. See the note about ONLY_SEED_ARTIST_RELEASES in the explanation.
RESTRICT_TO_PROFILED_MENTEES = True


# ---- load the collaboration network ----

adj = defaultdict(dict)  # artist -> {partner: first year they released together}
names = {}
data_first_year = None
data_last_year = None

with COLLABS_CSV.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    if "collab_years" not in (reader.fieldnames or []):
        raise SystemExit(
            "The collaborations CSV has no collab_years column.\n"
            "Rerun build-collaborations.py (the version that writes collab_years) first."
        )

    for row in reader:
        years = [int(y) for y in row["collab_years"].split(";") if y]

        if not years:
            continue

        a, b = row["artist_1_id"], row["artist_2_id"]
        names[a] = row["artist_1_name"]
        names[b] = row["artist_2_name"]

        first = min(years)
        adj[a][b] = first
        adj[b][a] = first

        data_first_year = first if data_first_year is None else min(data_first_year, first)
        data_last_year = max(years) if data_last_year is None else max(data_last_year, max(years))

print(f"Loaded {len(adj)} artists, years {data_first_year}-{data_last_year}, "
      f"window = {WINDOW_YEARS} years")

profiled = None

if RESTRICT_TO_PROFILED_MENTEES:
    if ARTISTS_CSV.exists():
        with ARTISTS_CSV.open("r", newline="", encoding="utf-8") as file:
            profiled = {
                row["artist_id"]
                for row in csv.DictReader(file)
                if int(row.get("number_of_albums") or 0) >= 1
            }
        print(f"{len(profiled)} artists have their own discography in the artists CSV")
    else:
        print("Artists CSV not found; not restricting mentees to profiled artists.")

# ---- who are the chaperones? ----

top_chaperones = sorted(adj, key=lambda n: (-len(adj[n]), names.get(n, "")))[:TOP_N_CHAPERONES]
top_set = set(top_chaperones)

print(f"\nTop {TOP_N_CHAPERONES} chaperones (by number of collaborators):")
for i, node in enumerate(top_chaperones, start=1):
    print(f"  {i:2d}. {names[node]} ({len(adj[node])} collaborators)")


def career_stage(prior_degree):
    if prior_degree == 0:
        return "0"
    if prior_degree <= 2:
        return "1-2"
    if prior_degree <= 9:
        return "3-9"
    return "10+"


STAGES = ["0", "1-2", "3-9", "10+"]

# ---- one row per (chaperone M, mentee X) pair, for both directions of every link ----

dyads = []
dropped_window = 0
dropped_unprofiled = 0

for m, partners in adj.items():
    for x, t in partners.items():
        if t - WINDOW_YEARS < data_first_year or t + WINDOW_YEARS > data_last_year:
            dropped_window += 1
            continue

        if profiled is not None and x not in profiled:
            dropped_unprofiled += 1
            continue

        before = range(t - WINDOW_YEARS, t)
        after = range(t + 1, t + WINDOW_YEARS + 1)

        new_before = new_after = introduced = prior_degree = 0

        for z, year in adj[x].items():
            if z == m:
                continue

            if year < t:
                prior_degree += 1

            if year in before:
                new_before += 1
            elif year in after:
                new_after += 1

                # "introduced": X's new partner Z already worked with M by then
                m_to_z = adj[m].get(z)
                if m_to_z is not None and m_to_z <= year:
                    introduced += 1

        dyads.append({
            "chaperone_id": m,
            "chaperone_name": names[m],
            "mentee_id": x,
            "mentee_name": names[x],
            "first_collab_year": t,
            "chaperone_is_top": m in top_set,
            "chaperone_degree_at_year": sum(1 for y in partners.values() if y <= t),
            "mentee_prior_degree": prior_degree,
            "mentee_career_stage": career_stage(prior_degree),
            "new_links_before": new_before,
            "new_links_after": new_after,
            "gain": new_after - new_before,
            "introduced_links": introduced,
        })

print(f"\n{len(dyads)} usable chaperone-mentee pairs "
      f"({dropped_window} dropped: window outside data range, "
      f"{dropped_unprofiled} dropped: mentee has no own discography)")

if not dyads:
    raise SystemExit("No usable pairs. Check the CSV, or set RESTRICT_TO_PROFILED_MENTEES = False.")

with OUTPUT_DYADS_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(dyads[0].keys()))
    writer.writeheader()
    writer.writerows(dyads)

# ---- small statistics helpers (return None when a test is not possible) ----


def safe_wilcoxon(after_values, before_values):
    differences = [a - b for a, b in zip(after_values, before_values)]

    if len(differences) < 5 or all(d == 0 for d in differences):
        return None

    try:
        return float(wilcoxon(differences).pvalue)
    except ValueError:
        return None


def safe_mannwhitney(group_a, group_b):
    if len(group_a) < 5 or len(group_b) < 5:
        return None

    try:
        return float(mannwhitneyu(group_a, group_b, alternative="two-sided").pvalue)
    except ValueError:
        return None


def safe_spearman(x_values, y_values):
    if len(x_values) < 5 or len(set(x_values)) < 2 or len(set(y_values)) < 2:
        return None, None

    result = spearmanr(x_values, y_values)
    return float(result.statistic), float(result.pvalue)


def describe(group):
    if not group:
        return {"n": 0}

    before_values = [d["new_links_before"] for d in group]
    after_values = [d["new_links_after"] for d in group]
    gains = [d["gain"] for d in group]

    return {
        "n": len(group),
        "mean_new_links_before": mean(before_values),
        "mean_new_links_after": mean(after_values),
        "mean_gain": mean(gains),
        "median_gain": median(gains),
        "share_with_positive_gain": sum(1 for g in gains if g > 0) / len(gains),
        "wilcoxon_p_after_vs_before": safe_wilcoxon(after_values, before_values),
    }


chaperone_dyads = [d for d in dyads if d["chaperone_is_top"]]
other_dyads = [d for d in dyads if not d["chaperone_is_top"]]

summary = {
    "settings": {
        "top_n_chaperones": TOP_N_CHAPERONES,
        "window_years": WINDOW_YEARS,
        "restricted_to_profiled_mentees": profiled is not None,
        "data_years": [data_first_year, data_last_year],
    },
    "pairs_used": len(dyads),
    "pairs_dropped_window": dropped_window,
    "pairs_dropped_unprofiled_mentee": dropped_unprofiled,
}

# RQ A: do mentees of top chaperones gain new links after the collaboration?
summary["A_top_chaperones_before_vs_after"] = describe(chaperone_dyads)

# robustness: one row per mentee (their earliest collaboration with any top chaperone),
# because the same mentee can appear under several chaperones
first_by_mentee = {}
for d in chaperone_dyads:
    current = first_by_mentee.get(d["mentee_id"])
    if current is None or d["first_collab_year"] < current["first_collab_year"]:
        first_by_mentee[d["mentee_id"]] = d

summary["A_one_row_per_mentee"] = describe(list(first_by_mentee.values()))

# RQ B: compared with mentees of less connected artists, matched on career stage
summary["B_top_chaperones_vs_others"] = {
    "overall": {
        "top_mean_gain": mean(d["gain"] for d in chaperone_dyads) if chaperone_dyads else None,
        "others_mean_gain": mean(d["gain"] for d in other_dyads) if other_dyads else None,
        "mann_whitney_p": safe_mannwhitney(
            [d["gain"] for d in chaperone_dyads], [d["gain"] for d in other_dyads]
        ),
    },
    "by_mentee_career_stage": {},
}

for stage in STAGES:
    top_gains = [d["gain"] for d in chaperone_dyads if d["mentee_career_stage"] == stage]
    other_gains = [d["gain"] for d in other_dyads if d["mentee_career_stage"] == stage]

    summary["B_top_chaperones_vs_others"]["by_mentee_career_stage"][stage] = {
        "n_top": len(top_gains),
        "n_others": len(other_gains),
        "top_mean_gain": mean(top_gains) if top_gains else None,
        "others_mean_gain": mean(other_gains) if other_gains else None,
        "mann_whitney_p": safe_mannwhitney(top_gains, other_gains),
    }

# RQ C: dose-response. Does a more connected chaperone (at the time) mean a bigger gain?
rho, p_value = safe_spearman(
    [d["chaperone_degree_at_year"] for d in dyads], [d["gain"] for d in dyads]
)
summary["C_chaperone_prominence_vs_gain"] = {
    "spearman_all_pairs": rho,
    "spearman_p_all_pairs": p_value,
    "by_mentee_career_stage": {},
}

for stage in STAGES:
    group = [d for d in dyads if d["mentee_career_stage"] == stage]
    rho, p_value = safe_spearman(
        [d["chaperone_degree_at_year"] for d in group], [d["gain"] for d in group]
    )
    summary["C_chaperone_prominence_vs_gain"]["by_mentee_career_stage"][stage] = {
        "n": len(group), "spearman": rho, "p_value": p_value,
    }

# RQ D: are the new links to the chaperone's own circle ("introductions")?
total_after_top = sum(d["new_links_after"] for d in chaperone_dyads)
total_after_other = sum(d["new_links_after"] for d in other_dyads)

summary["D_introductions"] = {
    "top_share_of_new_links_that_were_introduced": (
        sum(d["introduced_links"] for d in chaperone_dyads) / total_after_top
        if total_after_top else None
    ),
    "others_share_of_new_links_that_were_introduced": (
        sum(d["introduced_links"] for d in other_dyads) / total_after_other
        if total_after_other else None
    ),
}

# ---- per-chaperone table ----

by_chaperone_rows = []

for m in top_chaperones:
    group = [d for d in chaperone_dyads if d["chaperone_id"] == m]
    stats = describe(group)
    after_total = sum(d["new_links_after"] for d in group)

    by_chaperone_rows.append({
        "chaperone_id": m,
        "chaperone_name": names[m],
        "collaborators_total": len(adj[m]),
        "mentees_analyzed": stats["n"],
        "mean_new_links_before": stats.get("mean_new_links_before"),
        "mean_new_links_after": stats.get("mean_new_links_after"),
        "mean_gain": stats.get("mean_gain"),
        "wilcoxon_p_after_vs_before": stats.get("wilcoxon_p_after_vs_before"),
        "share_of_new_links_introduced": (
            sum(d["introduced_links"] for d in group) / after_total if after_total else None
        ),
    })

with OUTPUT_BY_CHAPERONE_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(by_chaperone_rows[0].keys()))
    writer.writeheader()
    writer.writerows(by_chaperone_rows)

with OUTPUT_SUMMARY_JSON.open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

# ---- plots ----

plot_rows = sorted(
    (r for r in by_chaperone_rows if r["mentees_analyzed"]),
    key=lambda r: r["mean_gain"],
)

if plot_rows:
    positions = range(len(plot_rows))

    plt.figure(figsize=(10, 8))
    plt.barh([p - 0.2 for p in positions], [r["mean_new_links_before"] for r in plot_rows],
             height=0.4, label=f"{WINDOW_YEARS} years before")
    plt.barh([p + 0.2 for p in positions], [r["mean_new_links_after"] for r in plot_rows],
             height=0.4, label=f"{WINDOW_YEARS} years after")
    plt.yticks(list(positions), [f"{r['chaperone_name']} (n={r['mentees_analyzed']})" for r in plot_rows])
    plt.xlabel("Mean number of new collaborators of the mentee")
    plt.title("New collaborators of mentees before vs after working with each top chaperone")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PRE_POST_PLOT, dpi=150)
    plt.close()

stage_top = []
stage_other = []
for stage in STAGES:
    entry = summary["B_top_chaperones_vs_others"]["by_mentee_career_stage"][stage]
    stage_top.append(entry["top_mean_gain"] or 0)
    stage_other.append(entry["others_mean_gain"] or 0)

plt.figure(figsize=(9, 6))
x_positions = range(len(STAGES))
plt.bar([p - 0.2 for p in x_positions], stage_top, width=0.4, label=f"Top {TOP_N_CHAPERONES} chaperones")
plt.bar([p + 0.2 for p in x_positions], stage_other, width=0.4, label="All other artists")

for p, stage in zip(x_positions, STAGES):
    entry = summary["B_top_chaperones_vs_others"]["by_mentee_career_stage"][stage]
    plt.text(p - 0.2, 0, f"n={entry['n_top']}", ha="center", va="bottom", fontsize=8)
    plt.text(p + 0.2, 0, f"n={entry['n_others']}", ha="center", va="bottom", fontsize=8)

plt.xticks(list(x_positions), STAGES)
plt.xlabel("Mentee's number of collaborators before the collaboration")
plt.ylabel("Mean gain in new collaborators (after - before)")
plt.title("Gain in new collaborators: mentees of top chaperones vs mentees of other artists")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_STRATA_PLOT, dpi=150)
plt.close()
