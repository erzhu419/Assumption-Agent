"""Compare completed0.137 anchor against retained0.131/0.154 scores only."""
import json
from collections import Counter
from math import isclose
from pathlib import Path

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
output=BASE/"old_anchor_0137"
reference=json.loads((ROOT/"artifacts/version_direction_audit_20261001/old_capability_anchor/summary.json").read_text())
current=json.loads((output/"summary.json").read_text())
def direction(a,b):
    return "tie" if isclose(a,b,rel_tol=1e-9,abs_tol=0.) else "win" if b<a else "loss"
versions={"0.131.0":(reference,"old_cumulative"),"0.154.0":(reference,"new_cumulative"),"0.137.0":(current,"cumulative_0137")}
results={}
rows={}
for version,(summary,arm) in versions.items():
    total=dict(summary["totals"][arm])
    total["all_cpu_seconds"]=sum(total[k] for k in ("source_cpu_seconds","learning_cpu_seconds","terminal_cpu_seconds"))
    results[version]=total
    rows[version]={(r["root"],r["position"]):r for r in summary["rows"] if r["arm"]==arm}
comparisons={}
for version in ("0.131.0","0.154.0"):
    pairs=[]
    for key,r in rows["0.137.0"].items():
        ref=rows[version][key]
        a,b=ref["mean_response_mse"],r["mean_response_mse"]
        pairs.append({"root":key[0],"position":key[1],"stage":r["stage"],"reference_mean_response_mse":a,
            "candidate_mean_response_mse":b,"difference":b-a,"result":direction(a,b)})
    chains=[]
    for root in sorted({r["root"] for r in pairs}):
        select=[r for r in pairs if r["root"]==root]
        a=sum(r["reference_mean_response_mse"] for r in select)/len(select)
        b=sum(r["candidate_mean_response_mse"] for r in select)/len(select)
        chains.append({"root":root,"reference_mean_response_mse":a,"candidate_mean_response_mse":b,"result":direction(a,b)})
    a,b=results[version],results["0.137.0"]
    comparisons["0.137.0_vs_"+version]={
        "paired_tasks":{k:sum(r["result"]==k for r in pairs) for k in ("win","tie","loss")},
        "paired_chains":{k:sum(r["result"]==k for r in chains) for k in ("win","tie","loss")},
        "relative_changes":{k:b[k]/a[k]-1 for k in ("mean_response_mse","mean_mse","all_cpu_seconds")},
        "all_cost_changes":{k:b["all_costs"][k]/a["all_costs"][k]-1 for k in ("observation_cost","compute_cost")},
        "tasks":pairs,"chains":chains}
counts={}
for version,rr in rows.items():
    count=Counter()
    for row in rr.values():count.update(row["actions"])
    counts[version]=dict(count)
report={"scope":"Same previously seen0.131 task/seed/budget cohort; only0.137 arm newly learned/evaluated; no new generalization or author model causal claim",
    "numeric_equivalence":{"method":"math.isclose","rel_tol":1e-9,"abs_tol":0.,"scope":"Display-only roundoff equality; no task pass threshold"},
    "totals":results,"comparisons":comparisons,"actual_final_report_action_counts":counts}
(output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"totals":results,"comparisons":{k:{q:v for q,v in r.items() if q not in ("tasks","chains")} for k,r in comparisons.items()}},indent=2))
