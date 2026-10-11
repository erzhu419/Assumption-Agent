"""Compare the two fixed retention cohorts after all new terminal scores exist."""
import json
from collections import Counter
from math import isclose
from pathlib import Path

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
def read(path):return json.loads(path.read_text())
old=read(ROOT/"artifacts/version_direction_audit_20261001/old_capability_anchor/summary.json")
v137=read(BASE/"old_anchor_0137/summary.json")
repair=read(BASE/"old_anchor_repaired/summary.json")
condition=read(BASE/"condition_anchor/summary.json")
historical_condition=read(ROOT/"artifacts/condition_response_actions_20260930/fresh_replication/summary.json")

def summarize(arms,metric,tie_absolute=None,historical_cpu_references=()):
    totals={};rows={};counts={}
    for version,(summary,arm) in arms.items():
        total=dict(summary["totals"][arm]);total["all_cpu_seconds"]=sum(total.get(k,0.) for k in ("source_cpu_seconds","learning_cpu_seconds","terminal_cpu_seconds"))
        totals[version]=total;rows[version]={(r["root"],r["position"]):r for r in summary["rows"] if r["arm"]==arm}
        count=Counter()
        for row in rows[version].values():count.update(row["actions"])
        counts[version]=dict(count)
    comparisons={}
    for reference in arms:
        if reference=="0.155.0":continue
        pairs=[]
        for key,new in rows["0.155.0"].items():
            previous=rows[reference][key];a,b=previous[metric],new[metric]
            equal=isclose(a,b,rel_tol=1e-9,abs_tol=0.) if tie_absolute is None else abs(b-a)<=tie_absolute
            pairs.append({"root":key[0],"position":key[1],"stage":new["stage"],"reference_mse":a,"candidate_mse":b,
                          "difference":b-a,"result":"tie" if equal else "win" if b<a else "loss"})
        chains=[]
        for root in sorted({r["root"] for r in pairs}):
            rr=[r for r in pairs if r["root"]==root];a=sum(r["reference_mse"] for r in rr)/len(rr);b=sum(r["candidate_mse"] for r in rr)/len(rr)
            equal=isclose(a,b,rel_tol=1e-9,abs_tol=0.) if tie_absolute is None else abs(b-a)<=tie_absolute
            chains.append({"root":root,"reference_mse":a,"candidate_mse":b,"result":"tie" if equal else "win" if b<a else "loss"})
        a,b=totals[reference],totals["0.155.0"]
        comparisons[reference]={"paired_tasks":{k:sum(r["result"]==k for r in pairs) for k in ("win","tie","loss")},
            "paired_chains":{k:sum(r["result"]==k for r in chains) for k in ("win","tie","loss")},
            "relative_changes":{k:b[k]/a[k]-1 if a[k] else None for k in ("mean_response_mse","all_cpu_seconds")},
            "all_cost_changes":{k:b["all_costs"][k]/a["all_costs"][k]-1 if a["all_costs"][k] else None for k in ("observation_cost","measurement_cost","compute_cost")},
            "tasks":pairs,"chains":chains}
        if reference in historical_cpu_references:
            comparisons[reference]["relative_changes"].pop("all_cpu_seconds")
            comparisons[reference]["historical_cpu_reference"] = "Retained raw history only; no present-machine CPU ratio is reported."
    return {"totals":totals,"comparisons":comparisons,"actual_final_report_action_counts":counts}

result={"scope":"Two fixed previously seen task cohorts; offline retention diagnostic, no fresh-generalization or vanilla comparison",
        "old_noisy_anchor":summarize({"0.131.0":(old,"old_cumulative"),"0.137.0":(v137,"cumulative_0137"),"0.154.0":(old,"new_cumulative"),"0.155.0":(repair,"new_cumulative")},"mean_response_mse"),
        "condition_composition_anchor":summarize({"0.137.0":(historical_condition,"new_cumulative"),"0.154.0":(condition,"old_cumulative"),"0.155.0":(condition,"new_cumulative")},"mse",1e-12,("0.137.0",)),
        "tie_policy":{"noisy":"Display-only math.isclose rel1e-9 abs0; no pass threshold", "condition":"Original runner display-only absolute1e-12; solved threshold remains original1e-6"},
        "execution_parallelism":{"old_noisy_references":"Each earlier arm dispatcher used4workers", "condition_baseline":"4workers", "candidate_learning":"A and B ran concurrently,4workers each,8total", "cpu_interpretation":"Raw process CPU is retained, but parallel workloads differ; no causal runtime-speed improvement is claimed."},
        "comparison_cost_scope":"Each arm includes its full research/revisit and terminal costs. Prior completed summaries reused without additional scoring; source0.137 condition CPU is historical and not a current CPU comparison."}
(BASE/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
for name in ("old_noisy_anchor","condition_composition_anchor"):
    print(name)
    for version,t in result[name]["totals"].items():print(version,{k:t.get(k) for k in ("tasks","completed","solved","false_completed","mean_response_mse","all_costs","all_cpu_seconds")})
    print({k:{q:v for q,v in r.items() if q not in ("tasks","chains")} for k,r in result[name]["comparisons"].items()})
