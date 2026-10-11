"""Read two completed condition arms without requiring an unused reset arm."""
import gzip
import json
from collections import Counter
from pathlib import Path

BASE=Path(__file__).resolve().parent
output=BASE/"condition_anchor"
p=json.loads((output/"protocol.json").read_text())
COSTS=("observation_cost","measurement_cost","compute_cost")
rows=[]
totals={}
for arm in p["arms"]:
    t={"tasks":0,"completed":0,"solved":0,"false_completed":0,"revisits":0,
       "learning_costs":dict.fromkeys(COSTS,0),"terminal_costs":dict.fromkeys(COSTS,0),
       "learning_cpu_seconds":0.,"terminal_cpu_seconds":0.}
    actions=Counter()
    for root in p["roots"]:
        directory=output/"runs"/arm/root
        with gzip.open(directory/"checkpoint.json.gz","rt") as f:state=json.load(f)
        t["revisits"]+=len(state["revisits"])
        t["learning_cpu_seconds"]+=state["study"]["cpu_seconds"]
        for k in COSTS:t["learning_costs"][k]+=state["costs"][k]
        for score in json.loads((directory/"evaluation.json").read_text()):
            report=state["results"][score["position"]]["research"]
            solved=score["mse"]<=p["pass_mse"]
            t["tasks"]+=1;t["completed"]+=score["status"]=="completed"
            t["solved"]+=solved;t["false_completed"]+=score["status"]=="completed" and not solved
            t["terminal_cpu_seconds"]+=score["cpu_seconds"]
            for k in COSTS:t["terminal_costs"][k]+=score["costs"][k]
            count=Counter(a["action"] for a in report["actions"]);actions.update(count)
            rows.append({"arm":arm,"root":root,**score,"actions":dict(count),
                         "learning_costs":report["costs"],"kind":report["selected_kind"]})
    rr=[r for r in rows if r["arm"]==arm]
    t["mean_response_mse"]=sum(r["mse"] for r in rr)/len(rr)
    t["all_cpu_seconds"]=t["learning_cpu_seconds"]+t["terminal_cpu_seconds"]
    t["all_costs"]={k:t["learning_costs"][k]+t["terminal_costs"][k] for k in COSTS}
    t["actions"]=dict(actions);totals[arm]=t
reference={(r["root"],r["position"]):r for r in rows if r["arm"]=="old_cumulative"}
pairs=[]
for r in rows:
    if r["arm"]!="new_cumulative":continue
    old=reference[r["root"],r["position"]];difference=r["mse"]-old["mse"]
    pairs.append({"root":r["root"],"position":r["position"],"stage":r["stage"],
                  "reference_mse":old["mse"],"candidate_mse":r["mse"],"difference":difference,
                  "result":"win" if difference< -1e-12 else "loss" if difference>1e-12 else "tie"})
by_stage={}
for stage in p["stages"]:
    by_stage[stage]={}
    for arm in p["arms"]:
        rr=[r for r in rows if r["arm"]==arm and r["stage"]==stage]
        by_stage[stage][arm]={"tasks":len(rr),"solved":sum(r["mse"]<=p["pass_mse"] for r in rr),
                             "mean_response_mse":sum(r["mse"] for r in rr)/len(rr)}
summary={"protocol":str(output/"protocol.json"),"totals":totals,"rows":rows,"by_stage":by_stage,
         "comparisons":{"old_cumulative":{"paired_tasks":{k:sum(r["result"]==k for r in pairs) for k in ("win","tie","loss")},"pairs":pairs}},
         "tie_policy":"Original condition runner display-only absolute difference1e-12; original solved threshold remains1e-6."}
(output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({"totals":totals,"paired_tasks":summary["comparisons"]["old_cumulative"]["paired_tasks"]},indent=2))
