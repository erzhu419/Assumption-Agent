"""Read completed fixed-cohort scores; no Agent runs or terminal rescoring."""
import gzip
import json
from collections import Counter
from math import isclose
from pathlib import Path

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
COSTS=("observation_cost","measurement_cost","compute_cost")
def read(path):return json.loads(path.read_text())
def equivalent(a,b,noise):return isclose(a,b,rel_tol=1e-9,abs_tol=0.) if noise else abs(a-b)<=1e-12

def collect(name):
    out=BASE/name;protocol=read(out/"protocol.json");rows=[];actions=Counter();repeat_records=[]
    t={"tasks":0,"completed":0,"solved":None if protocol["pass_mse"] is None else 0,
       "false_completed":None if protocol["pass_mse"] is None else 0,"revisits":0,
       "learning_cpu_seconds":0.,"terminal_cpu_seconds":0.,
       "learning_costs":dict.fromkeys(COSTS,0),"terminal_costs":dict.fromkeys(COSTS,0)}
    part={"first_visit_costs":dict.fromkeys(COSTS,0),"followup_visit_costs":dict.fromkeys(COSTS,0),
          "screening_compute_cost":0,"visit_bases":Counter(),"visit_costs_by_basis":{},"tasks":[]}
    for root in protocol["roots"]:
        directory=out/"runs/new_cumulative"/root
        with gzip.open(directory/"checkpoint.json.gz","rt") as f:state=json.load(f)
        t["revisits"]+=len(state["revisits"]);t["learning_cpu_seconds"]+=state["study"]["cpu_seconds"]
        part["screening_compute_cost"]+=state.get("screening_compute_cost",0)
        visit_by_task={}
        for visit in state["revisits"]:
            basis=visit.get("basis","legacy_unspecified");part["visit_bases"][basis]+=1
            by_basis=part["visit_costs_by_basis"].setdefault(basis,dict.fromkeys(COSTS,0))
            by_task=visit_by_task.setdefault(visit["task_position"],dict.fromkeys(COSTS,0))
            for k in COSTS:
                fee=visit["visit_costs"][k];part["followup_visit_costs"][k]+=fee;by_basis[k]+=fee;by_task[k]+=fee
        for k in COSTS:t["learning_costs"][k]+=state["costs"][k]
        for score in read(directory/"evaluation.json"):
            report=state["results"][score["position"]]["research"]
            t["tasks"]+=1;t["completed"]+=score["status"]=="completed"
            if protocol["pass_mse"] is not None:
                solved=score["mse"]<=protocol["pass_mse"]
                t["solved"]+=solved;t["false_completed"]+=score["status"]=="completed" and not solved
            t["terminal_cpu_seconds"]+=score["cpu_seconds"]
            for k in COSTS:t["terminal_costs"][k]+=score["costs"][k]
            count=Counter(a["action"] for a in report["actions"]);actions.update(count)
            events=report["actions"]
            repeated=[(i,e) for i,e in enumerate(events) if e["action"]=="test_mechanism_predictions" and e.get("repeat_probe_id")]
            if repeated:
                purchases=[e for e in events if e["action"]=="acquire" and e.get("purpose")=="repeat_mechanism_predictions"]
                later=events[repeated[-1][0]+1:]
                confirmations=[e for e in later if e["action"]=="validate_frozen_program"]
                final_checks=[e for e in confirmations if e.get("program")==report["program"]]
                outcomes=Counter()
                for _,event in repeated:
                    for hypothesis in event["evidence"]["hypotheses"]:
                        outcomes[hypothesis["status"]]+=1
                repeat_records.append({"root":root,"position":score["position"],"repeated_tests":len(repeated),
                    "repeat_purchases":len(purchases),"repeat_observation_cost":sum(e["observation_cost"] for e in purchases),
                    "hypothesis_statuses":dict(outcomes),
                    "later_confirmation_purchases":sum(e["action"]=="acquire" and e.get("purpose")=="confirm_frozen_program" for e in later),
                    "later_confirmation_assessments":len(confirmations),"later_confirmation_acceptances":sum(e.get("accepted",False) for e in confirmations),
                    "final_program_has_later_accepted_confirmation":bool(final_checks and final_checks[-1].get("accepted") and report.get("confirmation_loss") is not None),
                    "final_status":report["status"],"final_confirmation_loss":report.get("confirmation_loss")})
            rows.append({"arm":"new_cumulative","root":root,**score,"actions":dict(count),
                         "learning_costs":report["costs"],"kind":report["selected_kind"]})
            visits=visit_by_task.get(score["position"],dict.fromkeys(COSTS,0))
            first={k:report["costs"][k]-visits[k] for k in COSTS}
            for k in COSTS:part["first_visit_costs"][k]+=first[k]
            part["tasks"].append({"root":root,"position":score["position"],"first_visit_costs":first,"followup_visit_costs":visits})
    t["mean_mse"]=sum(r["mse"] for r in rows)/len(rows)
    t["mean_response_mse"]=sum(r.get("mean_response_mse") if r.get("mean_response_mse") is not None else r["mse"] for r in rows)/len(rows)
    t["all_cpu_seconds"]=t["learning_cpu_seconds"]+t["terminal_cpu_seconds"]
    t["all_costs"]={k:t["learning_costs"][k]+t["terminal_costs"][k] for k in COSTS};t["actions"]=dict(actions)
    for k in COSTS:
        assert t["learning_costs"][k]==part["first_visit_costs"][k]+part["followup_visit_costs"][k]+(part["screening_compute_cost"] if k=="compute_cost" else 0)
    summary={"protocol":str(out/"protocol.json"),"totals":{"new_cumulative":t},"rows":rows,"paid_cost_decomposition":part,
             "actual_mechanism_repeats_and_later_confirmation":repeat_records}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def compare(candidate,references,noise):
    versions={version:(read(ROOT/path),arm) for version,(path,arm) in references.items()}
    versions["0.156.0"]=(candidate,"new_cumulative");totals={};rows={};counts={}
    for version,(summary,arm) in versions.items():
        total=dict(summary["totals"][arm]);total["all_cpu_seconds"]=sum(total.get(k,0.) for k in ("source_cpu_seconds","learning_cpu_seconds","terminal_cpu_seconds"))
        totals[version]=total;rows[version]={(r["root"],r["position"]):r for r in summary["rows"] if r["arm"]==arm}
        count=Counter()
        for r in rows[version].values():count.update(r["actions"])
        counts[version]=dict(count)
    comparisons={};metric="mean_response_mse" if noise else "mse"
    for version in references:
        pairs=[]
        for key,new in rows["0.156.0"].items():
            old=rows[version][key];a,b=old[metric],new[metric]
            pairs.append({"root":key[0],"position":key[1],"stage":new["stage"],"reference_mse":a,"candidate_mse":b,"difference":b-a,
                          "result":"tie" if equivalent(a,b,noise) else "win" if b<a else "loss"})
        chains=[]
        for root in sorted({r["root"] for r in pairs}):
            rr=[r for r in pairs if r["root"]==root];a=sum(r["reference_mse"] for r in rr)/len(rr);b=sum(r["candidate_mse"] for r in rr)/len(rr)
            chains.append({"root":root,"reference_mse":a,"candidate_mse":b,"result":"tie" if equivalent(a,b,noise) else "win" if b<a else "loss"})
        a,b=totals[version],totals["0.156.0"]
        relative={"mean_response_mse":b["mean_response_mse"]/a["mean_response_mse"]-1 if a["mean_response_mse"] else None}
        if noise or version!="0.137.0":relative["all_cpu_seconds"]=b["all_cpu_seconds"]/a["all_cpu_seconds"]-1
        comparisons[version]={"paired_tasks":dict(Counter(r["result"] for r in pairs)),"paired_chains":dict(Counter(r["result"] for r in chains)),
            "relative_changes":relative,"all_cost_changes":{k:b["all_costs"][k]/a["all_costs"][k]-1 if a["all_costs"][k] else None for k in COSTS},"tasks":pairs,"chains":chains}
    return {"totals":totals,"comparisons":comparisons,"actual_final_report_action_counts":counts,"candidate_paid_cost_decomposition":candidate["paid_cost_decomposition"],
            "actual_mechanism_repeats_and_later_confirmation":candidate["actual_mechanism_repeats_and_later_confirmation"]}

old_summary="artifacts/version_direction_audit_20261001/old_capability_anchor/summary.json"
repair="artifacts/research_routing_repair_20261001/"
a=collect("old_anchor_repaired");b=collect("condition_anchor")
result={"scope":"Previously seen two-cohort fixed retention diagnostic, offline, no source/threshold/budget/seed changes from outcomes",
    "old_noisy_anchor":compare(a,{"0.131.0":(old_summary,"old_cumulative"),"0.137.0":(repair+"old_anchor_0137/summary.json","cumulative_0137"),"0.154.0":(old_summary,"new_cumulative"),"0.155.0":(repair+"old_anchor_repaired/summary.json","new_cumulative")},True),
    "condition_composition_anchor":compare(b,{"0.137.0":("artifacts/condition_response_actions_20260930/fresh_replication/summary.json","new_cumulative"),"0.154.0":(repair+"condition_anchor/summary.json","old_cumulative"),"0.155.0":(repair+"condition_anchor/summary.json","new_cumulative")},False),
    "numeric_equivalence":"Display-only: noisy math.isclose rel1e-9 abs0; condition original absolute1e-12. No new pass threshold.",
    "cpu_scope":"Candidate cohorts concurrently4workers each8total, matching0.155candidate dispatch; CPU remains descriptive.0.137condition CPU is historical and not a present-machine ratio.",
    "priorities":["Research investment with complete confirmation","Knowledge contribution attribution, wrong-transfer withdrawal and stable long-term net benefit"]}
(BASE/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
for name in ("old_noisy_anchor","condition_composition_anchor"):
    print(name)
    for v,t in result[name]["totals"].items():print(v,{k:t.get(k) for k in ("completed","solved","mean_response_mse","all_costs","all_cpu_seconds")})
    print({v:{k:x for k,x in r.items() if k not in ("tasks","chains")} for v,r in result[name]["comparisons"].items()})
