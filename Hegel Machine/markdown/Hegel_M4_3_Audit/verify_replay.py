"""Compare committed and newly generated M4.3 evidence without concealing numeric differences."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--committed',type=Path,required=True)
p.add_argument('--replay',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def walkdiff(x,y,path=''):
    if isinstance(x,dict) and isinstance(y,dict):
        out=[]
        for k in sorted(x.keys()|y.keys()):
            if k not in x or k not in y:out.append({'path':path+'/'+str(k),'committed':x.get(k),'replay':y.get(k)})
            else:out.extend(walkdiff(x[k],y[k],path+'/'+str(k)))
        return out
    if isinstance(x,list) and isinstance(y,list):
        if len(x)!=len(y):return [{'path':path,'committed_length':len(x),'replay_length':len(y)}]
        return [d for i,(xx,yy) in enumerate(zip(x,y)) for d in walkdiff(xx,yy,path+'/'+str(i))]
    if x==y:return []
    d={'path':path,'committed':x,'replay':y}
    if isinstance(x,(float,int)) and isinstance(y,(float,int)):d['numeric_delta']=y-x
    return [d]

hash_checks={}
for label,root in [('committed',a.committed),('replay',a.replay)]:
    bad=[];count=0
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        if not line.strip():continue
        expected,name=line.split(maxsplit=1);name=name.lstrip('*');count+=1
        if not (root/name).is_file() or sha(root/name)!=expected:bad.append(name)
    hash_checks[label]={'count':count,'mismatches':bad}

orig_files={x.relative_to(a.committed).as_posix():x for x in a.committed.rglob('*') if x.is_file() and '__pycache__' not in x.parts}
new_files={x.relative_to(a.replay).as_posix():x for x in a.replay.rglob('*') if x.is_file() and '__pycache__' not in x.parts}
filediff=[];jsondiff={}
for name in sorted(orig_files.keys()|new_files.keys()):
    if name not in orig_files or name not in new_files:
        filediff.append({'path':name,'missing_in':'committed' if name not in orig_files else 'replay'});continue
    if sha(orig_files[name])!=sha(new_files[name]):
        filediff.append({'path':name,'committed_sha256':sha(orig_files[name]),'replay_sha256':sha(new_files[name])})
        if name.endswith('.json'):
            jsondiff[name]=walkdiff(json.loads(orig_files[name].read_text()),json.loads(new_files[name].read_text()))
        elif name.endswith('.jsonl'):
            jsondiff[name]=walkdiff([json.loads(l) for l in orig_files[name].read_text().splitlines()], [json.loads(l) for l in new_files[name].read_text().splitlines()])

rows=[json.loads(l) for l in (a.replay/'part2_m4_3_execution_receipts.jsonl').read_text().splitlines()]
summary=json.loads((a.replay/'part2_m4_3_channel_bound_continual_summary.json').read_text())
old_summary=json.loads((a.committed/'part2_m4_3_channel_bound_continual_summary.json').read_text())
unique={(r['arm'],r['seed'],r['target_id']) for r in rows}
arm={}
ledger_bad=[]
for r in rows:
    c=r['cost']; phases=c['phase_breakdown']
    if abs(sum(p['meas'] for p in phases.values())-c['measurement_cost'])>1e-5 or abs(sum(p['compute'] for p in phases.values())-c['compute_cost'])>1e-4 or abs(c['measurement_cost']+.05*c['compute_cost']-c['composite_cost'])>.000051:
        ledger_bad.append([r['arm'],r['seed'],r['target_id']])
for name in sorted({r['arm'] for r in rows}):
    inside=[r for r in rows if r['arm']==name and not r['is_out_of_scope']]
    outside=[r for r in rows if r['arm']==name and r['is_out_of_scope']]
    arm[name]={'in_scope':len(inside),'strict_pass':sum(r['strict_pass'] for r in inside),'false_confirmation':sum(r['false_confirmation'] for r in inside),
       'oos':len(outside),'oos_unknown':sum(r['status']=='scoped_unknown' for r in outside),
       'max_overlap':max(r['prefix_confirm_overlap_count'] for r in inside+outside),
       'total_C':round(sum(r['cost']['composite_cost'] for r in inside),4),
       'fine_direct_selected':sum(r['selected_node_id']=='ode_micro_2mode_node' for r in inside)}
result={'python':platform.python_version(),'sha_checks':hash_checks,'committed_file_count':len(orig_files),'replay_file_count':len(new_files),
        'byte_identical_files':len(orig_files.keys()&new_files.keys())-sum('missing_in' not in d for d in filediff),
        'file_differences':filediff,'structured_differences':jsondiff,
        'receipt_count':len(rows),'unique_receipt_keys':len(unique),'ledger_internal_consistency_mismatches':ledger_bad,
        'stage3_arm_summaries_exact':summary['stage3_arms']==old_summary['stage3_arms'],
        'lifecycle_breakeven_exact':summary['lifecycle_breakeven']==old_summary['lifecycle_breakeven'],
        'fork_comparison_exact':summary['fork_noiseless_vs_instrument_noise']==old_summary['fork_noiseless_vs_instrument_noise'],
        'arm_recalculation':arm}
a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in {'structured_differences','file_differences'}},ensure_ascii=False,indent=2))
print('file_differences',len(filediff),'structured_diff_counts',{k:len(v) for k,v in jsondiff.items()})
