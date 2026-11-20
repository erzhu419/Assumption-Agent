"""Read-only audit of all committed Part3 receipts and archive manifest."""
from collections import Counter
import json
import os
from pathlib import Path
BASE=Path(os.environ.get('HEGEL_AUDIT_REPO','/workspace/scratch/3f14ad2da0ed/repo_m44_b965f022b/Hegel Machine'))
OUT=Path(os.environ.get('HEGEL_AUDIT_OUT','/workspace/scratch/3f14ad2da0ed/audit_m44_b965f022b/online'))
EVID=BASE/'docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120'
rows=[json.loads(line) for line in (EVID/'part3_m4_4_online_writeback_receipts.jsonl').read_text().splitlines()]
summary=json.loads((EVID/'part3_m4_4_online_writeback_stream_summary.json').read_text())
report={'commit':'b965f022b214f47a1341559f20041da6a224d66d','receipt_count':len(rows),'arms':{}}
for arm in sorted({r['arm'] for r in rows}):
 rr=[r for r in rows if r['arm']==arm]
 dd=[r for r in rr if r['stream_phase'].startswith('downstream_')]
 per_phase={}
 for phase in sorted({r['stream_phase'] for r in rr}):
  pp=[r for r in rr if r['stream_phase']==phase]
  per_phase[phase]={'count':len(pp),'strict_pass_count':sum(r['strict_pass'] for r in pp),'false_confirmation_count':sum(r['false_confirmation'] for r in pp),'heldout_rmse_max':max(r['heldout_rmse'] for r in pp),'selected_nodes':dict(Counter(r['selected_node_id'] for r in pp)),'recall_count':sum(r['confirmed_via_recall'] for r in pp),'non_direct_walk_count':sum(r['non_direct_walk_win'] for r in pp)}
 report['arms'][arm]={'count':len(rr),'strict_pass_count':sum(r['strict_pass'] for r in rr),'false_confirmation_count':sum(r['false_confirmation'] for r in rr),'max_prefix_confirm_overlap':max(r['prefix_confirm_overlap_count'] for r in rr),'downstream_count':len(dd),'downstream_recall_count':sum(r['confirmed_via_recall'] for r in dd),'downstream_non_direct_walk_count':sum(r['non_direct_walk_win'] for r in dd),'phases':per_phase,'writeback_events':summary['arms'][arm]['writeback_events'],'post_stream_retention_fields':{k:v for k,v in summary['arms'][arm].items() if k.startswith('post_stream_') or k.startswith('new_')}}
on=[r for r in rows if r['arm']=='Hegel_LiveWriteBack_PPR_PoolOn']
off=[r for r in rows if r['arm']=='Hegel_LiveWriteBack_PPR_PoolOff']
report['pool_on_off_receipts_equal_except_arm']=all({k:v for k,v in a.items() if k!='arm'}=={k:v for k,v in b.items() if k!='arm'} for a,b in zip(on,off))
files=[p for p in EVID.rglob('*') if p.is_file()]
relpaths=[p.relative_to(EVID).as_posix() for p in files]
report['archive']={'file_count':len(files),'online_artifact_paths':[p for p in relpaths if 'online_writebacks/' in p or '/online_W6' in p or '/online_W7' in p],'post_stream_snapshot_paths':[p for p in relpaths if 'snapshot' in p and ('online' in p or 'post_stream' in p)],'forest_snapshots':{}}
for p in sorted((EVID/'relocatable_forest_snapshots').glob('*forest_snapshot.json')):
 f=json.loads(p.read_text()); report['archive']['forest_snapshots'][p.name]={'node_count':len(f['nodes']),'online_nodes':[nid for nid in f['nodes'] if nid.startswith('online_')]}
report['scope_notes']=['Each of 20 stream rows is evaluated by actual final heldout RMSE in evaluate_m4_3_target_under_arm; the event rows are also prediction tasks.', 'Recall means structural program reuse plus refit and one fresh confirmation; it is not zero-fit reuse of downstream instance parameters.', 'Only S01, S02, S09 have wb_meta. Downstream targets have none.', 'post_stream_r3_all_routes_exact compares node ids only; numerical success is returned separately by R3 helper but not retained by Part3 summary.']
(OUT/'online_receipts_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True))
print(json.dumps({'receipt_count':report['receipt_count'],'arms':{a:{k:v for k,v in r.items() if k in ['count','strict_pass_count','false_confirmation_count','downstream_recall_count','downstream_non_direct_walk_count']} for a,r in report['arms'].items()},'pool_on_off_equal':report['pool_on_off_receipts_equal_except_arm'],'archive':report['archive']},ensure_ascii=False,indent=2))
