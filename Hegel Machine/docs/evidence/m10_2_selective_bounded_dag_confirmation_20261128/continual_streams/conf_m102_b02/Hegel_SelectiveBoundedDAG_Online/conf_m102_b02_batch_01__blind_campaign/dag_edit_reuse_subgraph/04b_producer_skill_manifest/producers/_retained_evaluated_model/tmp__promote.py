import subprocess,os,glob,shutil,json
p='/holdout_task/node_00.py'; test='/holdout_task/test_dag_candidate.py'
paths=['/tmp/node00_baseline.py','/tmp/node00_nested_result.py']+sorted(glob.glob('/tmp/node00_panel_*.py'))
best=None
for f in paths:
 if f.endswith('nested_result.py') and open(f,'rb').read()==open(paths[0],'rb').read():continue
 shutil.copyfile(f,p)
 try:
  r=subprocess.run(['python3','-I',test],capture_output=True,text=True,timeout=14)
  print(f,r.stdout.strip(),r.stderr[-350:],flush=True)
  d=json.loads(r.stdout.strip().splitlines()[-1]); m=d['cv_metrics']; q=max(m['public_cv_y_mae'],m['public_cv_traj_mae']); rank=(not d['accepted'],q)
  if best is None or rank<best[0]:best=(rank,f)
 except (ValueError,IndexError,KeyError,subprocess.TimeoutExpired) as e:print(f,str(e),flush=True)
shutil.copyfile(best[1] if best else paths[0],p)
print('Final candidate:',best,flush=True)
