import pickle,runpy,time,subprocess,sys,numpy as np
path='/holdout_task/node_00.py'
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a=runpy.run_path('/tmp/last_node.py'); sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']
base=(0,2,0.,0)
source=open('/tmp/last_node.py').read()
def install(opt):
    open(path,'w').write(source+'\nOPTIONS = '+repr(opt)+'\n')
# Start with the calibrated baseline, which outperformed the previous joint edit on both official metrics.
install(base)
x=s['train']['x']; n=len(x); cut=n-20
all_t=a['inv_a'](x,a['fit_a'](s['panel_a']))
order=np.random.default_rng(341).permutation(cut)
splits=[(np.delete(order,np.arange(i,cut,3)),order[i::3]) for i in range(3)]
def score(opt,tr,va):
    q={**s,'train':{k:s['train'][k][tr] for k in ('x','y','traj')},'test':{'x':x[va]}}
    z=a['predict'](q,opt,all_t[np.r_[tr,va]])
    p=sink({'__input__':q,'node_00':{'z_tr':z[:len(tr)],'z_te':z[len(tr):]}})
    return [np.mean(abs(p[k]-s['train'][j][va])) for k,j in [('y_hat','y'),('state_pred','traj')]]
opts=[base]+[(m,q,w,0) for w in (0.,.45,1.1,2.4) for m in range(6) for q in (0,1,2) if (m,q,w,0)!=base]
start=time.monotonic(); rows=[]
def try_opt(opt):
    e=np.array([score(opt,tr,va) for tr,va in splits]); worst=e.max(0)
    val=np.mean(e[:,0])+.25*worst[0]+1.2*np.mean(e[:,1])+.3*worst[1]
    rows.append((val,opt,worst))
for opt in opts:
    if time.monotonic()-start>85: break
    try_opt(opt)
for _,opt,_ in sorted(rows,key=lambda r:r[0])[:7]:
    for k in (1,2,3):
        if time.monotonic()-start>112: break
        try_opt((*opt[:3],k))
for row in sorted(rows,key=lambda r:r[0])[:8]: print('inner selection:',row,flush=True)
_,opt,_=min(rows,key=lambda r:r[0]); install(opt)
cmd=[sys.executable,'-I','/holdout_task/test_dag_candidate.py']
# Only replace a known failing result if the challenger passes the stated contract;
# otherwise leave the best verified baseline by the official error limits.
r=subprocess.run(cmd,capture_output=True,text=True,timeout=22)
print('selected',opt,r.stdout,r.stderr,flush=True)
if r.returncode!=0:
    import json
    try:
        metrics=json.loads(r.stdout)['cv_metrics']
        better=metrics['public_cv_y_mae']<=.223302 and metrics['public_cv_traj_mae']<=.077979
    except (ValueError,KeyError):
        better=False
    if not better:
        install(base)
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=22)
        print('baseline restored',r.stdout,r.stderr,flush=True)
raise SystemExit(r.returncode)
