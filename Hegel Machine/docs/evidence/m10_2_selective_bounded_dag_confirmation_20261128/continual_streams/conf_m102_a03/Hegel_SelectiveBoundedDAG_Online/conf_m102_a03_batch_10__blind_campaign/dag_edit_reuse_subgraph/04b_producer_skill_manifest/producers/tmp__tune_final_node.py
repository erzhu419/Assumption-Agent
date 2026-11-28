import numpy as np,pickle,sys,itertools,time
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_01 as frozen
base=open('/tmp/node00_calibrated_base.py').read()
helpers=open('/tmp/readout_features.py').read()
m={};exec(base+'\n'+helpers,m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; n=len(y); ids=np.arange(n)
z=m['_calibrate'](s['panel_a'],tr['x']); b,h=m['_branch'](s['panel_b'],z)
R,T=m['_blocks'](z,b,h); d=m['_readout_dict'](s,z,b,h)
print('n',n,'available',[(k,v.shape[1]) for k,v in d.items()]); print('panel C',[(k,np.shape(v)) for k,v in s['panel_c'].items()])
parts=list(np.array_split(ids,4)); tail=ids[max(40,n-20):]; parts.append(tail)
# Capture the frozen fitter's readout prediction, without repeatedly refitting the trajectory in this probe.
orig=frozen._fit
class ReadoutDone(Exception): pass

def score(A):
    ans=[]
    for va in parts:
        it=np.setdiff1d(ids,va)
        tx={k:np.asarray(tr[k])[it] for k in ('x','y','traj')}
        tx['summary']=np.zeros(4)
        def capture(x,Y,*args,**kwargs):
            if np.ndim(Y)==1:
                raise ReadoutDone(A[va]@orig(x,Y,*args,**kwargs))
            return orig(x,Y,*args,**kwargs)
        frozen._fit=capture
        try:
            frozen.run_node({'__input__':dict(s,train=tx,test={'x':tr['x'][va]}),'node_00':{'readout_train':A[it],'readout_test':A[va],'z':T[it],'zt':T[va]}})
        except ReadoutDone as e:
            ans.append(float(np.mean(np.abs(e.args[0]-y[va]))))
        finally:
            frozen._fit=orig
    if len(ans)!=len(parts): raise RuntimeError('frozen readout not intercepted')
    return np.array(ans)

r=score(R); print('baseline slices',r,flush=True)
rows=[(None,r)]; valid=[k for k in d if d[k].shape[1]==4]; core=[k for k in ('z','w','b','bw','bt','ba','br','h','wa','wb') if k in valid]
configs=set()
for a in core:
    for f in core:
        for g in set([a,f,'z','b','w','h']):
            for mode in ('simple','quadratic','mixed','cube','rational','harmonic'):
                if mode!='simple' and f not in (a,g,'z','b','w'): continue
                configs.add((a,f,g,mode))
for c in valid:
    if not c.startswith('c'): continue
    for k in ('z','b','w','bt'):
        for a,f,g in ((c,c,k),(c,k,k),(k,c,c),(k,k,c),(c,c,c),(k,c,k)):
            for mode in ('simple','quadratic','mixed','rational'):
                configs.add((a,f,g,mode))
start=time.monotonic()
# Prioritize direct warped and panel-transfer features, then broaden if time permits.
for j,spec in enumerate(sorted(configs,key=lambda a:(sum(k not in ('z','b','w','bt') for k in a[:3]),a))):
    if time.monotonic()-start>119: break
    A=m['_readout_form'](d,spec)
    if not np.all(np.isfinite(A)): continue
    try: r=score(A)
    except np.linalg.LinAlgError: continue
    rows.append((spec,r))

def rank(row):
    _,a=row
    return (float(max(a[:4]))*.60+float(np.mean(a[:4]))*.40, a[-1])
print('scored',len(rows),'elapsed',round(time.monotonic()-start,2),flush=True)
print('best independent slices:', sorted(rows,key=rank)[:8],flush=True)
passed=[row for row in rows if row[1][-1]<=.0379]
print('best official split:',sorted(rows,key=lambda row:row[1][-1])[:5],flush=True)
chosen,r=min(passed,key=rank) if passed else min(rows,key=lambda row:row[1][-1]); print('selected',chosen,r,flush=True)
if chosen is not None:
    body=base[:base.index('def run_node')]+helpers+base[base.index('def run_node'):]
    body=body.replace('    r,t=_blocks(z,b,h)','    _,t=_blocks(z,b,h)\n    r=_readout_form(_readout_dict(s,z,b,h), '+repr(chosen)+')')
    # Avoid fitting unused reference-panel alternatives at inference time.
    body=body.replace("    for mode,name in [('rational','br'),('warp','bw'),('sintanh','bt'),('sinatan','ba')]:\n        out[name]=_branch(s['panel_b'],z,mode)[0]\n    out.update(_panel_signals(s['panel_c'],{'z':z,'b':b,'w':W}))", "    for mode,name in [('rational','br'),('warp','bw'),('sintanh','bt'),('sinatan','ba')]:\n        if name in "+repr(chosen[:3])+":\n            out[name]=_branch(s['panel_b'],z,mode)[0]"+("\n    out.update(_panel_signals(s['panel_c'],{'z':z,'b':b,'w':W}))" if any(k.startswith('c') for k in chosen[:3]) else ''))
    if not any(k.startswith('c') for k in chosen[:3]): body=body[:body.index('def _panel_signals')]+body[body.index('def _readout_dict'):]
    open('/holdout_task/node_00.py','w').write(body)
