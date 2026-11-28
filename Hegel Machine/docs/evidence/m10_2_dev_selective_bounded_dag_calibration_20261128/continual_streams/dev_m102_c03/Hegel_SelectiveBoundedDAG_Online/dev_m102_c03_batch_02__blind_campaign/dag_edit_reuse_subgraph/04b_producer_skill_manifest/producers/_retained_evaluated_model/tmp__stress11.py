import os,pickle,numpy as np
b={};exec(open('/tmp/base9.py').read(),b);exec(open('/tmp/scalar11.py').read(),b)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];a=s['train']; k=np.arange(len(a['x'])); z=b['_calibrate'](s,a['x']); y=a['y']; spl=[(k<52,k>=52),(k>=20,k<20)]+[(k%4!=j,k%4==j) for j in range(4)]; configs=[]
for ks in [(0,),(1,),(2,),(3,),(4,),(5,),(1,2),(1,2,3,4),(0,1,2,3,4,5)]:
 for d in (2,3,4,5):
  for g in (.2,.7,1.5):
   for r in (.00002,.0004,.004,.02):configs.append(('poly',ks,d,g,r))
 for typ in ('rbf','matern','plus'):
  for g in (.025,.065,.15,.35,.7):
   for r in (.00002,.0004,.004,.02):configs.append((typ,ks,3,g,r))
for ks in [(1,),(2,),(4,),(1,2),(1,2,4),(0,1,2,3,4,5)]:
 for typ in ('tensor','evidence','loo'):
  for d in (3,4,5):
   for r in (-10,-7):configs.append((typ,ks,d,.65,r))
Ps=[]; L=[]; baseline=[]
for i,j in spl:
 ss=dict(s,train={t:a[t][i] for t in ('x','y','traj')}, test={'x':a['x'][j]});base,tr=b['_infer'](ss); baseline.append(base); o=b['_scalar_models'](z[np.r_[np.flatnonzero(i),np.flatnonzero(j)]],s,y[i],configs); L.append(np.array([p[0] for p in o]));Ps.append(np.array([p[1] for p in o]));print('split',np.flatnonzero(j)[0],flush=True)
errs=lambda p:np.array([np.mean(abs(x-y[j]),axis=-1) for x,(_,j) in zip(p,spl)])
E=errs(Ps); L=np.array(L)
for mode,rank in [('mean',E.mean(0)),('max',E.max(0)),('public',E[0])]:
 print(mode,[(configs[h],np.round(E[:,h],4)) for h in np.argsort(rank)[:8]],flush=True)
# Choose compact structural classes; pool only families that improve the validation ratchet.
opts=[]
for typ in ['poly','rbf','matern','plus','tensor','evidence','loo','all']:
 for nn in [1,3,8,16,32,64,128]:
  for scope in ['inner','outer','worst']:
   mask=np.array([typ=='all' or c[0]==typ for c in configs]); ids=np.flatnonzero(mask); rank=E.mean(0) if scope=='outer' else E.max(0)
   if scope!='inner':ids=ids[np.argsort(rank[ids])[:nn]]
   for w in (0,.25,.5):
    pred=[]
    for m,p in enumerate(Ps):
     ix=ids[np.argsort(L[m,ids])[:nn]] if scope=='inner' else ids
     pred.append((1-w)*p[ix].mean(0)+w*baseline[m])
    e=errs(pred); opts.append((e,typ,scope,nn,w,ids))
# Prefer threshold-compliant candidates with the lowest worst-slice error.
opts.sort(key=lambda q:(q[0][0]>.0379,max(q[0]),np.mean(q[0])))
e,typ,scope,nn,w,ids=opts[0]; print('best ensemble',e,typ,scope,nn,w,flush=True)
path='/holdout_task/node_01.py'; code=open('/tmp/base9.py').read(); code+=open('/tmp/scalar11.py').read(); cc=[configs[h] for h in ids]; code+='\n_SCALAR_CONFIGS = '+repr(cc)+'\n'
code+='''\n\ndef run_node(inputs: dict) -> dict:
    study=inputs['__input__']; tr=study['train']; y,traj=_infer(study)
    z=_calibrate(study,np.vstack((tr['x'],study['test']['x'])))
    estimates=_scalar_models(z,study,np.asarray(tr['y']),_SCALAR_CONFIGS)
    scalar=np.mean([p for _,p in sorted(estimates,key=lambda a:a[0])[:'''+str(nn)+''']],axis=0)
    y='''+repr(w)+'''*y+'''+repr(1-w)+'''*scalar
    ref=_baseline({'__input__':study})
    return {'panel_pred':np.column_stack((2*y-ref['y_hat'],(traj-.35*ref['state_pred'])/.65))}
'''
# Keep a single exported entry point.
code=code.replace('def run_node(inputs: dict) -> dict:', 'def _old_entry(inputs: dict) -> dict:',1); p=code.index('\ndef _old_entry'); q=code.index('\ndef _scalar_kernels',p); code=code[:p]+code[q:];open(path,'w').write(code)
