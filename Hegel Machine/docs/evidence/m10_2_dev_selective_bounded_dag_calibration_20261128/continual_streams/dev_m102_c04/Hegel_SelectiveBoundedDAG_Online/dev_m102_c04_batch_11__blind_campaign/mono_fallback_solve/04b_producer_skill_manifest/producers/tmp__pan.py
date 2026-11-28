import pickle,numpy as np,itertools as it
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=lambda a,b: np.linalg.lstsq(a,b,rcond=1e-10)[0]
for key in ('panel_a','panel_b','panel_c'):
 p=s[key]; u=p['u']; mask=np.ones(len(u),bool)
 for k,v in p.items():
  if k=='u':continue
  if key=='panel_a': mask=np.linalg.norm(v-u,axis=1)<4
  u0=u[mask]; v0=v[mask]; print(key,k,'n',len(u0),'range',v0.min(0),v0.max(0))
  funcs={'quadratic':np.column_stack([u0[:,i]*u0[:,j] for i,j in it.combinations_with_replacement(range(4),2)]),'sin':np.sin(u0),'tanh':np.tanh(u0),'square':u0**2,'abs':abs(u0),'cube':u0**3,'pair-roll':u0*np.roll(u0,1,axis=1),'sin-pair':np.column_stack([np.sin(u0[:,i]+sig*u0[:,j]) for i,j in it.combinations(range(4),2) for sig in (1,-1)])}
  base=np.c_[np.ones(len(u0)),u0]; rng=np.random.default_rng(54)
  for n,B in [('affine',base)]+[(name,np.c_[base,U]) for name,U in funcs.items()]:
   a=[]
   for i in range(10):
    ids=rng.permutation(len(B)); tr,te=ids[:-5],ids[-5:]; a.append(np.mean(abs(B[te]@ls(B[tr],v0[tr])-v0[te])))
   W=ls(B,v0); print(n,round(np.mean(abs(B@W-v0)),6),round(np.mean(a),6));
   if n=='quadratic' and key=='panel_a': print(np.round(W,4))
print('ranges train/test',s['train']['x'].min(0),s['train']['x'].max(0),s['test']['x'].min(0),s['test']['x'].max(0))
