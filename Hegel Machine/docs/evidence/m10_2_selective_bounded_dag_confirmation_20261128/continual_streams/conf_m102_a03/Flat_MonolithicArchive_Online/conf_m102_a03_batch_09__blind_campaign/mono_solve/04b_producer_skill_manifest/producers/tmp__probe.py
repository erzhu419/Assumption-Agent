import pickle, numpy as np, importlib.util
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; X=t['x']; Y=np.c_[t['y'],t['traj']]; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; aff=lambda a:np.c_[np.ones(len(a)),a]
def get(i):
 sp=importlib.util.spec_from_file_location('m',f'/tmp/hegel_continual_archive/conf_m102_a03_batch_{i:02}_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
m=get(2); A=m._calibration(s['panel_a']); Z=m._decode(X,A); print('calibration',A); print('Z range',Z.min(0),Z.max(0))
b=s['panel_b']; B=fit(aff(b['u']),b['v']); c=s['panel_c']; op=m._panel_operators(c); print('C inferred rms',[np.sqrt(np.mean((f(c['u'])-c[k])**2)) for f,k in zip(op,('v1','v2'))]);
for i in [0,1,2,3,4,6,8]:
 st={**s,'train':{**t,**{k:t[k][:52] for k in ('x','y','traj')}},'test':{'x':X[52:]}}
 try:
  g=get(i).solve_batch(st); print('archive',i,np.abs(np.c_[g['y_hat'],g['state_pred']]-Y[52:]).mean(0).round(4))
 except Exception as e: print(i,e)
for name,Q in [('X',X),('Z',Z),('B',aff(Z)@B),('C',op[0](Z)),('D',op[1](Z)),('BC',op[0](aff(Z)@B)),('BD',op[1](aff(Z)@B)),('ZCD',np.c_[Z,op[0](Z),op[1](Z)]),('ZBCD',np.c_[Z,op[0](aff(Z)@B),op[1](aff(Z)@B)])]:
 for deg in ([1,2,3] if Q.shape[1]==4 else [1]):
  F=np.column_stack([np.ones(len(Q))]+[np.prod(Q[:,ij],axis=1) for d in range(1,deg+1) for ij in combinations_with_replacement(range(Q.shape[1]),d)]); C=fit(F[:52],Y[:52]); r=F@C-Y; print(name,deg,'train',np.abs(r[:52]).mean(0)[[0,1,8]].round(5),'valid',np.abs(r[52:]).mean(0)[[0,1,8]].round(5))
