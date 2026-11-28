import runpy,pickle,numpy as np
from itertools import combinations_with_replacement
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
a=runpy.run_path(p+'_frozen_nodes/node_02.py'); b=runpy.run_path(p+'_frozen_nodes/node_00.py')
up=a['run_node']({'__input__':s}); mix=b['run_node']({'node_02':up})
u=up['u_tr']; z=mix['z_tr']; T=s['eval_times']; Y=s['train']['y']; R=s['train']['traj']; C=s['panel_c']; B=np.linalg.lstsq(b['_affine'](up['b_u']),up['b_v'],rcond=None)[0]
def fit(F,V,n):
    co=np.linalg.lstsq(F[:n],V[:n],rcond=None)[0]; e=F@co-V
    return np.round(np.mean(abs(e[:n]),axis=0),5),np.round(np.mean(abs(e[n:]),axis=0),5)
def basis(x,d):
    return np.column_stack([np.ones(len(x)),*[np.prod(x[:,ks],axis=1) for i in range(1,d+1) for ks in combinations_with_replacement(range(x.shape[1]),i)]])
if __name__=='__main__':
 print('times',T,'B',np.round(B,4)); print('c ranges',C['u'].min(axis=0),C['u'].max(axis=0))
 c=C['u']; cz=b['_affine'](c)@B; cu=(c-B[0])@np.linalg.inv(B[1:]); print('trajectory singulars',np.linalg.svd(R,compute_uv=False)); print('mean',R.mean(0),'std',R.std(0))
 for name,f in [('t',T),('sqrt',np.sqrt(T)),('log',np.log1p(T)),('exp',1-np.exp(-T))]:
  P=np.outer(R@f/(f@f),f); print(name,round(abs(P-R).mean(),6),(R.T@R[:,-1]/(R[:,-1]@R[:,-1])).round(4))
 for D in ['v1','v2']:
  V=C[D]
  for xname,x in [('c',c),('cu',cu),('cz',cz)]:
   for nm,F in [('aff',basis(x,1)),('quad',basis(x,2)),('tanh',basis(np.tanh(x),1)),('sin',basis(np.sin(x),1)),('reduced cubic',np.c_[basis(x,2),x**3])]:
    print(D,xname,nm,fit(F,V,19))
 for xname,x in [('u',u),('z',z),('raw',s['train']['x'])]:
  for nm,F in [('aff',basis(x,1)),('quad',basis(x,2)),('cubic',basis(x,3)),('tanh',basis(np.tanh(x),1)),('sin',basis(np.sin(x),1)),('reduced cubic',np.c_[basis(x,2),x**3])]:
   print(xname,nm,fit(F,Y,52),fit(F,R,52)[1])
 print('panel sample',np.round(np.c_[c,C['v1'],C['v2']][:6],4))
