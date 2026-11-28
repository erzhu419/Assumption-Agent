import pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=5,suppress=True,linewidth=150)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
u=s['panel_a']['u'];v=s['panel_a']['v'];a=aff(u);print('rows u,v,delta'); print(np.c_[u,v,v-u]);
rng=np.random.default_rng(1);best=None
for i in range(3000):
 j=rng.choice(len(a),5,replace=False); w=ls(a[j],v[j]); e=np.linalg.norm(a@w-v,axis=1); q=np.median(e)
 if best is None or q<best[0]:best=(q,w,e)
print('ransac',best[:2],np.sort(best[2]));
for j in range(4):
 ij=[0,1+j,1+(j+1)%4]; p=a[:,ij]; print('col',j,'linear err',np.sort(np.abs(p@ls(p,v[:,j])-v[:,j]))); print('rob',least_squares(lambda w:p@w-v[:,j],ls(p,v[:,j]),loss='cauchy',f_scale=.025).x)
# Fit the observed cyclic structure and inspect how much residual can be recovered by a low-order algebraic term.
for label,f in [('linear',lambda u:u),('square',lambda u:np.c_[u,u*u]),('cubic',lambda u:np.c_[u,u*u,u**3]),('qcross',lambda u:np.c_[u,*[u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]]),('kink',lambda u:np.c_[u,abs(u),u*abs(u)]),('rat',lambda u:np.c_[u,u/(1+.8*abs(u)),np.log1p(.36*u*u)])]:
 p=aff(f(u)); print(label, 'rmse',np.std(p@ls(p,v)-v),'rob med',np.median(np.abs(p@np.column_stack([least_squares(lambda w:p@w-v[:,j],ls(p,v[:,j]),loss='cauchy',f_scale=.035).x for j in range(4)])-v)))
