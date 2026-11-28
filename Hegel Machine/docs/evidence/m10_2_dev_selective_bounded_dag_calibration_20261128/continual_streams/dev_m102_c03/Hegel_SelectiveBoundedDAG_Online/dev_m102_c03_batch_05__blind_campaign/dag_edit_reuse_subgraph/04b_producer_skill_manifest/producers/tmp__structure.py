import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];pairs=list(it.combinations_with_replacement(range(4),2));print(pairs);np.set_printoptions(precision=4,suppress=True)
def Q(x):return np.c_[np.ones(len(x)),x,*[x[:,i]*x[:,j] for i,j in pairs]]
for k,vkey in [('panel_a','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[k];u=p['u'];v=p[vkey];h=np.c_[np.ones(len(u)),u]; keep=np.linalg.norm(v-h@np.linalg.lstsq(h,v,rcond=None)[0],axis=1)<3 if k=='panel_a' else np.ones(len(u),bool);f=Q(u[keep]);v=v[keep];c=np.linalg.lstsq(f,v,rcond=None)[0];print(k,vkey,c, 'err',np.std(v-f@c,0))
 if k=='panel_a':
  u=u[keep];w=np.roll(u,-1,axis=1);w2=np.roll(u,-2,axis=1)
  for j in range(4):
   z,a,b=u[:,j],w[:,j],w2[:,j]; out=v[:,j];
   for name,h in [('prod',z*a),('sinprod',np.sin(z*a)),('tanhprod',np.tanh(z*a)),('rational',z*a/(1+.5*(z*z+a*a))),('second',z*b),('both',np.c_[z*a,z*b]),('satprod',z*a*np.exp(-.2962*(z*z+a*a))),('sinz',np.sin(z))]:
    d=np.c_[np.ones(len(z)),u,h];c=np.linalg.lstsq(d,out,rcond=None)[0]; print(j,name,np.std(out-d@c),c.round(3))
