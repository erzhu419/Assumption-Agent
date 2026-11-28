import pickle,numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def D(u): return np.c_[np.ones(len(u)),u]
def fit(u,v): return np.linalg.lstsq(u,v,rcond=1e-10)[0]
def err(F,v): return np.linalg.norm(F@fit(F,v)-v,axis=1)
if __name__=='__main__':
 for name in ('panel_a','panel_b','panel_c'):
  p=S[name]; u=p['u']; pair=np.column_stack([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)])
  for key in p.keys()-{'u'}:
   v=p[key]; print(name,key,'std',v.std(0).round(4))
   for nm,F in [('linear',D(u)),('diagonal2',D(np.c_[u,u*u])),('pair+3',D(np.c_[u,u*np.roll(u,-1,axis=1),u**3])),('poly2',D(np.c_[u,u*u,pair])),('erf',D(erf(1.25*u))),('tanh',D(np.tanh(1.25*u))),('threshold',D(np.sign(u)*np.maximum(abs(u)-.22,0)**1.25))]:
    e=err(F,v); k=e<3*np.median(e); print(nm, np.round(np.quantile(e,[0,.5,.9,1]),5),'trim',k.sum(),np.sqrt(np.mean((F[k]@fit(F[k],v[k])-v[k])**2)).round(5))
   print('linear coeff',fit(D(u),v).round(4))
   for nm,tf,p0,lo,hi in [('thresh',lambda u,q: np.sign(u)*np.maximum(abs(u)-q[0],0)**q[1],[.22,1.25],[0,.1],[1,4]),('erf+u',lambda u,q: erf(q[0]*u)+q[1]*u,[1.25,.1],[.05,-2],[8,2]),('tanh+u',lambda u,q: np.tanh(q[0]*u)+q[1]*u,[1.25,.1],[.05,-2],[8,2]),('rational+u',lambda u,q: u/(1+q[0]*u*u)**q[1],[1.25,.5],[.01,-2],[20,3])]:
    def res(q):
     F=D(tf(u,q));return (F@fit(F,v)-v).ravel()
    r=least_squares(res,p0,bounds=(lo,hi));print(nm,r.x.round(5),np.sqrt(np.mean(r.fun**2)).round(6))
