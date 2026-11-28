import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fit=lambda a,y:np.linalg.lstsq(a,y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for k in range(1,d+1) for i in cr(range(4),k)])
def learn(u,v,f,initial,lo,hi):
 def r(par):
  d=f(u,par);return (d@fit(d,v)-v).ravel()
 o=least_squares(r,initial,bounds=(lo,hi),max_nfev=160);return o.x,fit(f(u,o.x),v),np.mean(abs(o.fun))
def c1(u,p):return np.c_[np.ones(len(u)), np.tanh(u*p[:4]+p[4:8])+u*p[8:]]
def nonlinear(u,p,fn):return np.c_[aff(u),fn(u*p[:4]+p[4:])]
u,v=s['panel_a']['u'],s['panel_a']['v'];keep=np.arange(len(u));
for _ in range(5):
 e=np.linalg.norm(aff(u)@fit(aff(u[keep]),v[keep])-v,axis=1);keep=np.argsort(e)[:45]
print('A quantiles by channel and max',np.quantile(u,[0,.25,.5,.75,1],axis=0)); print('A residual sample', np.round(np.c_[u,v-aff(u)@fit(aff(u[keep]),v[keep])][keep[:12]],3))
for name,f,ini,lo,hi in [('quadratic',lambda z,p:poly(z,2),[0],[-1],[1]),('cubic',lambda z,p:poly(z,3),[0],[-1],[1])]+[(k,lambda z,p,fn=fn:nonlinear(z,p,fn),[1.5]*4+[0]*4,[.08]*4+[-4]*4,[7]*4+[4]*4) for k,fn in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('rat',lambda w:1/(1+w*w))]]:
 par,co,e=learn(u[keep],v[keep],f,ini,lo,hi); vals=[]
 for a,b in zip(np.array_split(keep,5),range(5)):
  tr=np.setdiff1d(keep,a);p,c,_=learn(u[tr],v[tr],f,ini,lo,hi);vals.append(np.mean(abs(f(u[a],p)@c-v[a])))
 print('A',name,'score',e,np.round(vals,4),'par',np.round(par,3)); print(np.round(co,3) if name in ('quadratic','tanh') else '')
u=s['panel_c']['u'];v=s['panel_c']['v1']; ini=[2]*4+[0]*4+[.4]*4;lo=[.4]*4+[-2]*4+[-2]*4;hi=[6]*4+[2]*4+[3]*4
p,co,e=learn(u,v,c1,ini,lo,hi);print('C1 structured',p,co,e)
