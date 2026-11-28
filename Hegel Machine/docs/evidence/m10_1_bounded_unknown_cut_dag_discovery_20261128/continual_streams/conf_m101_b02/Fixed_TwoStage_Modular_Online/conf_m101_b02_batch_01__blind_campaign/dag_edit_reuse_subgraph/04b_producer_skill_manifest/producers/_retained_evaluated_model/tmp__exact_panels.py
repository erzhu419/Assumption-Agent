import itertools as it, pickle
import numpy as np
from scipy.optimize import minimize, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
funs={'sin':np.sin,'tanh':np.tanh,'rat':lambda x:x/(1+x*x),'atan':np.arctan,'erf':__import__('scipy.special',fromlist=['erf']).erf,'cos':np.cos,'abs2':lambda x:x*abs(x),'square':np.square}
def design(u,names,ks,linear):
 return np.column_stack([np.ones(len(u))]+([u] if linear else [])+[funs[name](k*u) for name,k in zip(names,ks)])
def find(u,v,choices,linear=True):
 out=[]
 for names in choices:
  active=[j for j,name in enumerate(names) if name not in ['abs2','square']]
  def ks(p):
   a=np.ones(len(names));a[active]=p;return a
  def score(p):
   if len(p)==2 and names[0]==names[1] and abs(p[0]-p[1])<1e-5:return 1.
   a=design(u,names,ks(p),linear); r=v-a@np.linalg.lstsq(a,v,rcond=None)[0];return np.mean(r*r)
  ps=list(it.product([.8,1.6,2.6,3.6],repeat=len(active)))
  p,y=min(((r.x,r.fun) for p in ps if (len(p)!=2 or names[0]!=names[1] or p[0]<p[1]) for r in [minimize(lambda q:1e4*score(q),p,bounds=[(.3,5)]*len(p),method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-9})]),key=lambda e:e[1]);
  a=design(u,names,ks(p),linear); q=np.linalg.qr(a,mode='reduced')[0]; err=(v-q@(q.T@v))/(1-(q*q).sum(1))[:,None];out.append((y, names,ks(p),np.mean(abs(err),axis=0).max()))
 return sorted(out,key=lambda a:a[0])
if __name__=='__main__':
 for pn,vn,opts in [('panel_b','v',list(it.combinations_with_replacement(['sin','tanh','rat','atan','erf'],2))+[('sin','abs2'),('tanh','abs2')]),('panel_c','v1',[('sin','abs2'),('tanh','abs2')]),('panel_c','v2',[('cos','square'),('cos',)])]:
  u,v=s[pn]['u'],s[pn][vn]; print(pn,vn,find(u,v,opts)[:6],flush=True)
