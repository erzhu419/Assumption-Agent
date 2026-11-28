from operators04 import S,ls,aff,calibrate,F
import numpy as np
from scipy.optimize import least_squares
p=S['panel_b']; u,v=p['u'],p['v']; tr=S['train']; w=calibrate(S,tr['x']); y,t=tr['y'],tr['traj']; times=S['eval_times']

def candidates():
 yield 'base', lambda q,p:q**3/(1+p[0]*abs(q)**p[1])+p[2]*q, [.55,2.4,-.095],[.015,.6,-4],[8,6,4]
 yield 'rational numerator', lambda q,p:(q**3+p[2]*q)/(1+p[0]*abs(q)**p[1]), [.55,2.4,-.095],[.015,.6,-4],[8,6,4]
 yield 'rational mixed', lambda q,p:(q**3+p[3]*q)/(1+p[0]*abs(q)**p[1])+p[2]*q, [.55,2.4,-.095,0],[.015,.6,-4,-4],[8,6,4,4]
 yield 'soft clipping', lambda q,p:q/(1+p[0]*abs(q)**p[1])+p[2]*q, [.55,2.4,-.095],[.015,.6,-4],[8,6,4]
 yield 'powered cubic', lambda q,p:q**3/(1+p[0]*abs(q)**p[1])**p[3]+p[2]*q, [.55,2.4,-.095,1],[.015,.6,-4,.15],[8,6,4,5]
 yield 'separate rational', lambda q,p:q**3/(1+p[0:4]*abs(q)**p[4])+p[5:9]*q, [.55]*4+[2.4]+[-.095]*4,[.015]*4+[.6]+[-4]*4,[8]*4+[6]+[4]*4
 yield 'rational cross linear', lambda q,p:np.c_[q,q**3/(1+p[0]*abs(q)**p[1])], [.55,2.4],[.015,.6],[8,6]

def profile(f, p, wfit, yfit, tfit, alpha):
 a=aff(f(u,p)); b=ls(a,v); z=aff(f(wfit,p))@b
 d=np.c_[aff(z),z[:,0]*z[:,2]+z[:,1]*z[:,3]]; a2=aff(z)
 cy=ls(d,yfit); ct=ls(a2,tfit)
 return np.r_[(a@b-v).ravel(),alpha*(d@cy-yfit),alpha*(a2@ct-tfit).ravel()/np.sqrt(tfit.shape[1])],b,cy,ct

def run(f,p,lo,hi, i,j, alpha):
 opt=least_squares(lambda p:profile(f,p,w[i],y[i],t[i],alpha)[0],p,bounds=(lo,hi),max_nfev=100)
 _,b,cy,ct=profile(f,opt.x,w[i],y[i],t[i],alpha); z=aff(f(w[j],opt.x))@b; q=np.c_[aff(z),z[:,0]*z[:,2]+z[:,1]*z[:,3]]
 return np.array([abs(q@cy-y[j]).mean(),abs(aff(z)@ct-t[j]).mean()]),opt.x,b

if __name__=='__main__':
 rng=np.random.default_rng(194); splits=[(np.arange(56),np.arange(56,72))]; order=rng.permutation(72); splits += [(np.setdiff1d(order,j),j) for j in np.array_split(order,5)]
 for name,f,p,lo,hi in candidates():
  for alpha in (0, .7, 2):
   out=[run(f,p,lo,hi,i,j,alpha) for i,j in splits]; es=np.array([e[0] for e in out]); _,k,b=out[0]; a=aff(f(u,k)); r=a@b-v
   print(name,alpha,np.round(k,5),'panelrms',round(np.sqrt(np.mean(r*r)),6),'56/16',np.round(es[0],5),'mean',np.round(es[1:].mean(0),5),'max',np.round(es.max(0),5))
   if name=='base' and alpha==0:
    z=aff(f(w,k))@b; c=ls(aff(z),t); print('operator',np.round(b,5),'temporal coefficients',np.round(c,5),sep='\n'); print('trajectory residual spectrum',np.linalg.svd(t-aff(z)@c,compute_uv=False)); print('calibrated query max',abs(calibrate(S,S['test']['x'])).max(0))
