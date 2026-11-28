import numpy as np
from scipy.optimize import least_squares
import cal10 as m
s=m.s
u,v=m.u,m.v
ls=m.ls
def aff(x):return np.c_[np.ones(len(x)),x]
def calibration(f, start):
 def H(x,p):return np.stack([np.ones_like(x), f(x,p[:-1]),np.sin(p[-1]*np.roll(x,2,axis=1))],axis=2)
 def fit(p):return m.fit(H(u,p))
 p=least_squares(lambda p:fit(p)[0].ravel(),start,max_nfev=200).x
 r,c=fit(p);print('cal',np.std(r),p,c.T)
 return np.std(r),lambda z:np.einsum('ijk,jk->ij',H(z,p),c)
cals=[calibration(f,start) for f,start in [(lambda x,p:x+p[0]*np.tanh(p[1]*x),[.22,2.15,2.4]),(lambda x,p:x+p[0]*np.arctan(p[1]*x),[.22,2.15,2.4]),(lambda x,p:x+p[0]*np.sin(p[1]*x),[.22,2.15,2.4]),(lambda x,p:x/(1+p[0]*abs(x)**p[1]),[.22,1.25,2.4]),(lambda x,p:x+p[0]*x/(1+p[1]*x*x),[.22,2.15,2.4]),(lambda x,p:x+p[0]*x/(1+p[1]*abs(x)**p[2]),[.22,2.15,1.25,2.4]),(lambda x,p:x+p[0]*m.erf(p[1]*x),[.22,2.15,2.4])]]
f=min(cals,key=lambda t:t[0])[1]
z=np.array([least_squares(lambda z:f(z[None])[0]-x,x,gtol=1e-12).x for x in np.r_[s['train']['x'],s['test']['x']]])
print('z ranges',z.min(0),z.max(0),'max inverse error',np.max(abs(f(z)-np.r_[s['train']['x'],s['test']['x']])))
rat=lambda x,p:np.sign(x)*abs(x)**p[0]/(1+p[1]*abs(x)**p[2]); soft=lambda x,p:np.logaddexp(0,p[0]*x+p[1]); dead=lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1]
def panel(k,j,forms):
 u,v=s[k]['u'],s[k][j]; fits=[]
 for f,p,bounds in forms:
  def calc(p):
   h=aff(f(u,p)); c=ls(h,v);return (h@c-v).ravel(),c
  p=least_squares(lambda p:calc(p)[0],p,bounds=bounds,max_nfev=400).x; r,c=calc(p); print('operator',k,j,np.std(r),p); fits.append((np.std(r),aff(f(z,p))@c))
 return min(fits,key=lambda t:t[0])[1]
b=panel('panel_b','v',[(dead,[.35,1.5],([0,.2],[.8,12])),(rat,[7,30,5],([.1,.001,.1],[20,10000,20])),(lambda x,p:np.sign(x)*abs(x)**p[0],[2],([.1],[20])),(lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1]/(1+p[2]*abs(x)**2),[.2,2,.2],([0,.2,0],[.8,12,10]))]); c=panel('panel_c','v1',[(rat,[1,.9,1.25],([.1,.001,.1],[5,30,7]))]); d=panel('panel_c','v2',[(soft,[2.2,0],([.1,-5],[10,5]))]); np.savez('/tmp/f10.npz',z=z,b=b,c=c,d=d)
y=s['train']['y']; t=s['train']['traj']; print('y versus traj',ls(aff(t),y),np.std(aff(t)@ls(aff(t),y)-y)); print('summary',s['train']['summary']); print('y/traj stats',y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std());
def error(h,Y):
 pred=h[52:72]@ls(h[:52],Y[:52]);return np.mean(abs(pred-Y[52:]),axis=0)
for n,x in [('raw',np.r_[s['train']['x'],s['test']['x']]),('z',z),('b',b),('c',c),('d',d)]:
 pair=np.array([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]).T
 for name,H in [('lin',x),('pair',np.c_[x,x[:,0]*x[:,2]+x[:,1]*x[:,3]]),('quad',np.c_[x,pair])]:
  print('cv',n,name,round(error(aff(H),y),5),np.round(error(aff(H),t),4))
