from stress import *
from scipy.optimize import least_squares
from itertools import combinations
u,v=s['panel_b']['u'],s['panel_b']['v']; ls=m._ls
F={'sin':np.sin,'tanh':np.tanh,'asinh':np.arcsinh,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'erf':__import__('scipy').special.erf,'exp':lambda x:x*np.exp(-x*x),'cube':lambda x:x**3,'soft':lambda x:x/(1+abs(x)),'sq':lambda x:x*x,'logcosh':lambda x:np.logaddexp(x,-x)-np.log(2),'cos':lambda x:1-np.cos(x)}
rec=[]
def test(tag,f,starts,lo,hi):
 def d(x,p):return np.c_[np.ones(len(x)),f(x,p)]
 def r(p):
  a=d(u,p);return (a@ls(a,v)-v).ravel()
 fit=min((least_squares(r,p,bounds=(lo,hi),max_nfev=90) for p in starts),key=lambda a:a.cost);p=fit.x; a=d(u,p); h=(np.linalg.qr(a)[0]**2).sum(1); rm=np.sqrt(np.mean((r(p).reshape(v.shape)/(1-h[:,None]))**2));rec.append((rm,tag,p,lambda x:d(x,p)@ls(a,v)))
for name,g in F.items():
 for kind in ('offset','coordgain','coordoffset'):
  if kind=='offset':
   test((name,kind),lambda x,p,g=g:x+p[1]*g(p[0]*x+p[2]),[[a,b,0] for a in (1.25,2.6,4.2) for b in (-.4,.4)],[.05,-10,-3],[10,10,3])
  else:
   test((name,kind),lambda x,p,g=g,kind=kind:x+(p[1:5] if kind=='coordgain' else p[1])*g(p[0]*x+(0 if kind=='coordgain' else p[2:6])),[[a]+([b]*4 if kind=='coordgain' else [b]+[0]*4) for a in (1.25,2.6,4.2) for b in (-.4,.4)],[.05]+[-10]*(4 if kind=='coordgain' else 5),[10]+[10]*(4 if kind=='coordgain' else 5))
for (name,g),(name2,h) in combinations(F.items(),2):
 test((name,name2),lambda x,p,g=g,h=h:x+p[2]*g(p[0]*x)+p[3]*h(p[1]*x),[[a,b,c,d] for a,b in [(1.25,2.15),(2.15,1.25),(4,3)] for c,d in [(-.4,.4),(.4,-.4),(-.4,-.4)]],[.05,.05,-10,-10],[10,10,10,10])
z,h=m._coords(s,xx); C=m._panel(s['panel_c'],'v1',lambda x,p:np.c_[np.sin(p[0]*x),np.tanh(p[1]*x)],[2.1,1.75],([.15,.15],[6,6])); rec.append((.002775,'baseline',None,None));
for rm,n,p,b in sorted(rec,key=lambda v:v[0])[:18]:
 H=h if b is None else C(b(z)); X=np.c_[np.ones(len(z)),z,H,(z*H).sum(1),z[:,1]*H[:,0]]; es=[]
 for _,i,j in slices:
  e=np.abs(X[j]@ls(X[i],y[i])-y[j]);es.append([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()])
 print(n,round(rm,7),np.round(p,5) if p is not None else '',np.round(es[0],6),np.round(np.max(es,axis=0),6))
