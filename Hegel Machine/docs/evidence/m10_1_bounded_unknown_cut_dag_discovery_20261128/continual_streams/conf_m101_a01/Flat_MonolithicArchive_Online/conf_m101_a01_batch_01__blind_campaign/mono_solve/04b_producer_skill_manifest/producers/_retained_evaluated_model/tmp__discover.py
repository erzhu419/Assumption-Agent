from features import *
from scipy.optimize import minimize
from scipy.sparse import issparse
from sklearn.linear_model import lars_path
P=s['panel_c']; U0=P['u']; Z=P['v1']; Z2=P['v2']
I=np.arange(52); J=np.arange(52,72)
def vp(fun,z,start,bounds,linear=False):
 def H(u,p):return np.c_[np.ones(len(u)),u,fun(u,*p)] if linear else af(fun(u,*p))
 def res(p):
  a=H(U0,p);return (a@ls(a,z)-z).ravel()
 p=least_squares(res,start,bounds=bounds,max_nfev=120).x
 return np.sqrt(np.mean(res(p)**2)),p,lambda u:H(u,p)@ls(H(U0,p),z)
funcs=[('tanh',lambda u,a,b:np.tanh(a*u+b),[2,0],([.2,-1],[6,1]),True)]
for name,f in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('cos',np.cos),('root',lambda x:np.sqrt(1+x*x)-1),('rat',lambda x:x/(1+x*x)),('cubic',lambda x:x**3)]:
 for j in [0,1]:
  funcs.append((name+str(j),lambda u,a,b,c,f=f:np.tanh(a*u)+b*f(c*u),[1.6,.2,2.5],([.2,-2,.2],[6,2,7]),bool(j)))
  funcs.append((name+'inner'+str(j),lambda u,a,b,c,f=f:np.tanh(a*u+b*f(c*u)),[1.6,.2,2.5],([.2,-2,.2],[6,2,7]),bool(j)))
funcs.extend([('erf',lambda u,a:__import__('scipy').special.erf(a*u),[1.6],([.2],[5]),True),('sqrtodd',lambda u,a,b:u/(1+(a*u)**2)**b,[1.6,.5],([.2,.1],[5,2]),True)])
r=[]
for nm,f,p,b,l in funcs:
 try:r.append((vp(f,Z,p,b,l),nm))
 except ValueError:pass
for (e,p,_),nm in sorted(r,key=lambda a:a[0][0])[:12]:print(nm,e,p,flush=True)
c=min(r,key=lambda a:a[0][0])[0][2]; e,p,d=vp(lambda u,a,b:np.sqrt(1+(a*u+b)**2)-1,Z2,[.95,0],([.5,-.2],[1.5,.2]),True); print('root',e,p,flush=True)
S={}; originals={k:coords[k] for k in ['x','a','ai','xb','ab','aib']}
for k,u in originals.items():
 S[k]=u
 for name,f in [('c',c),('d',d)]:
  v0=f(u); S[k+name]=v0; S[k+name+'b']=af(v0)@B
  for name2,f2 in [('c',c),('d',d)]:S[k+name+name2]=f2(v0)
def rank(F,reg):
 from scipy.linalg import solve
 m=F[I].mean(0); z=F-m; scale=np.maximum(np.sqrt(np.mean(z[I]**2,axis=0)),1e-7); z/=scale
 if reg==0:w=ls(np.c_[np.ones(len(I)),z[I]],Y[I]); return np.mean(abs(np.c_[np.ones(len(J)),z[J]]@w-Y[J]),0)
 K=z@z.T; a=solve(K[np.ix_(I,I)]+reg*np.eye(len(I)),Y[I]-Y[I].mean(0),assume_a='pos'); return np.mean(abs(K[np.ix_(J,I)]@a+Y[I].mean(0)-Y[J]),0)
rows=[]
for k,u in S.items():
 for l,v0 in [('none',u),('raw',x),('cal',coords['a']),('C',S.get(k[0]+'c',u)),('D',S.get(k[0]+'d',u))]:
  if l in ('C','D') and k[0]+'c' not in S:continue
  H=np.c_[u,v0] if l!='none' else u
  for nm,h in [('q',pol(H,2)[:,1:]),('p3',pol(H,3)[:,1:]),('rat',np.c_[pol(H,2)[:,1:],u/(1+v0**2),u*u/(1+v0**2),u*v0/(1+v0**2)]),('sat',np.c_[pol(H,2)[:,1:],np.tanh(1.6*H),np.sin(1.6*H),np.cos(1.6*H)])]:
   for reg in [.1,3.]:rows.append((rank(h,reg),k,l,nm,reg))
for t0 in [0,1]:
 for row in sorted(rows,key=lambda a:a[0][t0])[:12]:print(t0,row,flush=True)
# Profile a sparse residual dictionary around the independently observed quadratic trend.
base=pol(x,2); F=[]; labels=[]
for k,u in S.items():
 for nm,H in [('p3',pol(u,3)[:,1:]),('ratio',np.c_[u/(1+u*u),1/(1+u*u)]),('trig',np.c_[np.sin(1.6*u),np.cos(1.6*u),np.tanh(1.6*u)]),('mixed',(u[:,:,None]*x[:,None,:]).reshape(len(x),-1))]:
  F.append(H);labels.extend((k,nm,j) for j in range(H.shape[1]))
F=np.column_stack(F); F-=base@ls(base[I],F[I]); F/=np.maximum(np.linalg.norm(F[I],axis=0),1e-8); Y0=Y-base@ls(base[I],Y[I]);
for j in range(2):
 _,ids,W=lars_path(F[I],Y0[I,j],method='lasso',max_iter=45)
 scores=np.mean(abs(F[J]@W-Y0[J,j,None]),0); ix=scores.argmin(); print('sparse',j,scores[ix],ix,[labels[k] for k in ids[:ix]],flush=True)
