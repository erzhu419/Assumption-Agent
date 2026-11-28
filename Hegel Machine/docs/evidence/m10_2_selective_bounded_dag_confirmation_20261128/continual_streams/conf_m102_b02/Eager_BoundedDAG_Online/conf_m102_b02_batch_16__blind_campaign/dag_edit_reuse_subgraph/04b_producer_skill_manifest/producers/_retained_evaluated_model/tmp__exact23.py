import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from model23 import D,fit,get
from probe23 import S

def sat(x,a,k):
 if k=='tanh': return np.tanh(a*x)
 if k=='erf': return erf(a*x)
 if k=='sin': return np.sin(a*x)
 if k=='rat': return x/(1+(a*x)**2)
 if k=='soft': return x/np.sqrt(1+(a*x)**2)
 if k=='atan': return np.arctan(a*x)

def models(s,verbose=False):
 z0,b0,c,f,Y,aux=get(s); keep=aux[1]; u,v=[s['panel_a'][k][keep] for k in ['u','v']]; x=np.r_[s['train']['x'],s['test']['x']]
 cand=[]
 for k in ['tanh','erf','sin','rat','soft','atan']:
  for typ in [0,1,2]:
   def F(z,a,k=k,typ=typ):
    h=sat(z,a,k); return D(np.c_[z,h] if typ==0 else np.c_[z,h,z*z] if typ==1 else np.c_[z,h,z*np.roll(z,-1,axis=1)])
   def e(a):
    A=F(u,a);return (A@fit(A,v)-v).ravel()
   r=least_squares(e,1.25,bounds=(.1,5));cand.append((np.mean(r.fun**2)**.5,k,typ,r.x,F))
 cand.sort(key=lambda t:t[0]); ee,k,typ,a,F=cand[0]; w=fit(F(u,a),v); z=np.array([least_squares(lambda zz:(F(zz[None,:],a)@w-xx).ravel(),zz,bounds=(-3,3)).x for xx,zz in zip(x,z0)])
 if verbose:print('A',[(round(e,7),k,t,a.round(5)) for e,k,t,a,_ in cand[:10]],'z',z.min(),z.max(),'inverse',np.max(abs(F(z,a)@w-x)), 'w',w.round(4),flush=True)
 u,v=[s['panel_b'][k] for k in ['u','v']]; H=lambda z:D(np.sign(z)*np.maximum(abs(z)-.35,0)**1.5); B=lambda z:H(z)@fit(H(u),v)
 u,h=[s['panel_c'][k] for k in ['u','v2']]; cand=[]
 for k in ['tanh','erf','sin','rat','soft','atan']:
  for typ in range(4):
   def F(z,a,k=k,typ=typ):
    h=sat(z,a,k);return D([h,h*h][0] if typ==0 else np.c_[z,h] if typ==1 else np.c_[h,h*h] if typ==2 else np.c_[z,h,h*h])
   def e(a):
    A=F(u,a);return (A@fit(A,h)-h).ravel()
   r=least_squares(e,1.25,bounds=(.1,5));cand.append((np.mean(r.fun**2)**.5,k,typ,r.x,F))
 cand.sort(key=lambda t:t[0]); ee,k,typ,a,F=cand[0];w=fit(F(u,a),h); E=lambda z:F(z,a)@w
 if verbose:print('C2',[(round(e,7),k,t,a.round(5)) for e,k,t,a,_ in cand[:10]],w.round(4),flush=True)
 return z,B,c,E,f,Y

if __name__=='__main__':
 z,B,C,E,f,Y=models(S,True); b=B(z); n=len(Y); rng=np.random.default_rng(171); splits=[]
 for rep in range(2):
  ii=rng.permutation(n)
  for ts in np.array_split(ii,4):splits.append((np.setdiff1d(np.arange(n),ts),ts))
 splits.append((np.arange(52),np.arange(52,n)))
 def score(K,ridge):
  a=[]
  for tr,ts in splits:
   p=K[np.ix_(ts,tr)]@np.linalg.solve(K[np.ix_(tr,tr)]+ridge*np.eye(len(tr)),Y[tr]); e=p-Y[ts]; a.append([abs(e[:,0]).mean(),abs(e[:,1:]@f.T).mean()]);
  a=np.array(a);return a[:-1].mean(0),a[-1]
 base={'z':z,'b':b,'cz':C(z),'cb':C(b),'ez':E(z),'eb':E(b)}
 for a in [-2,-1,-.5,.5,1,2]:
  for name,g in [('i',lambda x:x),('c',C),('e',E)]:base[f'{name}(b{a:+}z)']=g(b+a*z)
 sets=dict(base)
 for a,xa in base.items():
  for b,xb in base.items():
   if a<b and (a in ['z','b','cz','cb','ez','eb'] or b in ['z','b','cz','cb','ez','eb']): sets[a+' '+b]=np.c_[xa,xb]
 sets['all']=np.c_[tuple(base[k] for k in ['z','b','cz','cb','ez','eb'])]
 rows=[]
 for nm,xx in sets.items():
  for gf,x in [('raw',xx),('asinh',np.arcsinh(xx)),('tanh',np.tanh(1.25*xx)),('rational',xx/(1+(1.25*xx)**2))]:
   x=x[:n]; dot=x@x.T/x.shape[1]; dist=np.maximum(np.diag(dot)[:,None]+np.diag(dot)-2*dot,0)
   for deg,dec in [(1,0),(2,0),(3,0),(2,.5),(3,1.4)]:
    K=(1+dot)**deg+(np.exp(-dec*dist) if dec else 0)
    for r in [1e-6,.001,.02]:
     m,p=score(K,r);rows.append((*m,*p,nm,gf,deg,dec,r))
 for i in range(2):
  print('CV',i,sorted(rows,key=lambda t:t[i])[:12]); print('public',i,sorted(rows,key=lambda t:t[i+2])[:6]);
