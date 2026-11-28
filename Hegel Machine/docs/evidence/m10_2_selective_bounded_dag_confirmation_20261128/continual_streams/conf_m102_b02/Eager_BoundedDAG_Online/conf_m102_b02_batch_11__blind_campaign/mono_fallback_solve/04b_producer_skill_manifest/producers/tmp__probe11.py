import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; t=s['eval_times']; fit=lambda A,B:np.linalg.lstsq(A,B,rcond=1e-9)[0]; D=lambda x:np.c_[np.ones(len(x)),x]
def poly(u,d): return np.column_stack([np.prod(u[:,i],axis=1) for j in range(d+1) for i in combinations_with_replacement(range(4),j)])
def af(u):return D(np.c_[u,u*np.roll(u,-1,axis=1),u**3])
a=s['panel_a']; u,v=a['u'],a['v']; keep=np.ones(len(u),bool)
for _ in range(8):
 c=fit(D(u[keep]),v[keep]);r=np.linalg.norm(D(u)@c-v,axis=1);keep=r<max(.001,3*np.median(r))
print('A resid',np.sort(r), 'coef',c);z=np.linalg.solve(c[1:].T,(x-c[0]).T).T
w=fit(af(u[keep]),v[keep]);print('A nonlin',np.linalg.norm(af(u[keep])@w-v[keep]),np.linalg.norm(w[5:]));
b=s['panel_b'];w=fit(D(b['u']),b['v']); b=D(z)@w
print('B',w)
C=s['panel_c']; U=C['u']; cs=[]
for k in ['v1','v2']:
 V=C[k]; specs={'aff':D,'af':af,'quad':lambda u:poly(u,2),'cubdiag':lambda u:D(np.c_[u,u**2,u**3]), 'tanh':lambda u:D(np.tanh(u)),'erf':lambda u:D(__import__('scipy').special.erf(u)),'sin':lambda u:D(np.sin(u)),'asinh':lambda u:D(np.arcsinh(u))}
 for name,F in specs.items():
  f=F(U); e=f@fit(f,V)-V; h=(f*np.linalg.pinv(f).T).sum(1); print(k,name,np.sqrt((e**2).mean()),np.sqrt(((e/(1-h[:,None]))**2).mean()))
 cs.append(af(z)@fit(af(U),V))
for mode in range(4):
 def F(p):
  g=-np.expm1(-t[:,None]*p)
  if mode==0:return g
  if mode==1:return np.c_[g[:,0],t*np.exp(-p[1]*t)]
  if mode==2:return np.c_[g[:,0],np.exp(-p[1]*t)]
  return np.c_[g[:,0],t]
 def err(p):
  f=F(p);return (f@fit(f,T.T)-T.T).ravel()
 q=least_squares(err,[.45,2.15],bounds=([.001,.001],[10,15])).x;print('time',mode,q,np.abs(err(q)).mean());
 if mode==0: f=F(q); amps=fit(f,T.T).T
print('corr',np.corrcoef(np.c_[z,b,*cs,y,amps].T)[-3:,:].round(3))
Zs={'x':x,'z':z,'b':b,'c1':cs[0],'c2':cs[1],'zb':np.c_[z,b],'zc':np.c_[z,*cs], 'bc':np.c_[b,*cs]}
Y=np.c_[y,amps]; rng=np.random.default_rng(110); idx=np.r_[np.arange(len(y))[None,:],np.array([rng.permutation(len(y)) for _ in range(4)])]; out=[]
for name,P in Zs.items():
 for form in ('lin','asinh0.5','asinh1.5','asinh3.5','tanh1','tanh2','joint'):
  Z=P if form=='lin' else (np.arcsinh(float(form[5:])*P) if form.startswith('asinh') else np.tanh(float(form[4:])*P) if form.startswith('tanh') else np.c_[.4*P,np.arcsinh(2*P)])
  g=Z@Z.T/Z.shape[1]; dist=np.maximum(np.diag(g)[:,None]+np.diag(g)-2*g,0)
  for deg in (1,2,3,4,5):
   for gam in (0,.15,.5,1.5,4):
    K=(1+g)**deg+(np.exp(-gam*dist) if gam else 0)
    for ridge in (1e-7,1e-5,.0003,.003):
     er=[]
     for p in idx:
      i,j=p[:52],p[52:]; pred=K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+ridge*np.eye(52),Y[i]);er.append([np.abs(pred[:,0]-y[j]).mean(),np.abs(pred[:,1:]@f.T-T[j]).mean()])
     er=np.array(er);out.append((er,(name,form,deg,gam,ridge)))
for i in (0,1):
 print('best',i)
 for e,k in sorted(out,key=lambda ek:ek[0][:,i].mean())[:15]:print(k,e[:,i].round(5),'mean',e[:,i].mean())
np.savez('/tmp/d11.npz',x=x,z=z,b=b,c1=cs[0],c2=cs[1],y=y,T=T,t=t,f=f,amps=amps,idx=idx)
