from graphprobe import *
from itertools import combinations_with_replacement as cr
T=S['train']; t=S['eval_times']; h=T['traj']; y=T['y']; ai,af,B,C,D=ops(S); z=ai(T['x']); c,d=C(z),D(z); b=B(z); n=len(z)
def p(x,k): return np.column_stack([np.prod(x[:,i],axis=1) for i in cr(range(x.shape[1]),k)])
P=p(z,3); bc=p(b,3); q=p(z,2); Q=np.cos(2.4*z); dot=np.sum(c*d,axis=1,keepdims=True); cross=(c[:,:,None]*d[:,None,:]).reshape(n,16); cz=(c[:,:,None]*z[:,None,:]).reshape(n,16)
F={'d':d,'d+q':np.c_[d,q],'d+q4':np.c_[d,q,p(z,4)],'d+cd':np.c_[d,c,cross],'d+dd':np.c_[d,p(d,2)],'d+rat':np.c_[d,c/(1+d*d),d/(1+c*c)],'d+tanh':np.c_[d,np.tanh(d),np.tanh(c)],'d+cos4':np.c_[d,Q**2,p(Q,2)],'d+dnorm':np.c_[d,d/(1+(d*d).sum(1)[:,None]),(d*d).sum(1)],'d+cnorm':np.c_[d,d/(1+(c*c).sum(1)[:,None]),(c*c).sum(1)],'d+trig':np.c_[d,np.sin(d),np.cos(d)],'d+ratq':np.c_[d,q/(1+q*q)],'d+lin':np.c_[d,z],'d+allrat':np.c_[d,*[d/(1+k*d*d) for k in [.1,.2,.5,1]],*[d/(1+k*c*c) for k in [.1,.2,.5,1]]],'d+dot2':np.c_[d,dot**2]}
folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(n),i),i) for i in np.array_split(np.random.default_rng(123).permutation(n),5)]
def scale(G,tr): return [(a-a[tr].mean(0))/np.maximum(a[tr].std(0),1e-8) for a in G]
def time(r,kind):
 a,k,g=r; return np.array([-np.expm1(-a*t),t**k,{'exp':lambda:-np.expm1(-g*t),'rat':lambda:t/(1+g*t),'log':lambda:np.log1p(g*t)}[kind]()]).T

def model(G,tr,te,kind,pen):
 G=scale(G,tr); mu=h[tr].mean(0); Y=(h[tr]-mu).ravel();
 def mat(v,ix):return np.concatenate([(a[ix,:,None]*w[None,None,:]).transpose(0,2,1).reshape(len(ix)*len(t),-1) for a,w in zip(G,v.T)],axis=1)
 L=np.concatenate([np.full(a.shape[1],np.sqrt(pen)*(1 if k!=1 else .4)) for k,a in enumerate(G)]); reg=np.diag(L); YY=np.r_[Y,np.zeros(len(L))];
 def profile(r,out=False):
  v=time(r,kind); v/=v[-1]; A=np.vstack([mat(v,tr),reg]); w=fit(A,YY); return (mat(v,te)@w).reshape(len(te),-1)+mu if out else A@w-YY
 r=least_squares(profile,[1.8,.5,.55],bounds=([.5,.25,.015],[4,.8,4]),max_nfev=40); return profile(r.x,True),r.x,np.std(r.fun[:len(Y)])
rows=[]
for name,W in F.items():
 for kind in ['exp','rat','log']:
  for pen in [.001,.1]:
   G=[np.c_[c,dot],P,W]; e=[]
   for tr,te in folds:
    pred,r,ir=model(G,tr,te,kind,pen); e.append(np.abs(pred-h[te]).mean(axis=0));
   e=np.array(e); rows.append((e.mean(),e.mean(1).max(),name,kind,pen,e[0].mean(),e[:,-1].max(),r,ir))
for row in sorted(rows)[:22]:print(row)
for name,W in [('base',np.c_[c,d,dot,P,z]),('nod',np.c_[c,dot,P,z]),('cz',np.c_[c,d,dot,cz,z]),('b cubic',np.c_[c,d,dot,bc,z])]:
 er=[]
 for tr,te in folds:
  V=one(scale([W],tr)[0]); w=fit(V[tr],y[tr]); er.append(np.mean(abs(V[te]@w-y[te])));
 print('scalar',name,np.mean(er),max(er),er[0],np.std(y-one(W)@fit(one(W),y)))
