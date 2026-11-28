from discover import *
from itertools import combinations
p=S['panel_b'];u,v=p['u'],p['v'];tr=S['train'];z=coords(S,tr['x']);t=S['eval_times'];T=tr['traj'];
def hinge(x,q):return np.sign(x)*np.maximum(abs(x)-q[0],0)**q[1]
def bdesign(x,q,mode):
 h=hinge(x,q);return D(h if mode==0 else h+q[2]*x**3 if mode==1 else np.c_[h,x**3] if mode==2 else np.c_[h,x,x**3])
Bs=[]
for mode in range(4):
 def res(q):
  F=bdesign(u,q,mode);return (F@fit(F,v)-v).ravel()
 o=least_squares(res,[.35,1.5,0],bounds=([.15,.4,-2],[.6,3,2]));F=bdesign(u,o.x,mode);Bs.append(bdesign(z,o.x,mode)@fit(F,v));print('hinge',mode,o.x,np.std(o.fun),np.linalg.svd(fit(F,v))[1]);
b=Bs[1]; w=np.c_[z,b]; time=np.c_[-np.expm1(-1.7936*t),-np.expm1(-.4178*t)];Y=np.c_[tr['y'],T@np.linalg.pinv(time).T];
from sklearn.model_selection import KFold
splits=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=921).split(z));
def validate(get):
 E=[]
 for i,j in splits:
  h=get(i,j);E.append([np.mean(abs(h[:,0]-Y[j,0])),np.mean(abs(h[:,1:]@time.T-T[j])),np.mean(abs(h[:,1]-Y[j,1])),np.mean(abs(h[:,2]-Y[j,2]))]);
 return np.array(E)
# Algebraic basis after hinge recovery.
C=bases(S,z);c=C['c20'];c1=C['c1'];
groups={'z':z,'b':b,'bb':b*np.roll(b,-1,axis=1),'bq':P(b,2)[:,5:],'zq':P(z,2)[:,5:],'cc':np.c_[c,c1],'zb':(z[:,:,None]*b[:,None,:]).reshape(-1,16),'sb':np.sin(1.25*b),'tb':np.tanh(1.25*b),'sz':np.sin(1.25*z),'tz':np.tanh(1.25*z),'cb':b**3,'sum':np.tanh(1.25*(z+b)),'dif':np.tanh(1.25*(z-b)),'trip':np.prod(b[:,list(combinations(range(4),3))],axis=2),'ztrip':np.prod(z[:,list(combinations(range(4),3))],axis=2)}
for k in range(3):
 sel=['z','b'];print('target',k)
 for it in range(5):
  cand=[]
  for g in groups.keys()-set(sel):
   F=D(np.column_stack([groups[f] for f in sel+[g]]));
   if F.shape[1]>40:continue
   e=np.array([np.mean(abs(F[j]@fit(F[i],Y[i,k])-Y[j,k])) for i,j in splits]);cand.append((e.mean(),e[0],g))
  if not cand:break
  q=min(cand);sel.append(q[-1]);print(q)
# Profile smooth ridge functions atop physics bases.
rng=np.random.default_rng(32911);out=[]
for base in ['z','b','z b','z b bb','z b bq']:
 F=D(np.column_stack([groups[g] for g in base.split()]));
 for name,H in [('z',z),('b',b),('w',w)]:
  for non,fun in [('tanh',np.tanh),('sin',np.sin),('rat',lambda x:x/(1+x*x)),('sqrt',lambda x:np.sqrt(1+x*x)),('cos',np.cos)]:
   def run(i,j):
    hs=H/np.maximum(H[i].std(0),.01);R=Y[i]-F[i]@fit(F[i],Y[i]);g=fit(D(hs[i]),Y[i]); ys=[]
    for k in range(2):
     def calc(q,I):return np.c_[F[I],fun(q[0]+hs[I]@q[1:])];
     def res(q):
      f=calc(q,i);return f@fit(f,Y[i,k])-Y[i,k]
     local=np.random.default_rng(40);v0=g[1:,k];v0=v0/max(np.linalg.norm(v0),1e-8);st=[np.r_[a,v0*p] for a,p in [(0,.6),(.6,1.3),(1.5,.7)]]+[local.normal(0,.7,hs.shape[1]+1) for _ in range(5)];o=min((least_squares(res,s,max_nfev=60,bounds=(-7,7)) for s in st),key=lambda p:p.cost);ys.append(calc(o.x,j)@fit(calc(o.x,i),Y[i,k]));
    ys.append(F[j]@fit(F[i],Y[i,2]));return np.array(ys).T
   e=validate(run);out.append((e.mean(0),e[0],base,name,non));
for k in range(2):
 print('ridge best',k,*sorted(out,key=lambda r:r[0][k])[:8],sep='\n')
