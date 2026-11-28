import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=52
D=lambda x:np.c_[np.ones(len(x)),x]; L=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
def setup(s,x):
 u,v=s['panel_a']['u'],s['panel_a']['v']; i,j=np.triu_indices(len(u),1);z=[]
 for a,b,c in zip(u.T,v.T,x.T):
  dx=a[i]-a[j];g=abs(dx)>1e-9;m=np.median((b[i[g]]-b[j[g]])/dx[g]); r=b-m*a; g=abs(r-np.median(r))<max(.001,6*np.median(abs(r-np.median(r))));k,h=L(D(a[g]),b[g]);z.append((c-k)/h)
 z=np.array(z).T
 u,v=s['panel_b']['u'],s['panel_b']['v']; B=L(D(u),v)
 p=s['panel_c'];u=p['u']; F=[]
 for key,f in [('v1',lambda q,k:D(np.c_[q,q/(1+k*abs(q))])),('v2',lambda q,k:D(np.c_[q,q*np.roll(q,-1,axis=1),np.arctan(k*q)]))]:
  y=p[key]; obj=lambda k:np.mean((f(u,k)@L(f(u,k),y)-y)**2);k=minimize_scalar(obj,bounds=(.2,2),method='bounded').x;coef=L(f(u,k),y);F.append(lambda q,f=f,k=k,coef=coef:f(q,k)@coef)
 return z,B,*F

def groups(s,x):
 z,B,P,Q=setup(s,x); b=D(z)@B; G={};add=lambda k,v:G.update({k:v.reshape(len(z),-1)}); adj=lambda a,i:a*np.roll(a,i,1)
 def single(k,v):
  add(k,v)
  for h in (.22,.55,1.,1.5,2.15,3.):
   for nm,f in [('sin',lambda a:np.sin(h*a)),('cos',lambda a:np.cos(h*a)),('tanh',lambda a:np.tanh(h*a)),('sat',lambda a:a/(1+h*abs(a))),('rat',lambda a:a/(1+h*a*a)),('arc',lambda a:np.arctan(h*a)),('soft',lambda a:np.logaddexp(0,h*a))]:add(f'{nm}{h}_{k}',f(v))
  for k2,u in [('sq',v*v),('cu',v**3),('adj',adj(v,1)),('opp',adj(v,2)[:,:2]),('trip',v*np.roll(v,1,1)*np.roll(v,2,1))]:add(k2+'_'+k,u)
 for k,w in [('z',z),('b',b),('bt',z@B[1:].T),('z+.22b',z+.22*b),('b+.22z',b+.22*z),('tz',np.tanh(2.15*z)),('rz',z/(1+.22*z*z)),('tb',np.tanh(2.15*b)),('rb',b/(1+.22*b*b))]:
  single(k,w);p,q=P(w),Q(w);single('p_'+k,p);single('q_'+k,q)
  for nm,a,c in [('pq',p,q),('wp',w,p),('wq',w,q)]:
   for j in range(4):add(f'{nm}{j}_{k}',a*np.roll(c,j,1))
   add(f'sum{nm}_{k}',np.sum(a*c,axis=1))
   add(f'rat{nm}_{k}',a/(1+.22*c*c));add(f'ratrev{nm}_{k}',c/(1+.22*a*a));add(f'th{nm}_{k}',a*np.tanh(2.15*c));add(f'threv{nm}_{k}',c*np.tanh(2.15*a));add(f'thsum{nm}_{k}',np.tanh(2.15*(a+c)));add(f'thdiff{nm}_{k}',np.tanh(2.15*(a-c)))
 return G

def scan(G,Y,n):
 names=list(G); g=[]
 for f in G.values():
  f=f-f[:n].mean(0);g.append(f/np.maximum(f[:n].std(0),1e-8))
 y=Y-Y[:n].mean(0); out=[]; base=np.ones((len(Y),1)); chosen=[]
 for it in range(11):
  vals=[]
  for j,f in enumerate(g):
   if j in chosen:continue
   a=np.c_[base,f];ai=np.linalg.pinv(a[:n],rcond=1e-7);res=y[:n]-a[:n]@(ai@y[:n]);h=np.sum(a[:n]*ai.T,axis=1); cv=np.mean(abs(res/np.maximum(.02,1-h)[:,None]));te=np.mean(abs(a[n:]@(ai@y[:n])-y[n:]));vals.append((cv,te,j))
  vals.sort(); print(it,[(round(a,5),round(b,5),names[j]) for a,b,j in vals[:(12 if it<2 else 4)]],flush=True);chosen.append(vals[0][2]);base=np.c_[base,g[chosen[-1]]]
 return [names[i] for i in chosen]
if __name__=='__main__':
 G=groups(s,tr['x']);print('scalar');scan(G,tr['y'][:,None],n);print('terminal');scan(G,tr['traj'][:,-1:],n); print('contrast');scan(G,(tr['traj'][:,0]-.42*tr['traj'][:,-1])[:,None],n)
