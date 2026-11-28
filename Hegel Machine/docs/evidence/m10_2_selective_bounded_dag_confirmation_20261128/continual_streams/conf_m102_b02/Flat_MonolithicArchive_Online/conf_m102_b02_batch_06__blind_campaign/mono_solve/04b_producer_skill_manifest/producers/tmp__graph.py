import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def I(x):return np.column_stack((np.ones(len(x)),x))
def ls(x,y):return np.linalg.lstsq(x,y,rcond=1e-9)[0]
def feat(a):
 p=np.roll(a,-1,axis=1);return np.stack((np.ones_like(a),a,p,a*p,a**3),axis=-1)
u,v=(s['panel_a'][k] for k in ('u','v')); d=feat(u);keep=np.ones(len(u),bool)
for _ in range(6):
 C=np.array([ls(d[keep,i],v[keep,i]) for i in range(4)]);keep=np.max(abs(np.einsum('nik,ik->ni',d,C)-v),axis=1)<.75
z=np.array([least_squares(lambda a:np.einsum('ik,ik->i',feat(a[None])[0],C)-r,r).x for r in np.r_[s['train']['x'],s['test']['x']]])
b=s['panel_b']; c=s['panel_c'];u=b['u'];v=b['v']
def fitop(u,v,design,init):
 def fun(a):
  d=design(u,a);return (d@ls(d,v)-v).ravel()
 f=least_squares(fun,init,max_nfev=160); a=f.x;k=ls(design(u,a),v); return lambda x:design(x,a)@k,a,k,np.mean(abs(f.fun))
ops={};
for name,fn,a in [('lin',lambda x,a:I(x),[]),('tanh',lambda x,a:I(np.c_[x,np.tanh(x*np.exp(a[0]))]),[.35]),('sat',lambda x,a:I(np.c_[x,x/(1+np.exp(a[0])*x*x)]),[-.2]),('sin',lambda x,a:I(np.c_[x,np.sin(x*np.exp(a[0]))]),[1.05])]:
 if not a:ops[name]=lambda x:I(x)@ls(I(u),v);continue
 ops[name],a,k,e=fitop(u,v,fn,a);print('B',name,a,e,'coef',np.round(k,3))
pairs=np.array(list(itertools.combinations_with_replacement(range(4),2)))
def quad(x):return x[:,pairs[:,0]]*x[:,pairs[:,1]]
def pair(x):return x*np.roll(x,-1,axis=1)
qop=lambda x:pair(x)@ls(I(pair(c['u'])),c['v1'])[1:]
for n,d in [('v1',I(pair(c['u']))),('v2',I(np.c_[c['u'],quad(c['u'])]))]:print(n,np.round(ls(d,c[n]),4))
rop=lambda x:I(np.c_[x,quad(x)])@ls(I(np.c_[c['u'],quad(c['u'])]),c['v2'])
y=s['train']['y'];tr=s['train']['traj'];
def scores(d,target=y):
 d=d[:72]; a=d[:52]; coef=ls(a,target[:52]);return np.abs(d[52:]@coef-target[52:]).mean(),np.abs(a@coef-target[:52]).mean()
def outer(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
if __name__=='__main__':
 results=[]
 for bn,bf in ops.items():
  w=bf(z)
  for qn,q in [('z',qop(z)),('w',qop(w))]:
   for rn,r in [('z',rop(z)),('w',rop(w)),('q',rop(q))]:
    V={'z':z,'w':w,'q':q,'r':r}; blocks=list(V.items());opts=[('',np.empty((len(z),0)))]
    for an,a in blocks:
     for dn,d in blocks:
      if an==dn:continue
      opts.extend([(an+dn+'dot',(a*d).sum(1)[:,None]),(an+dn+'diag',a*d),(an+dn+'outer',outer(a,d))]);
    for n in range(1,5):
     for base in itertools.combinations(blocks,n):
      for cn,cx in opts:
       d=I(np.column_stack([p[1] for p in base]+[cx])); yn,yt=scores(d);tn,tt=scores(d,tr);results.append((yn,yt,tn,tt,bn,qn,rn,''.join(p[0] for p in base),cn))
 print('Y best',*sorted(results)[:15],sep='\n');print('T best',*sorted(results,key=lambda r:r[2])[:10],sep='\n');print('z ranges',z[:72].min(0),z[:72].max(0),z[72:].min(0),z[72:].max(0))
