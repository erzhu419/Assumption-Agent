from grammar import *
from scipy.optimize import minimize_scalar

def panel(key,out,kind,per=False):
 p=s[key]; u=p['u']; v=p[out]
 fun={'sin':np.sin,'tanh':np.tanh,'cos':np.cos}[kind]
 def feat(x,w):
  a=aff(x)
  if kind=='cos': a=np.c_[a,quad(x)]
  return np.c_[a,fun(x*w)]
 def r(w):
  f=feat(u,w); return (f@ls(f,v)-v).ravel()
 obj=lambda w: np.mean(r(w)**2)
 opt=minimize_scalar(obj,bounds=(0.4,5),method='bounded',options={'xatol':1e-9})
 w=least_squares(r,np.full(4,opt.x),bounds=(0.3,6),gtol=1e-13).x if per else opt.x
 f=feat(u,w); c=ls(f,v); print(key,out,kind,'w',np.round(w,4),'err',np.round(fit(f,v),7),'norm',np.round(np.linalg.norm(c,axis=1),5))
 return lambda x:feat(x,w)@c

def cross(x):return x[:,0]*x[:,2]+x[:,1]*x[:,3]

def coords():
 funcs={}
 for key,out,kinds in [('panel_b','v',('tanh','sin')),('panel_c','v1',('tanh','sin')),('panel_c','v2',('cos',))]:
  for k in kinds:
   name={'panel_b':'b','panel_c':'c' if out=='v1' else 'd'}[key]+k[0]; funcs[name]=panel(key,out,k); panel(key,out,k,True)
 u=s['panel_a']['u']; v=s['panel_a']['v']; pars=[]
 for ui,vi in zip(u.T,v.T):
  i,j=np.triu_indices(len(u),1); g=np.median((vi[i]-vi[j])/(ui[i]-ui[j])); b=np.median(vi-g*ui); keep=abs(vi-g*ui-b)<.001; pars.append(ls(aff(ui[keep]),vi[keep]))
 b,g=np.array(pars).T; x=(s['train']['x']-b)/g
 z={'x':x,**{k:f(x) for k,f in funcs.items()}}
 for a,f in funcs.items():
  for b,q in list(z.items())[:6]:
   if b!='x': z[a+'('+b+')']=f(q)
 return z

def scan(z):
 Y=np.c_[s['train']['y'],s['train']['traj']]; records=[]
 idx=np.stack([np.arange(72),np.random.default_rng(310).permutation(72),np.random.default_rng(312).permutation(72)])
 def add(n,F):
  e=np.stack([np.mean(abs(Y[i[56:]]-F[i[56:]]@ls(F[i[:56]],Y[i[:56]])),axis=0) for i in idx]); er=np.sqrt(np.mean((Y-F@ls(F,Y))**2,axis=0)); records.append((n,*e[0,[0]],e[0,1:].mean(),e[:,0].max(),e[:,1:].mean(1).max(),er[0],np.mean(er[1:])))
 for n,x in z.items():
  for k,F in [('a',aff(x)),('q',np.c_[aff(x),quad(x)]),('cross',np.c_[aff(x),cross(x)])]:add(n+k,F)
 keys=list(z)[:6]
 for a,b in itertools.combinations(keys,2):
  x,y=z[a],z[b]; f=np.c_[aff(x),y]; add(a+'+'+b,f); add(a+'+'+b+'prod',np.c_[f,x*y,cross(x),cross(y)]); add(a+'+'+b+'cross',np.c_[f,cross(x),cross(y)])
 print('columns: name publicY publicT worstY worstT trainRmsY trainRmsT')
 for k in (1,2,3,4):
  print('rank by',k)
  for r in sorted(records,key=lambda r:r[k])[:9]:print(r[0],np.round(r[1:],5))
if __name__=='__main__':scan(coords())
