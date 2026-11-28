import numpy as np,pickle
from itertools import combinations
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; y=s['panel_c']['v2']; r=np.roll(u,-1,axis=1); o=np.roll(u,2,axis=1); p=u*r
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
one=np.ones((len(u),1))
bases={'lin':u,'sq':u*u,'cube':u**3,'cyc':p,'opp':(u*o)[:,:2],'cyc2':p*p,'cyc_a2b':p*u,'cyc_ab2':p*r,'cyc_a2+b2':p*(u*u+r*r),'cyc_a2-b2':p*(u*u-r*r),'triple':p*o}
funcs={}
for name,a in [('u',u),('cyc',p),('sum',u+r),('dif',u-r),('opp', (u*o)[:,:2])]:
 for kind,fun in [('sin',np.sin),('cos',np.cos),('atan',np.arctan),('tanh',np.tanh),('rat',lambda t:1/(1+t*t)),('ratodd',lambda t:t/(1+t*t))]: funcs[name+'_'+kind]=lambda q,a=a,fun=fun:fun(q*a)
for name,a in [('u',u),('r',r),('sum',u+r),('dif',u-r),('p',p),('o',o),('radius',np.sqrt(u*u+r*r))]:
 for kind,fun in [('rat',lambda t:1/(1+t*t)),('cos',np.cos),('exp',lambda t:np.exp(-t*t)),('root',lambda t:1/np.sqrt(1+t*t))]: funcs['p*'+name+'_'+kind]=lambda q,a=a,fun=fun:p*fun(q*a)
def scan(target, bs, fs):
 out=[]
 for bn,b in bs.items():
  for n,fn in fs.items():
   def design(q):return np.c_[one,b,fn(q)]
   def loss(q):
    a=design(q);return np.linalg.norm(a@ls(a,target)-target)
   g=np.geomspace(.08,8,24); i=np.argmin([loss(q) for q in g]); q=minimize_scalar(loss,bounds=(g[max(i-1,0)],g[min(i+1,len(g)-1)]),method='bounded').x
   a=design(q); h=a@np.linalg.pinv(a); e=(target-h@target)/np.maximum(1-h.diagonal()[:,None],1e-8)
   out.append((np.mean(abs(e)),loss(q),bn,n,q))
 return sorted(out)
bs={'+'.join(ks):np.column_stack([bases[k] for k in ks]) for m in (1,2,3) for ks in combinations(bases,m) if m<3 or ks[0]=='lin'};bs['empty']=np.empty((len(u),0));print('v2',*scan(y,bs,funcs)[:22],sep='\n')
y=s['panel_c']['v1']; print('v1',*scan(y,{'+'.join(k):np.column_stack([bases[i] for i in k]) for m in (0,1,2,3) for k in combinations(bases,m) if m and (m<3 or k[:2]==('lin','cube'))},{'atan':lambda q:np.arctan(q*u)})[:14],sep='\n')
