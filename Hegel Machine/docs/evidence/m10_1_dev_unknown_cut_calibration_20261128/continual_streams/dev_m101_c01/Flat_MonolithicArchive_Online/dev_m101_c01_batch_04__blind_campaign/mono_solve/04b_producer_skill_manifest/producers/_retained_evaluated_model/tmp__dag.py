exec(open('/tmp/structure.py').read().split('for k in')[0])
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
F=[(np.arange(56),np.arange(56,72)),*KFold(5,shuffle=True,random_state=12).split(x)]
def score(a,b):
 e=np.array([np.mean(abs(a[j]@fit(a[i],b[i])-b[j])) for i,j in F]);return np.round([e[0],e[1:].mean(),e[1:].max()],5)
u=pc['u']; mods={}
for k in ['v1','v2']:
 v=pc[k]; out=[]
 for name,fun,lo,hi in [('sin',np.sin,.1,6),('cos',np.cos,.1,6),('tanh',np.tanh,.1,5),('atan',np.arctan,.1,5),('rational',lambda w:w/(1+w*w),.05,3),('exp',np.exp,.1,4)]:
  for base in (0,1,2):
   def mat(w,f=fun,b=base):return np.column_stack([np.ones(len(w)),*([w] if b>0 else []),*([w*w] if b>1 else [])])
   def g(r,w,m=mat,f=fun):return np.c_[m(w),f(r*w)]
   def err(r):
    a=g(r,u);return rm(a@fit(a,v)-v)
   grid=np.linspace(lo,hi,100); j=np.argmin([err(r) for r in grid]); p=minimize_scalar(err,bounds=(grid[max(0,j-1)],grid[min(99,j+1)]),method='bounded').x
   a=g(p,u); c=fit(a,v); h=np.sum(a*np.linalg.pinv(a).T,axis=1); out.append((rm((a@c-v)/(1-h[:,None])),name,base,p,err(p),lambda w,g=g,p=p,c=c:g(p,w)@c))
 out.sort(key=lambda r:r[0]); print(k,[r[:-1] for r in out[:7]]); mods[k]=out[0][-1]
r=[]; A=poly(x,2); Y=z@np.linalg.svd(z,full_matrices=False)[2][0]
for aa,w in [('x',x),('inv',(x-bias)/scale),('fwd',x*scale+bias)]:
 for bb,b in [('direct',w),('B',poly(w,1)@cb),('Binv',(w-cb[0])@np.linalg.inv(cb[1:]))]:
  c=mods['v1'](b); d=mods['v2'](b)
  for name,base in [('x',poly(x,1)),('x2',A),('bdiag',np.c_[poly(b,1),b*b]),('b2',poly(b,2))]:
   for nn,extra in [('c',c),('d',d),('cd',np.c_[c,d]),('cd2',np.c_[c,d,c*c,d*d,c*d]),('ac',np.c_[c,d,*[b[:,j,None]*c for j in range(4)]]),('sin',np.c_[c,d,np.sin(b),np.cos(b)]),('rational',np.c_[c,d,c/(1+d*d),d/(1+c*c)])]:
    design=np.c_[base,extra]; sy=score(design,y); sz=score(design,Y)*.346; r.append((sy,sz,(aa,bb,name,nn)))
for k in (0,1):
 print('best',k); print(*sorted(r,key=lambda v:v[k][1]+.35*v[k][2])[:14],sep='\n')
