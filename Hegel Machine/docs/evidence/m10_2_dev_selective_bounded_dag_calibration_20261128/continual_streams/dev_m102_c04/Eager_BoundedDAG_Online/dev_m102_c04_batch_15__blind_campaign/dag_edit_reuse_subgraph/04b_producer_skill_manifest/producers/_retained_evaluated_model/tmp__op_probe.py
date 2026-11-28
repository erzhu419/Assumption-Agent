exec(open('/tmp/spatial_probe.py').read().split('for mode')[0])
from scipy.optimize import minimize_scalar
ls=lambda f,y: np.linalg.lstsq(f,y,rcond=1e-10)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
C=s['panel_c']; x=C['u']; y=C['v2']; pairs=list(combinations(range(4),2)); cross=np.column_stack([x[:,i]*x[:,j] for i,j in pairs]); cub=np.column_stack([x[:,i]*x[:,j]*x[:,h] for i,j,h in combinations(range(4),3)])
def cv(f):
 r=y-f@ls(f,y); q=np.linalg.svd(f,full_matrices=False)[0]; v=(r/np.maximum(1e-8,1-(q*q).sum(1))[:,None]); return np.linalg.norm(r),np.linalg.norm(v)
for expr,k in [('quad',x*x),('cross',cross),('fullquad',np.c_[x*x,cross]),('cubic',x**3),('cube_and_square',np.c_[x*x,x**3]),('triple',cub),('quad+triple',np.c_[x*x,cross,cub]),('cyclic',x*np.roll(x,-1,axis=1))]:
 f=np.c_[aff(x),k]; print(expr,cv(f),np.round(ls(f,y)[5:],3))
blocks={'raw':x,'pairdiff':np.column_stack([x[:,i]-x[:,j] for i,j in pairs]),'pairsum':np.column_stack([x[:,i]+x[:,j] for i,j in pairs]),'cyclicprod':x*np.roll(x,-1,axis=1),'pairprod':cross,'cyclediff':x-np.roll(x,-1,axis=1),'cyclesum':x+np.roll(x,-1,axis=1),'pairquad':x*x+np.roll(x,-1,axis=1)**2}
forms={'even':lambda d,k:d*d/(1+k*d*d),'odd':lambda d,k:d**3/(1+k*d*d),'sat':lambda d,k:d/(1+k*abs(d)),'sine':lambda d,k:np.sin(k*d),'cosine':lambda d,k:np.cos(k*d),'tanh':lambda d,k:np.tanh(k*d),'literal':lambda d,k:d,'abs':lambda d,k:abs(d)}
results=[]
for bn,d in blocks.items():
 for fn,form in forms.items():
  for base in [aff(x),np.c_[aff(x),x*x]]:
   def des(k):return np.c_[base,form(d,k)]
   grid=np.linspace(.015,5,26); j=np.argmin([cv(des(k))[0] for k in grid]); out=minimize_scalar(lambda k:cv(des(k))[0],bounds=(grid[max(j-1,0)],grid[min(j+1,25)]),method='bounded'); results.append((cv(des(out.x))[1],out.fun,bn,fn,base.shape[1],out.x))
print('top grammar',*sorted(results)[:22],sep='\n')
print('cube variants and pair-direction search')
for q in [np.roll(x,-1,axis=1),np.roll(x,1,axis=1),np.roll(x,2,axis=1)]:
 for label,make in [('prod rational',lambda k:x*q/(1+k*x*x)),('prod rational n',lambda k:x*q/(1+k*q*q)),('prod rational both',lambda k:x*q/(1+k*(x*x+q*q))),('square+cross',lambda k:(x+k*q)**2),('rational mixed square',lambda k:(x+q)**2/(1+k*x*x)),('square rational neighbor',lambda k:x*x/(1+k*q*q)),('square rational both',lambda k:x*x/(1+k*(x*x+q*q))),('square+prod',lambda k:np.c_[x*x/(1+k*x*x),x*q]),('quad+cubic mixed',lambda k:np.c_[x*x,x*q,x*x*q])]:
  f=lambda k:np.c_[aff(x),make(k)]; opt=minimize_scalar(lambda k:cv(f(k))[0],bounds=(.00001,5),method='bounded'); print(label,round(opt.x,5),np.round(cv(f(opt.x)),6))
