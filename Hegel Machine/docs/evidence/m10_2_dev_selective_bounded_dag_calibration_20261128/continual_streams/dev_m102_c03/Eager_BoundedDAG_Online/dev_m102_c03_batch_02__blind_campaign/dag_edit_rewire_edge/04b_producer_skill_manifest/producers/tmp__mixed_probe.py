exec(open('/tmp/probe_rewire.py').read().split('for key in')[0])
from scipy.special import erf,expit
from scipy.linalg import qr
from sklearn.linear_model import lars_path
import warnings
warnings.filterwarnings('ignore')
funcs={'atan':np.arctan,'tanh':np.tanh,'asinh':np.arcsinh,'erf':erf,'soft':lambda x:x/(1+abs(x)),'alg':lambda x:x/np.sqrt(1+x*x),'log':lambda x:np.sign(x)*np.log1p(abs(x)),'sin':np.sin}
models=[]
for nm,fn in funcs.items():
 for linear in (False,True):
  for shift in (False,True):
   def tra(x,par):return fn(x*np.exp(par[:4])+(par[4:] if shift else 0))
   def design(x,par):return aff(np.c_[x,tra(x,par)] if linear else tra(x,par))
   def res(par):
    f=design(u,par);return (f@fit(f,p['v1'])-p['v1']).ravel()
   a=least_squares(res,np.r_[np.log([2.]*4),np.zeros(4 if shift else 0)],bounds=(np.r_[[-3]*4,[-1]*(4 if shift else 0)],np.r_[[2]*4,[1]*(4 if shift else 0)]),max_nfev=100)
   models.append((np.std(a.fun),nm,linear,shift,a.x,design(z,a.x)@fit(design(u,a.x),p['v1']),tra(z,a.x)))
for rr in sorted(models,key=lambda a:a[0])[:12]:print('panel v1',rr[:5])
rr=min(models,key=lambda a:a[0]); S=rr[5]; U=rr[6]; V=np.cos(2.4*z); C=aff(V)@fit(aff(np.cos(2.4*u)),p['v2']); B=aff(z)@R; q=-np.expm1(-t[:,None]*[.4125,1.80995]); A=np.c_[Y[:,0],fit(q,Y[:,1:].T).T]; print('C2 mix',fit(aff(np.cos(2.4*u)),p['v2']).round(3));print('corr',np.corrcoef(A.T));
D={};
def add(n,x): D.setdefault(n,x)
blocks={'L':z,'S':S,'U':U,'V':V,'C':C,'B':B}
for nm,w in blocks.items():
 for i,v in enumerate(w.T):add(nm+str(i),v)
base=list(D.items())
for i,(n,x) in enumerate(base):
 for m,y in base[i:]:
  add(n+'*'+m,x*y)
  if n==m:continue
  if n[0]==m[0] or n[1]==m[1]:
   for k in [.5,1,2]:
    add(f'{n}*{m}/r{k}',x*y/(1+k*(x*x+y*y)))
    for c in [-1,-.5,.5,1]:
     for fn,func in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('gauss',lambda v:np.exp(-v*v))]:add(f'{fn}{k}{n}+{c}{m}',func(k*(x+c*y)))
for nm,w in blocks.items():
 for i,x in enumerate(w.T):
  for k in [0.5,1.,1.5,1.8,2.,2.4,2.8,3.,3.5,4.]:
   for fn,func in {**funcs,'cos':np.cos,'gauss':lambda x:np.exp(-x*x)}.items():add(f'{fn}{k}{nm}{i}',func(k*x))
 for ix in combinations(range(4),3):add(nm+str(ix),np.prod(w[:,ix],axis=1))
F=np.column_stack(list(D.values())); names=np.array(list(D)); n=52
for key,ix in [('all',np.ones(len(names),bool)),('base_prods',np.array([len(s)==2 or ('*' in s and len(s)<7) for s in names])),('LSVC',np.array([len(s)==2 or ('*' in s and len(s)<7 and all(ch not in s for ch in 'BU')) for s in names]))]:
 f=F[:,ix]; nm=names[ix];f=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-10)
 for j in range(3):
  target=A[:,j]; y=target[:n]-target[:n].mean(); al,ids,coef=lars_path(f[:n],y,method='lasso',max_iter=160); results=[]
  for k in range(1,coef.shape[1]):
   ii=np.flatnonzero(abs(coef[:,k])>1e-8)
   if len(ii)>28:continue
   g=aff(f[:,ii]); b=fit(g[:n],target[:n]); err=np.mean(abs(g[n:]@b-target[n:])); results.append((err,k,len(ii),np.std(g[:n]@b-target[:n]),ii,b))
  r=min(results,key=lambda a:a[0]); print(key,j,np.round(r[:4],5),list(zip(nm[r[4]],r[5][1:].round(3))))
for j in range(3):
 print('top raw corr',j, sorted(((abs(np.corrcoef(F[:,k],A[:,j])[0,1]),nm) for k,nm in enumerate(names)),reverse=True)[:15]);
