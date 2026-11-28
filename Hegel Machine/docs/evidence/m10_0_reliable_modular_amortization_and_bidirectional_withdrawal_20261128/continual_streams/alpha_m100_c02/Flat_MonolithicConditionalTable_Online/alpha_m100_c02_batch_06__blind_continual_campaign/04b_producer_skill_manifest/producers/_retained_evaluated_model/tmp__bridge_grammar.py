import pickle,numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.chebyshev import chebvander
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; L=c['true_channel_loads']; R=c['bridge_raw']; s=c['single_channel_sweeps']; sl=s['true_stress']; sr=s['cmd_stress']; ch=s['lab_id']; good=(abs(R).max(1)<4); sg=(abs(sr)<4); n=len(L)
def setup(j,m,sm):
 a=L[m,j]; b=L[m,(j+2)%4]; z=sm&(ch==j); return np.r_[a,sl[z]],np.r_[b,np.zeros(z.sum())],np.r_[R[m,j],sr[z]]
def basis(a,b,k,l,inter):
 U=chebvander(a/1.7,k); V=chebvander(b/1.7,l); A=np.c_[U,V[:,1:]]
 if inter: A=np.c_[A,*[U[:,i]*V[:,j] for i in range(1,inter+1) for j in range(1,inter+1)]]
 return A
def report(tag,errs):
 e=np.concatenate(errs); print(tag,'rms',np.sqrt(np.mean(e**2)).round(6),'abs q',np.quantile(abs(e),[.5,.9,.99,1]).round(6))
fold=np.arange(n)%5; mixed=np.count_nonzero(L,axis=1)==4
for k,l,inter in [(7,3,0),(9,4,0),(11,5,0),(9,4,1),(9,4,2),(11,5,2)]:
 errs=[]
 for f in range(5):
  test=good&mixed&(fold==f)
  for j in range(4):
   a,b,y=setup(j,good&~test,sg); q=np.linalg.lstsq(basis(a,b,k,l,inter),y,rcond=None)[0]; errs.append(basis(L[test,j],L[test,(j+2)%4],k,l,inter)@q-R[test,j])
 report((k,l,inter),errs)
for spec in ['tanh','arctan','rational','sin','tanh+rational','tanh+sin','tanh+poly']:
 def phi(x,p):
  w=np.exp(p[0]); funcs={'tanh':lambda:np.tanh(w*x),'arctan':lambda:np.arctan(w*x),'rational':lambda:x/(1+(w*x)**2),'sin':lambda:np.sin(w*x)}
  if spec in funcs: return np.c_[x,funcs[spec]()]
  extra={'tanh+rational':lambda:x/(1+(np.exp(p[1])*x)**2),'tanh+sin':lambda:np.sin(np.exp(p[1])*x),'tanh+poly':lambda:x**3}
  return np.c_[x,np.tanh(w*x),extra[spec]()]
 def design(a,b,p): return np.c_[np.ones(len(a)),phi(a,p),chebvander(b/1.7,5)[:,1:]]
 errs=[]; params=[]
 for f in range(5):
  test=good&mixed&(fold==f); data=[setup(j,good&~test,sg) for j in range(4)]
  def profile(p,mode=False):
   res=[]
   for a,b,y in data:
    A=design(a,b,p); q=np.linalg.lstsq(A,y,rcond=None)[0]; res.append(q if mode else A@q-y)
   return res if mode else np.concatenate(res)
  opts=[least_squares(profile,p,bounds=(-3,2),max_nfev=100) for p in ([[-.1],[.6],[1.1]] if spec.count('+')==0 or spec.endswith('poly') else [[a,b] for a in [-.1,.6,1.1] for b in [-.5,.5,1.]] )]; p=min(opts,key=lambda r:r.cost).x; params.append(np.exp(p));
  for j,q in enumerate(profile(p,True)): errs.append(design(L[test,j],L[test,(j+2)%4],p)@q-R[test,j])
 report(spec,errs); print('  shape',np.round(params,4))
for name in ['train','test']:
 d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset'][name]; a=np.vstack([d['hold_channel_raw'],d['past_channel_raw'].reshape(-1,4)]); print(name,'specimen raw ranges',a.min(0),a.max(0))
