from cal import *
from panels import poly
from itertools import combinations_with_replacement, combinations
x=s['train']['x']; z=invert(x); Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(z)
t=s['eval_times']; print('trajectory singular values',np.linalg.svd(Y[:,1:],compute_uv=False)); print('study keys',s.keys(),'train keys',s['train'].keys()); print('train summary',s['train'].get('summary')); print('telemetry',s.get('telemetry_vec'))
folds=[(np.arange(52),np.arange(52,n))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(72).permutation(n),5)]
F={'z':z,'b':forward(z),'bi':backward(z),'q':q(z),'h':h(z),'c':lin(q(z))@C,'d':lin(h(z))@D,'x':x}
np.set_printoptions(precision=5,suppress=True)
def scores(m,reg=0):
 out=[]
 for a,j in folds:
  mu=m[a].mean(0); sd=np.maximum(m[a].std(0),1e-10); f=(m[a]-mu)/sd; g=(m[j]-mu)/sd; beta=np.linalg.lstsq(f,Y[a]-Y[a].mean(0),rcond=1e-8)[0] if not reg else np.linalg.solve(f.T@f+reg*len(a)*np.eye(f.shape[1]),f.T@(Y[a]-Y[a].mean(0))); e=abs(g@beta+Y[a].mean(0)-Y[j]); out.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.r_[out[0],np.max(out,axis=0)]
base=poly(z,2)[0][:,1:]; ext={}
for k,g in F.items():
 for nm,fun in [('3',lambda x:x**3),('4',lambda x:x**4),('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('sat',lambda x:x/(1+x*x)),('sat2',lambda x:x*x/(1+x*x)),('lc',lambda x:np.logaddexp(x,-x)-np.log(2))]: ext[k+nm]=fun(g)
for k,l in combinations_with_replacement(F,2):
 if k==l: continue
 a,b=F[k],F[l]; prod=(a[:,:,None]*b[:,None,:]).reshape(n,-1); ext[k+'*'+l]=prod; ext[k+'@'+l]=a*b; ext[k+'sum'+l]=prod[:,[1,6,11,12]]; ext[k+'/'+l]=a/(1+b*b)
for k in ('z','b','bi','c','d'):
 g=F[k]; ext[k+'trip']=np.column_stack([g[:,j].prod(1) for j in combinations(range(4),3)])
for base_id,base_ in [('p2',base),('p2+h',np.c_[base,h(z)]),('p3',poly(z,3)[0][:,1:]),('compact',np.c_[z,q(z),h(z)])]:
 print('base',base_id,scores(base_,1e-5)); out=[]
 for k,g in ext.items():
  v=np.c_[base_,g]; e=scores(v,1e-5); a=np.c_[np.ones(n),v]; train=np.mean(abs(a@np.linalg.lstsq(a,Y,rcond=1e-8)[0]-Y),axis=0); out.append((k,e,train[0],train[1:].mean()))
 for j in (0,1,2,3):
  print('top',j,[(k,e.round(4),round(a,5),round(b,5)) for k,e,a,b in sorted(out,key=lambda v:v[1][j])[:6]])
# Fit a shared time subspace and check whether remaining trajectory components track feature amplitudes.
for k in (1,2,3,4,5):
 U,V,W=np.linalg.svd(Y[:,1:],full_matrices=False); print('temporal rank',k,'MAE',np.mean(abs(Y[:,1:]-(U[:,:k]*V[:k])@W[:k])),'times',W[k-1]);
np.savez('/tmp/grammar.npz',Y=Y,z=z,**{'f_'+k:v for k,v in F.items()})
