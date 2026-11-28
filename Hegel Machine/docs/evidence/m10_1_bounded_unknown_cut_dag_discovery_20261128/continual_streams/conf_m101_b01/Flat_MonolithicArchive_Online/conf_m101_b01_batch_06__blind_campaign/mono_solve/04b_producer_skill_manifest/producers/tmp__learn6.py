import numpy as np,pickle,sys
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
sys.path.insert(0,'/holdout_task'); from batch_solver import _branches
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; Q=_branches(s,tr['x']); b=Q[1]; x=Q[0];
for Z in (x,b):
 for k in (1,2,3):
  v=np.roll(Z,k,axis=1); Q.extend([Z*v,Z/(1+v*v)]);
c=s['panel_c']; cw=np.linalg.lstsq(np.c_[np.ones(len(c['u'])),c['u']*np.roll(c['u'],-1,axis=1)],c['v1'],rcond=None)[0]; print('C1',cw.round(3))
folds=[(np.setdiff1d(np.arange(52),j),j) for j in np.array_split(np.random.default_rng(624).permutation(52),4)]+[(np.arange(52),np.arange(52,72))]
def model(i,j,groups,z,typ,individual,reg):
 xx=np.column_stack(groups); xx=(xx-xx[i].mean(0))/np.maximum(xx[i].std(0),1e-8); n=len(i); nf=xx.shape[1];
 if individual: group=np.arange(nf)
 else: group=np.repeat(np.arange(len(groups)),[a.shape[1] for a in groups]);
 k=group.max()+1; diff=(xx[:,None,:]-xx[None,:,:])**2/nf; dot=xx[:,None,:]*xx[None,:,:]/nf
 d=np.stack([diff[:,:,group==a].sum(2) for a in range(k)]); dot=np.stack([dot[:,:,group==a].sum(2) for a in range(k)]);
 ii=np.ix_(i,i); di=d[:,i][:,:,i]; gi=dot[:,i][:,:,i]; m=z[i].mean(); yy=(z[i]-m)/z[i].std(); I=np.eye(n)
 def kern(params,d,g,jac=False):
  w=np.exp(params[:k]); a,l,noise=np.exp(params[k:]); D=np.tensordot(w,d,axes=1); G=np.tensordot(w,g,axes=1);
  if typ=='rq': f=1/(1+D); fp=-f*f
  elif typ=='rbf': f=np.exp(-D); fp=-f
  else: f=(1+G)**2; fp=0
  K=a*f+l*G
  if not jac:return K
  K=K+noise*I
  der=(a*fp*d+l*g) if typ!='poly' else (2*a*(1+G)+l)*g
  return K,np.concatenate((w[:,None,None]*der,np.stack([a*f,l*G,noise*I])))
 def nll(p):
  K,A=kern(p,di,gi,True); f=cho_factor(K); alpha=cho_solve(f,yy); inv=cho_solve(f,I)-alpha[:,None]*alpha; return .5*yy@alpha+np.log(f[0].diagonal()).sum()+reg*np.sum((p[:k]+1)**2), .5*np.einsum('ij,aij->a',inv,A)+np.r_[2*reg*(p[:k]+1),0,0,0]
 opt=minimize(nll,np.r_[np.full(k,-1.),0.,-3.,-7.],jac=True,method='L-BFGS-B',bounds=[(-8,4)]*k+[(-8,7),(-10,5),(-13,0)],options={'maxiter':100}); p=opt.x; K=kern(p,d,dot); out=m+z[i].std()*(K[np.ix_(j,i)]@cho_solve(cho_factor(K[ii]+np.exp(p[-1])*I),yy)); return out,p
for gs in [Q[:4],Q[:4]+Q[4:8],Q,Q[:2],[Q[1],Q[2],Q[3]]]:
 for typ in ('rq','rbf','poly'):
  for ix,reg in [(False,.15),(True,.2),(True,.8)]:
   rec=[]; params=[]
   for out,z in enumerate([y,T[:,-1]]):
    e=[]
    for i,j in folds:
     h,p=model(i,j,gs,z,typ,ix,reg); e.append(np.mean(abs(h-z[j]))); 
    rec+=list(np.round([np.mean(e[:4]),e[-1]],5)); params.append(np.round(p,1))
   print(len(gs),typ,ix,reg,rec, flush=True)
   if rec[1]<.040 or rec[0]<.065: print('weights',params,flush=True)
