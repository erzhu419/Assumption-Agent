import pickle, numpy as np
from itertools import combinations_with_replacement, combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=160)
def load(i):
 ns={};exec(open(f'/holdout_task/_frozen_nodes/node_0{i}.py').read(),ns);return ns['run_node']
a=load(0)({'__input__':s}); u=a['calibrated_x']; z=load(1)({'node_00':a})['x']; n=len(s['train']['y']); Y=np.column_stack([s['train']['y'],s['train']['traj']]); t=s['eval_times']
def poly(x,p):
 d=x.shape[1];return np.column_stack([np.prod(x[:,k],axis=1) for j in range(p+1) for k in combinations_with_replacement(range(d),j)])
def loo(A,Y):
 c,_,rank,_=np.linalg.lstsq(A,Y,rcond=1e-11); h=np.sum(np.linalg.svd(A,full_matrices=False)[0][:,:rank]**2,axis=1); e=A@c-Y; return np.mean(np.abs(e),axis=0),np.mean(np.abs(e/np.maximum(1e-9,1-h[:,None])),axis=0)
p=s['panel_c']; q=p['u']; P=s['panel_b']; B=np.linalg.lstsq(poly(P['u'],1),P['v'],rcond=None)[0]; w=poly(q,1)@B
print('B panel MAE',np.mean(abs(poly(P['u'],1)@B-P['v']))); print('v1 versus B',np.mean(abs(p['v1']-w),axis=0)); print('v1 canonical',np.linalg.lstsq(poly(q,1),p['v1'],rcond=None)[0]); print('B canonical',B)
for k in ['v1','v2']:
 for base in [q,w]:
  for kind,F in [('quad',poly(base,2)),('cubics',np.c_[poly(base,2),base**3]),('squash',np.c_[poly(base,2),np.tanh(2*base),np.sin(3*base)]),('trig',np.c_[poly(base,1),np.sin(2*base),np.sin(4*base),np.cos(3*base)])]:
   print(k,'q' if base is q else 'w',kind,loo(F,p[k])[1])
print('time right singular vectors',np.linalg.svd(Y[:,1:],full_matrices=False)[2][:2]);print('time mean',Y[:,1:].mean(axis=0)); print('scalar correlation with final',np.corrcoef(Y[:,0],Y[:,-1])[0,1]);
from scipy.linalg import lstsq
I=np.arange(n); folds=[(np.setdiff1d(I,h),h) for h in [I[-20:],I[:20],I[20:40],I[40:60]]]
def cv(A,reg=0,YY=Y):
 out=[]
 for j,k in folds:
  D=A[j]; V=A[k]; a=D.mean(0); scale=np.maximum(D.std(0),1e-10); D=(D-a)/scale; V=(V-a)/scale
  U,S,G=np.linalg.svd(D,full_matrices=False); c=(G.T*(S/(S**2+max(reg,1e-12))))@(U.T@(YY[j]-YY[j].mean(0))); err=np.mean(abs(V@c+YY[j].mean(0)-YY[k]),axis=0);out.append([err[0],err[1:].mean()]);
 return np.array(out)
from scipy.optimize import least_squares
b=Y[:,0]; print('scalar range/quantiles',np.quantile(b,[0,.02,.1,.25,.5,.75,.9,.98,1])); print('trajectory singular',np.linalg.svd(Y[:,1:],compute_uv=False)); print('z range',z[:n].min(0),z[:n].max(0))
z=z[:n];u=u[:n]; J=folds[0][0]; H=folds[0][1]; P=lambda a,c:(a[:,:,None]*c[:,None,:]).reshape(len(a),-1)
for F in [np.c_[z,np.tanh(2.2*z)],np.c_[poly(z,2),np.tanh(2.2*z)],np.c_[z,np.tanh(2.2*z),P(u,np.tanh(2.2*z))],np.c_[z,np.tanh(2.2*z),poly(np.tanh(2.2*z),2)]]:
 print('base shape',F.shape)
 for link in ['id','atanh','asinh','sin','tanh','softsign']:
  def fit(tr,te):
   D=np.c_[np.ones(n),F]; U,S,V=np.linalg.svd(D[tr],full_matrices=False); k=S>1e-8; Q=U[:,k]; M=(V[k].T/S[k]); p=Q.T@b[tr];
   if link in ['id','atanh','asinh']:
    best=[]
    for scale in ([1] if link=='id' else [1.01,1.1,1.3,1.6,2,3,5,10]):
     l=scale*np.max(abs(b[tr])); target=b[tr] if link=='id' else (np.arctanh if link=='atanh' else np.arcsinh)(b[tr]/l); a=Q.T@target; pred=D@M@a; pred=pred if link=='id' else l*(np.tanh if link=='atanh' else np.sinh)(pred); best.append((np.mean(abs(pred[tr]-b[tr])),np.mean(abs(pred[te]-b[te]))))
    return min(best)
   fun={'sin':np.sin,'tanh':np.tanh,'softsign':lambda x:x/np.sqrt(1+x*x)}[link];
   def residual(a):return a[-2]*fun(Q@a[:-2])+a[-1]-b[tr]
   fits=[least_squares(residual,np.r_[p/m,m,np.mean(b[tr])],max_nfev=80) for m in [.5,1,2]]; a=min(fits,key=lambda r:np.linalg.norm(r.fun)).x; return np.mean(abs(residual(a))),np.mean(abs(a[-2]*fun(D[te]@M@a[:-2])+a[-1]-b[te]))
  print(link,[fit(j,k) for j,k in folds],flush=True)
