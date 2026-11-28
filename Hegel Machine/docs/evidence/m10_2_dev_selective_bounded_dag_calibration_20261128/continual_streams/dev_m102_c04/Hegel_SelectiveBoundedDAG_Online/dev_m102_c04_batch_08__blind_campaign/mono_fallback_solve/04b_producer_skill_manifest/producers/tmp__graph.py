import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
pa=s['panel_a']; u,v=pa['u'],pa['v']; m=np.linalg.norm(u-v,axis=1)<1; w=np.array([ls(aff(u[m])[:,[0,j+1]],v[m,j]) for j in range(4)]).T
pb=s['panel_b']; wb=ls(aff(pb['u']),pb['v']); B=lambda x:aff(x)@wb
pairs=[(0,1),(0,3),(1,2),(2,3)]; prod=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in pairs]); cos=lambda x:np.cos(2.4*x)
pc=s['panel_c']; wc1=ls(aff(prod(pc['u'])),pc['v1']); wc2=ls(aff(cos(pc['u'])),pc['v2']); C=lambda x:aff(prod(x))@wc1; D=lambda x:aff(cos(x))@wc2
x=s['train']['x']; z=x*w[1]+w[0]; Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; tr=Y[:,1:];
def temporal(p):
 return -np.expm1(-t[:,None]*p)
o=least_squares(lambda p:(temporal(p)@ls(temporal(p),tr[:52].T)-tr[:52].T).ravel(),[.5,2.5],bounds=(.01,30)); print('best rates',o.x); print('B',wb,'C',wc1,'D',wc2,sep='\n');
base={'x':x,'z':z,'inv':(x-w[0])/w[1]}; sets=dict(base)
for nm,a in base.items():
 for name,op in [('B',B),('C',C),('D',D)]:sets[name+nm]=op(a)
for nm,a in list(sets.items()):
 if len(nm)==2 and nm[0] in 'BCD':
  for name,op in [('B',B),('C',C),('D',D)]:sets[name+nm]=op(a)

def score(f,reg=0):
 a=aff(f); coef=ls(a[:52],Y[:52]) if reg==0 else np.linalg.solve(a[:52].T@a[:52]+reg*np.eye(a.shape[1]),a[:52].T@Y[:52]); e=np.abs(a[52:]@coef-Y[52:]); return e[:,0].mean(),e[:,1:].mean()
rr=[]
for nm,a in sets.items():
 for method,h in [('linear',a),('diagsq',np.c_[a,a*a]),('quad',np.c_[a,np.column_stack([a[:,i]*a[:,j] for i in range(4) for j in range(i,4)])]),('sft',np.c_[a,a/(1+abs(a))]),('tnh',np.c_[a,np.tanh(a)]),('rat',np.c_[a,a/(1+a*a)]),('sin',np.c_[a,np.sin(2.4*a)]),('cos',np.c_[a,cos(a)]),('r2',np.c_[a,a*a/(1+a*a)])]:rr.append((*score(h),nm,method))
for nm1,nm2 in it.combinations(sets,2):
 a,b=sets[nm1],sets[nm2]; diag=a*b; allprod=(a[:,:,None]*b[:,None,:]).reshape(-1,16)
 for k,h in [('lin',np.c_[a,b]),('dot',np.c_[a,b,diag.sum(1)]),('diag',np.c_[a,b,diag]),('all',np.c_[a,b,allprod]),('diag-only',diag),('ratdot',np.c_[a,b,diag.sum(1)/(1+np.sum(b*b,1))])]:rr.append((*score(h,1e-7),nm1+'|'+nm2,k))
for key in [lambda r:r[0]+r[1],lambda r:r[0],lambda r:r[1]]:
 print('TOP',[(round(a,5),round(b,5),c,d) for a,b,c,d in sorted(rr,key=key)[:20]])
