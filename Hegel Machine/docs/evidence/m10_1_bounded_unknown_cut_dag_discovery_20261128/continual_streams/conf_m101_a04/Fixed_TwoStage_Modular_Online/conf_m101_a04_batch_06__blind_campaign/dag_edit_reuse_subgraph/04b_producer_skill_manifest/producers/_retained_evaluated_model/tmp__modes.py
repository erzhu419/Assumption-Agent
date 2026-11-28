import pickle,runpy,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['y']); Y=np.c_[t['y'],t['traj']]; m=runpy.run_path('/holdout_task/node_00.py'); x,b,c,d=[z[:n] for z in m['_features'](s)]; ids=np.arange(n); vals=[ids[-20:]]+list(np.array_split(np.random.default_rng(63).permutation(n),4))+[np.argsort(np.linalg.norm(x,axis=1))[-15:]]
np.set_printoptions(precision=4,suppress=True,linewidth=130)
D={'b':b,'c':c,'d':d,'bc':(b[:,:,None]*c[:,None,:]).reshape(n,-1),'bd':(b[:,:,None]*d[:,None,:]).reshape(n,-1),'cd':(c[:,:,None]*d[:,None,:]).reshape(n,-1),'BC':b*c,'BD':b*d,'CD':c*d,'BB':b*b,'CC':c*c,'DD':d*d,'z':x**2}
def fit(A,R):
 C=np.linalg.lstsq(A,R,rcond=1e-8)[0];return C,R-A@C
T=t['traj']; P,S,V=np.linalg.svd(T-T.mean(0),full_matrices=False); P*=S
print('temporal mean',T.mean(0),'temporal directions',V[:4]);
A=np.c_[np.ones(n),b,c,d,D['bc'],D['CD']]; M,R=fit(A,np.c_[Y,P[:,:4]]); print('base train rmse',np.mean(R*R,0)**.5); print('base y coeff:',M[:,0]); print('base mode coeff:',M[:,-4:]); print('base traj residual sv',np.linalg.svd(R[:,1:9],compute_uv=False));
F=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']
for extras in ('','BD','bd','BC BD BB CC DD','BD BB CC DD','cd','BD z','z','BD DD','BD CC','BD BB','BB','CC','DD','BD DD CC','BD DD BB'):
 B=np.c_[A,*[D[k] for k in extras.split()]]; e=[]; ee=[]
 for v in vals:
  tr=np.setdiff1d(ids,v); pred=B[v]@fit(B[tr],Y[tr])[0]; rr=np.abs(pred-Y[v]); e.append([rr[:,0].mean(),rr[:,1:].mean()]);
  trc=Y[tr]; C=fit(B[tr],trc)[0]; U,ss,VV=np.linalg.svd(B@C-(B@C)[tr].mean(0),full_matrices=False); H=(U[:,:4]*np.sqrt(n)); o=F({'node_00':{'h':H[np.r_[tr,v]],'n_train':len(tr),'y':t['y'][tr],'traj':t['traj'][tr]}}); rr=np.abs(np.c_[o['y_hat'],o['state_pred']]-Y[v]); ee.append([rr[:,0].mean(),rr[:,1:].mean()]);
 M,R=fit(B,Y); print(extras,'train rmse',np.round(np.mean(R*R,0)**.5,4),'ols',np.round(np.array(e)[[0,-1]],4),'compressed',np.round(np.array(ee)[[0,-1]],4));
