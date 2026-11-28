import pickle,numpy as np,itertools
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; X=np.asarray(c['true_channel_loads']); Y=np.asarray(c['bridge_raw']); rng=np.random.default_rng(48)
sets={'linear':[(i,) for i in range(4)],'additive2':[(i,)*p for p in range(1,3) for i in range(4)],'additive3':[(i,)*p for p in range(1,4) for i in range(4)],'full2':[s for p in range(1,3) for s in itertools.combinations_with_replacement(range(4),p)],'pair+add3':[s for p in range(1,3) for s in itertools.combinations_with_replacement(range(4),p)]+[(i,)*3 for i in range(4)],'full3':[s for p in range(1,4) for s in itertools.combinations_with_replacement(range(4),p)]}
def fit(A,Y):
 return np.stack([least_squares(lambda b:A@b-y,np.linalg.lstsq(A,y,rcond=None)[0],jac=lambda _:A.copy(),loss='soft_l1',f_scale=.002).x for y in Y.T],1)
for name,ss in sets.items():
 A=np.column_stack([np.ones(len(X))]+[X[:,s].prod(1) for s in ss]); E=np.empty_like(Y)
 for idx in np.array_split(rng.permutation(len(X)) if name=='linear' else perm,5) if False else np.array_split(np.random.default_rng(48).permutation(len(X)),5):
  tr=np.ones(len(X),bool); tr[idx]=False; E[idx]=A[idx]@fit(A[tr],Y[tr])-Y[idx]
 B=fit(A,Y); e=A@B-Y
 print(name,'cv abs',np.round(np.quantile(abs(E),[.5,.9,.95,.98,1]),5),'fit abs',np.round(np.quantile(abs(e),[.5,.9,.95,.98,1]),5))
 if name in ("pair+add3",):
  for s,b in zip([()]+ss,B): print(s,b.round(5))
  print('largest rows:',[(int(i),X[i].round(3).tolist(),e[i].round(4).tolist()) for i in np.argsort(np.max(abs(e),axis=1))[-8:]])
for lab in range(4):
 s=c['single_channel_sweeps']; m=s['lab_id']==lab;x=s['true_stress'][m]; y=s['cmd_stress'][m]; A=np.stack([x**i for i in range(4)],1); b=fit(A,y[:,None])[:,0]; print('sweep',lab,b.round(5),'error',(A@b-y).round(4))
