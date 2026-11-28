import pickle,runpy,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['y']); Y=np.c_[t['y'],t['traj']]; m=runpy.run_path('/holdout_task/node_00.py'); raw=t['x']; x,b,c,d=[z[:n] for z in m['_features'](s)]; ids=np.arange(n); vals=[ids[-20:]]+list(np.array_split(np.random.default_rng(63).permutation(n),4))+[np.argsort(np.linalg.norm(x,axis=1))[-15:]]
np.set_printoptions(precision=4,suppress=True,linewidth=130)
print('Y mean/std',Y.mean(0),Y.std(0),'singular values',np.linalg.svd(Y-Y.mean(0),compute_uv=False)); print('Y rows',Y[:6]); print('study fields',s.keys(),'summary',t.get('summary'),'telemetry',s.get('telemetry_vec')); print('calibrated stats', [(z.mean(0),z.std(0)) for z in (x,b,c,d)])
def phi(h):
 return np.c_[np.ones(len(h)),h,.4*np.column_stack([h[:,i]*h[:,j] for i,j in combinations_with_replacement(range(h.shape[1]),2)])]
def score(h):
 a=phi(h); K=a@a.T; err=[]
 for v in vals:
  tr=np.setdiff1d(ids,v); kk=K[np.ix_(tr,tr)]; kt=K[np.ix_(v,tr)]; p=np.column_stack([kt@np.linalg.solve(kk+reg*np.eye(len(tr)),Y[tr,cols]) for reg,cols in ((.0003,slice(0,1)),(.001,slice(1,None)))]); e=abs(Y[v]-p);err.append([e[:,0].mean(),e[:,1:].mean()])
 return np.array(err)
models={'all':[x,b,c,d],'canonical':[b,c,d],'raw':[raw,b,c,d],'sum':[b+c+d],'sum_v':[b,c+d],'sum_bc':[b+c,d],'sum_bd':[b+d,c],'sums_x':[x,c+d],'sum_diff':[b,c+d,c-d],'mean':[b,(c+d)/2],'products':[b,c*d],'sum_products':[b,c+d,c*d],'gated1':[b+c*d],'gated2':[c+b*d],'gated3':[d+b*c],'mixed':[b,c+d,c*d,b*(c+d)],'x_mix':[x,b,c+d,c*d],'std':[z/z.std(0) for z in (b,c,d)],'center':[z-z.mean(0) for z in (b,c,d)],'v_scaled':[b,.5*c,.5*d],'linear_scaled':[.3*b,c,d]}
for name,blocks in models.items():
 e=score(np.column_stack(blocks)); print(name,e[0],e[1:5].max(0),e[-1])
a=phi(np.c_[b,c,d]); tags=['1']+[f'{q}{j}' for q in 'bcd' for j in range(4)]; tags += [f'{tags[i+1]}*{tags[j+1]}' for i,j in combinations_with_replacement(range(12),2)]
for reg in [.0003,.01,.1,1]:
 z=a.T@np.linalg.solve(a@a.T+reg*np.eye(n),Y); r=Y-a@z; print('ridge',reg,'train MAE',np.abs(r).mean(0),'largest features',[(tags[j],round(np.linalg.norm(z[j]),3)) for j in np.argsort(np.linalg.norm(z,axis=1))[-14:][::-1]])
