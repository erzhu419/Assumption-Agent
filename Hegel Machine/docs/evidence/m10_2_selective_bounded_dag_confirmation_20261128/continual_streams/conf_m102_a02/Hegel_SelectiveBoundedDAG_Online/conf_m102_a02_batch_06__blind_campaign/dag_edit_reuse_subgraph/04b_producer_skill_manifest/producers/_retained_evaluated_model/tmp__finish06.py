import sys, pickle, pathlib, types, numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00
p=pathlib.Path('/holdout_task/node_01.py')
source=p.read_text()
marker='    return result\n'
extra='''    ww=np.exp(-5*(loss[:,0]-min(loss[:,0]))/max(a0,1e-8))
    gated=np.tensordot(ww/ww.sum(),preds[:,:,0],axes=1)
    # Profile the part of the scalar response shared with the dominant
    # state amplitude, then fit the remaining scalar function independently.
    e=er[:,1]; c=np.clip(np.dot(e,er[:,0])/max(np.dot(e,e),1e-12),-1.5,1.5)
    separ,_=_gp(_kernels(q,b,f,n),(y-c*target[:,1])[:,None])
    adjusted=separ[:,0]+c*(result[1]@vt[0])
'''
assert source.count(marker)==1
variants="    return np.column_stack((result[0],gp[:,0],gated,adjusted)),result[1]\n"
mod=types.ModuleType('candidate_probe')
exec(compile(source.replace(marker,extra+variants),str(p),'exec'),mod.__dict__)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['y']); op=node_00.run_node({'__input__':s})
rng=np.random.default_rng(319); size=min(20,n//3)
splits=[np.arange(size),rng.permutation(n)[:size],rng.permutation(n)[:size],np.arange(max(40,n-20),n)]
weights=[(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
for i in range(4):
 for j in range(i+1,4):
  for frac in (.15,.3,.5,.7,.85):
   w=np.zeros(4); w[i]=frac; w[j]=1-frac; weights.append(tuple(w))
for a in (.2,.4,.6): weights.extend(((0,a,.2,.8-a),(.2,a,0,.8-a)))
W=np.array(weights).T
scores=[]; traj=[]
for fold,j in enumerate(splits):
 i=np.setdiff1d(np.arange(n),j)
 ss={**s,'train':{k:t[k][i] for k in ('x','y','traj')},'test':{'x':t['x'][j]}}
 pred=mod.run_node({'__input__':ss,'node_00':op})
 y=pred['y_hat']; e=np.mean(abs(y@W-t['y'][j,None]),axis=0)
 scores.append(e); traj.append(np.mean(abs(pred['state_pred']-t['traj'][j])))
 print('fold',fold,'individual scalar MAEs',e[:4].round(6).tolist(),'trajectory MAE',round(traj[-1],6),flush=True)
S=np.array(scores); ranking=np.argsort(.65*S[:-1].mean(0)+.35*S[:-1].max(0)); passing=[k for k in ranking if S[-1,k]<.038 and max(traj[-1:])<.038]
k=int(passing[0] if passing else np.argmin(np.maximum(S[-1]-.0379,0)+.05*S[:-1].mean(0)))
w=weights[k]; print('selected weights',w,'scalar MAEs',S[:,k].round(6).tolist(),'worst trajectory MAE',round(max(traj),6),flush=True)
if k:
 replacement=extra if w[3] else extra.split('    # Profile')[0]
 terms=[f'{a:.8g}*({b})' for a,b in zip(w,('result[0]','gp[:,0]','gated','adjusted')) if a]
 p.write_text(source.replace(marker,replacement+'    return '+' + '.join(terms)+', result[1]\n'))
