import numpy as np, pickle, runpy,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/holdout_task/node_01.py'); up=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s}); y=s['train']['y']; q=m['_paired_coordinates'](s['train']['x'],s['panel_a'],s['telemetry_vec'][0]); b=m['_panel_features'](q,up); u=s['panel_c']['u']; c=np.c_[m['_design'](q,15),b]@m['_ls'](np.c_[m['_design'](u,15),m['_panel_features'](u,up)],np.c_[s['panel_c']['v1'],s['panel_c']['v2']]); inv=1/(1+s['telemetry_vec'][3]*q*q); ids=np.arange(len(y)); splits=[ids[52:]]+list(np.array_split(np.random.default_rng(12).permutation(len(y)),5));
for z in q.T:
 o=np.argsort(z); splits.extend([o[:14],o[-14:]])
D={'q':q,'c':c,'inv':inv,'quad':m['_design'](q,15)[:,5:],'q3':q**3,'bq0':b*q,'bq1':b*np.roll(q,-1,axis=1),'bq2':b*np.roll(q,-2,axis=1),'bq3':b*np.roll(q,-3,axis=1),'b2':m['_design'](b,15)[:,5:],'gate':b*inv,'cross':(b*np.roll(q,-1,axis=1)).sum(1)[:,None]}
def ev(X,a):
 e=[]
 for te in splits:
  tr=np.setdiff1d(ids,te); e.append(np.mean(abs(m['_ridge'](X[tr],y[tr],X[te],a)-y[te])))
 return np.array([e[0],max(e[1:6]),max(e[6:]),np.mean(e[6:])])
print('current',ev(np.c_[b,c,inv],.025)); rows=[]
for n in range(4):
 for terms in itertools.combinations(D,n):
  X=np.column_stack([b,*[D[k] for k in terms]])
  for a in [.0025,.025,.25]:
   e=ev(X,a); rows.append((max(e[:3]),e.tolist(),terms,a))
for row in sorted(rows)[:16]:print(row)
