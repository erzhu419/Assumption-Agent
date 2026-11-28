import importlib.util,pickle,numpy as np,copy,re
path='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('bs',path); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=s['train']; rng=np.random.default_rng(545); splits=[rng.permutation(52) for _ in range(4)]; book={}
for ids in splits+[np.arange(72)]:
 n=39 if len(ids)==52 else 52; st=copy.deepcopy(s); tr,te=ids[:n],ids[n:]; st['train'].update({k:T[k][tr] for k in ('x','y','traj')}); st['test']={'x':T['x'][te]}; by=mod.baseline(st)
 for v in range(3):
  cand=mod._new(st,v)
  for b in (0,.5,1):
   y=np.mean(abs(b*cand['y_hat']+(1-b)*by['y_hat']-T['y'][te])); t=np.mean(abs(b*cand['state_pred']+(1-b)*by['state_pred']-T['traj'][te])); book.setdefault((v,b),[]).append((y,t));
for k,v in book.items():print(k,'inner mean',np.mean(v[:4],0),'inner worst',np.max(v[:4],0),'outer',v[-1])
k=min(book,key=lambda k: max(np.mean(book[k][:4],0))+.2*max(np.max(book[k][:4],0))); print('selected',k); text=open(path).read();text=re.sub(r'_VARIANT=0\n_BLEND=1.0',f'_VARIANT={k[0]}\n_BLEND={k[1]}',text);open(path,'w').write(text)
