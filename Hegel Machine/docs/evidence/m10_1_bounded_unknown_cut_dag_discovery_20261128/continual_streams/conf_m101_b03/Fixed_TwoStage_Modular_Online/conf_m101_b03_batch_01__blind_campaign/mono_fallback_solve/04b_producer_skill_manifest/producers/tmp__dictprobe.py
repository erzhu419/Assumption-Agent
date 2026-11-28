import pickle,numpy as np
from itertools import combinations,combinations_with_replacement
from sklearn.linear_model import OrthogonalMatchingPursuit, LassoLars
from sklearn.model_selection import KFold
from calibrated import calibrate,inverse,time_modes
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; q=tr['traj']; z=inverse(tr['x'],calibrate(s['panel_a'])); x=z.T
specs=[]
def add(name,f): specs.append((name,f))
for k in range(1,5):
 for ix in combinations_with_replacement(range(4),k): add(''.join(map(str,ix)),np.prod(x[list(ix)],axis=0))
for i in range(4):
 for j in range(i):
  for c in [-2,-1,-.5,.5,1,2]:
   w=x[i]+c*x[j]
   for k in [.7,1.5,2.5,3.5]:
    for nm,f in [('sin',np.sin(k*w)),('cos',np.cos(k*w)),('tanh',np.tanh(k*w)),('rat',1/(1+k*w*w))]: add(f'{nm}({k}*({i}+{c}*{j}))',f)
  for k in [.7,1.5,2.5,3.5]:
   w=x[i]*x[j];add(f'sin({k}*{i}{j})',np.sin(k*w));add(f'tanh({k}*{i}{j})',np.tanh(k*w));add(f'cos({k}*{i}{j})',np.cos(k*w))
for i in range(4):
 for k in [.5,1,1.5,2,2.5,3,3.5,4]:
  for nm,f in [('sin',np.sin(k*x[i])),('cos',np.cos(k*x[i])),('tanh',np.tanh(k*x[i])),('rat',1/(1+k*x[i]**2)),('orat',x[i]/(1+k*x[i]**2)),('soft',np.sqrt(1+k*x[i]**2))]:
   add(f'{nm}({k}*{i})',f)
   for j in range(4):
    if j!=i: add(f'{j}*{nm}({k}*{i})',x[j]*f)
F=np.column_stack([v for _,v in specs]); labels=np.array([k for k,_ in specs]); splits=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=111).split(z))
for k in [0,1,2]:
 for fam,mask in [('all',np.ones(len(labels),bool)),('uni',np.array(['+' not in v and not ('*' in v and len(v.split('*')[-1])==3) for v in labels])),('add',np.array([not ('+' in v or v[0].isdigit() and '*' in v) for v in labels]))]:
  for mode,p in [('omp',10),('omp',16),('lars',.001),('lars',.004),('ridge',.03),('ridge',.3)]:
   errs=[]
   for it,(a,b) in enumerate(splits):
    h,m=time_modes(s['eval_times'],q[a],len(a)); m=q@np.linalg.pinv(h.T); Y=np.c_[tr['y'],m]; y=Y[:,k]; g=F[:,mask]; mu=g[a].mean(0); sd=g[a].std(0); g=(g-mu)/sd; ya=y[a]-y[a].mean();
    if mode=='omp': w=OrthogonalMatchingPursuit(n_nonzero_coefs=p,fit_intercept=False).fit(g[a],ya).coef_
    elif mode=='lars': w=LassoLars(alpha=p,fit_intercept=False).fit(g[a],ya).coef_
    else: w=g[a].T@np.linalg.solve(g[a]@g[a].T+p*np.eye(len(a)),ya)
    pred=g@w+y[a].mean(); errs.append(np.mean(abs(pred[b]-y[b])))
    if it==0:
     ix=np.argsort(-abs(w))[:13]; desc=list(zip(labels[mask][ix],np.round(w[ix],3))); fit=np.mean(abs(pred[a]-y[a]));
     if k==0 and fam=='all' and mode=='omp' and p==10:
      for cols in [[1,2], [1,2,3,4,5]]:
       H=np.c_[np.ones(len(x.T)),Y[:,1:],z]; ph=H@np.linalg.lstsq(H[a][:,[0]+cols],y[a],rcond=None)[0] if len(cols)==6 else H[:,[0]+cols]@np.linalg.lstsq(H[a][:,[0]+cols],y[a],rcond=None)[0]; print('cross-head MAE',np.mean(abs(ph[b]-y[b])))
   print(k,fam,mode,p,np.round(errs,4),round(fit,4),desc if mode!='ridge' and fam!='uni' else '')
