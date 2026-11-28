import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
from scipy.special import erf
from sklearn.linear_model import OrthogonalMatchingPursuit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; rr=np.load('/tmp/coords.npz');z=rr['z'];Y=rr['Y'];B=rr['B'];n=len(z)
def ls(A,Y):return np.linalg.lstsq(A,Y,rcond=1e-10)[0]
def add(a):return np.c_[np.ones(len(a)),a]
p=s['panel_b']; u,v=p['u'],p['v']; print('telemetry',s['telemetry_vec'],'z ranges',z.min(0),z.max(0),'B ranges',u.min(0),u.max(0));
funcs={'tanh':np.tanh,'sin':np.sin,'erf':erf,'sat':lambda h:h/np.sqrt(1+h*h),'rat':lambda h:h/(1+h*h),'exp':lambda h:h*np.exp(-h*h),'atan':np.arctan,'log':lambda h:h-np.arcsinh(h),'cubert':lambda h:np.cbrt(h),'cubic':lambda h:h**3}
models=[]
for name,f in funcs.items():
 for form in range(5):
  def ff(h,q):
   a,b,c=q
   if form==0:return f(a*h)+b*h+c*h**3
   if form==1:return f(a*h+b*h**3)+c*h
   if form==2:return h*np.tanh(a*h*h)+b*h+c*h**3
   if form==3:return h*np.tanh(a*h*h+b*h**4)+c*h
   if form==4:return h*(a+h*h)/(1+b*h*h+c*h**4)
  def res(q):
   A=add(ff(u,q));return (A@ls(A,v)-v).ravel()
  opts=[least_squares(res,q,max_nfev=130) for q in [[1,0,0],[.7,-.5,.25],[1.5,-1.4,.05]]];o=min(opts,key=lambda o:np.mean(o.fun**2));models.append((np.sqrt(np.mean(o.fun**2)),name,form,o.x,add(ff(z,o.x))@ls(add(ff(u,o.x)),v)))
for m in sorted(models,key=lambda a:a[0])[:12]:print('panel',m[:4])
bs=[('erf',rr['erf0']),('gauss',rr['gauss0'])]+[(m[1]+str(m[2]),m[-1]) for m in sorted(models,key=lambda a:a[0])[:3]]
for bn,b in bs:
 for raw in [z,b,np.c_[z,b]]:
  fs=[];ns=[]
  def ins(nm,a):
   if np.std(a)>1e-10 and np.isfinite(a).all():fs.append(a);ns.append(nm)
  m=raw.shape[1]; partitions=[('r'+str(k),raw[:,k]) for k in range(m)]
  for i,j in it.combinations(range(m),2):
   for a in [-1,1]:partitions.append((f'r{i}+{a}r{j}',raw[:,i]+a*raw[:,j]))
  for lab,a in partitions:
   for fname,f in [('id',lambda z:z),('sq',lambda z:z*z),('cube',lambda z:z**3),('abs',abs)]+list(funcs.items())[:-2]+[('cos',np.cos),('log1p',lambda z:np.log1p(z*z))]:
    for sc in ([1] if fname in ('id','sq','cube','abs') else [.4,.8,1.2,2.]):ins(f'{fname}{sc}({lab})',f(sc*a))
  for i,j in it.combinations_with_replacement(range(m),2):
   a,c=raw[:,i],raw[:,j];ins(f'{i}*{j}',a*c)
   for sc in [.5,1,2]:
    ins(f'tanh{sc}({i}*{j})',np.tanh(sc*a*c));ins(f'{i}/(1+{sc}*{j}^2)',a/(1+sc*c*c));ins(f'{j}/(1+{sc}*{i}^2)',c/(1+sc*a*a))
  for i,j,k in it.combinations_with_replacement(range(m),3):ins(f'{i}*{j}*{k}',raw[:,i]*raw[:,j]*raw[:,k]);
  F=np.column_stack(fs);mu=F[:52].mean(0);st=F[:52].std(0);F=(F-mu)/np.maximum(st,1e-12); A=Y.copy();A[:,1:]=tr['traj'][:,[0,-1]]
  for jj in range(3):
   y=A[:,jj]; lm=OrthogonalMatchingPursuit(n_nonzero_coefs=10).fit(F[:52],y[:52]); ind=np.flatnonzero(lm.coef_); p=lm.predict(F); print(bn,m,'out',jj,'MAE',np.round([abs(p[:52]-y[:52]).mean(),abs(p[52:]-y[52:]).mean()],4),'terms',[(ns[k],round(lm.coef_[k],3)) for k in ind]);
  for typ in ['linear','quadratic']:
   F=add(raw) if typ=='linear' else add(np.c_[raw,np.column_stack([raw[:,i]*raw[:,j] for i,j in it.combinations_with_replacement(range(m),2)])]);P=F@ls(F[:52],Y[:52]); d=P-Y;print(bn,m,typ,np.round([abs(d[:52,0]).mean(),abs(d[52:,0]).mean(),abs(d[:52,1:]@B.T).mean(),abs(d[52:,1:]@B.T).mean()],4))
