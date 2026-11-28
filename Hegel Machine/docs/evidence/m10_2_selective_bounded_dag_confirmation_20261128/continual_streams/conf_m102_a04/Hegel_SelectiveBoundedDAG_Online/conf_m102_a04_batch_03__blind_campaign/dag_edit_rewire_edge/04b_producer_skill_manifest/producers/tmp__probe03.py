exec(open('/tmp/inspect03.py').read().split("print('times'")[0])
from scipy.optimize import least_squares
from sklearn.linear_model import OrthogonalMatchingPursuit
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; print(k)
 for key,v in p.items():
  if key=='u': continue
  print(key)
  for name,x in [('lin',u),('quad',PolynomialFeatures(2,include_bias=False).fit_transform(u))]:
   f=np.c_[np.ones(len(x)),x]; fit=np.linalg.lstsq(f,v,rcond=None)[0]; print(name,np.mean(abs(v-f@fit),0),fit if name=='lin' else '')
for seeds in [-1,0,1,2]:
 idx=np.arange(n) if seeds<0 else np.random.default_rng(seeds).permutation(n); i,j=idx[:52],idx[52:]; ss={**s,'train':{k:tr[k][i] for k in ('x','y','traj')},'test':{'x':tr['x'][j]}}; w=a.run_node({'__input__':ss})['features']; T=b.run_node({'__input__':ss}); A=T['basis']; Y=np.c_[tr['y'][idx],tr['traj'][idx]@np.linalg.pinv(A).T];
 M=np.linalg.lstsq(np.c_[np.ones(n),w],tr['x'][idx],rcond=None)[0]; print('reconstruction',np.mean(abs(np.c_[np.ones(n),w]@M-tr['x'][idx]),0));
 def measure(pred): return np.round([np.mean(abs((pred-Y[52:])[:,0])),np.mean(abs((pred-Y[52:])[:,1:]@A.T))],4)
 for name,v in [('w',w),('raw',tr['x'][idx]),('combined',np.c_[w,tr['x'][idx]])]:
  p=PolynomialFeatures(3,include_bias=False); F=p.fit_transform(v); names=p.get_feature_names_out();
  F=np.c_[F,*[func(freq*v).T.T for func in (np.sin,np.cos,np.tanh) for freq in (.7,1.4,2.8)]]; F-=F[:52].mean(0); F/=np.maximum(F[:52].std(0),1e-7); pairs=[]
  for k in [4,8,12,18,25]:
   pred=[]
   for q in range(3):
    reg=OrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(F[:52],Y[:52,q]); pred.append(reg.predict(F[52:]));
    if k==8 and seeds==-1 and name=='w': print('chosen',q,[(names[z] if z<len(names) else str(z),round(reg.coef_[z],3)) for z in np.flatnonzero(reg.coef_)])
   pairs.append(measure(np.array(pred).T))
  print(seeds,name,np.array(pairs).tolist())
