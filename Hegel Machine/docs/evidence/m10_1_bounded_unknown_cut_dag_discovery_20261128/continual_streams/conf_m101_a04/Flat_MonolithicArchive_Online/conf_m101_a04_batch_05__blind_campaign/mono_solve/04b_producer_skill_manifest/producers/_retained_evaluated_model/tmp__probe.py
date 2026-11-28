import numpy as np,pickle,runpy,itertools as it
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
bias=lambda a:np.c_[np.ones(len(a)),a]
rms=lambda a:np.sqrt(np.mean(a*a))
mae=lambda a:np.mean(abs(a))
cal=runpy.run_path('/tmp/hegel_continual_archive/conf_m101_a04_batch_04_solver.py')['_calibrate']; p=s['panel_a']; off=cal(p,np.zeros((1,4)))[0]; inv=cal(p,np.ones((1,4)))[0]-off
print('calibration offset',-off/inv,'scale',1/inv,'row residuals',np.sort(np.linalg.norm(cal(p,p['v'])-p['u'],axis=1)))
x=cal(p,tr['x']); b=s['panel_b']; bc=fit(bias(b['u']),b['v']); print('B coeff',bc); b=bias(x)@bc
quad=lambda u:np.column_stack([u[:,i]*u[:,j] for i,j in it.combinations_with_replacement(range(4),2)])
cubic=lambda u:np.column_stack([u[:,i]*u[:,j]*u[:,k] for i,j,k in it.combinations_with_replacement(range(4),3)])
for j in ['v1','v2']:
 p=s['panel_c']; u,v=p['u'],p[j]; print(j)
 bases={'quad':lambda u:bias(np.c_[u,quad(u)]),'bilin':lambda u:bias(quad(u)[:,[1,3,5,8]]),'cubic_odd':lambda u:bias(np.c_[u,cubic(u)]),'cubic_sep':lambda u:bias(np.c_[u,u**2,u**3]),'pure_quad':lambda u:bias(quad(u))}
 funs={'sin':lambda u,w:np.sin(w*u),'cos':lambda u,w:np.cos(w*u),'2sin':lambda u,w:np.c_[np.sin(w*u),np.sin(2*w*u)],'sincos':lambda u,w:np.c_[np.sin(w*u),np.cos(w*u)],'tanh':lambda u,w:np.tanh(w*u),'rat':lambda u,w:u/(1+w*u*u),'sin2cos':lambda u,w:np.c_[np.sin(w*u),np.sin(2*w*u),np.cos(w*u)]}
 for name,fn in funs.items():
  for linear in [False,True]:
   def design(u,w): return bias(np.c_[u,fn(u,w)] if linear else fn(u,w))
   def err(w): a=design(u,w); return rms(v-a@fit(a,v))
   ws=np.linspace(.12,10,200); k=np.argmin([err(w) for w in ws]); z=minimize_scalar(err,bounds=(ws[max(0,k-1)],ws[min(k+1,len(ws)-1)]),method='bounded'); print(name,linear,round(z.x,5),round(z.fun,6)); bases[name+str(linear)]=lambda u,w=z.x,fn=fn,linear=linear:bias(np.c_[u,fn(u,w)] if linear else fn(u,w))
 for name,fn in bases.items():
  a=fn(u); e=v-a@fit(a,v); hat=np.linalg.svd(a,full_matrices=False)[0]; h=(hat[:,:np.linalg.matrix_rank(a)]**2).sum(1); print('panel',name,'fit',round(rms(e),6),'LOO',round(rms(e/np.maximum(1-h[:,None],1e-12)),6))
 print('v1 fits along transformed u?',j, rms(v-bias(bias(u)@bc)@fit(bias(bias(u)@bc),v)))
Y=tr['traj'][:52]; _,S,V=np.linalg.svd(Y,full_matrices=False)
print('temporal vectors',V[:2]);
forms={'exp':lambda t,r:-np.expm1(-r*t),'rat':lambda t,r:t/(1+r*t),'sin':lambda t,r:np.sin(t*r),'decay':lambda t,r:t*np.exp(-r*t),'linear':lambda t,r:t,'cos':lambda t,r:1-np.cos(t*r)}
res=[]
for (n1,a),(n2,bf) in it.combinations_with_replacement(forms.items(),2):
 def mats(r): return np.c_[a(t,r[0]),bf(t,r[1])]
 def er(r): m=mats(r);return (Y-fit(m,Y.T).T@m.T).ravel()
 opts=[least_squares(er,q,bounds=([.0001,.0001],[9,9]),max_nfev=120) for q in [[1.8,.4],[.4,1.8],[1,2],[2,1]]]
 z=min(opts,key=lambda q:q.cost);res.append((rms(z.fun),n1,n2,z.x))
print('temporal basis options',sorted(res,key=lambda q:q[0])[:12])
for name,f in [('x',x),('B',b),('x quad',np.c_[x,quad(x)]),('B quad',np.c_[b,quad(b)]),('x/B',np.c_[x,b])]:
 a=bias(f); q=np.c_[tr['y'],tr['traj']];e=q[52:]-a[52:]@fit(a[:52],q[:52]);print('direct',name,mae(e[:,0]),mae(e[:,1:]))
