import pickle,numpy as np
from scipy.optimize import root,least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tr=s['train']; t=s['eval_times']; x=np.r_[tr['x'], s['test']['x']]
ls=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
def pair(x): return np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(4),2)])
def cal(u): return np.stack([np.ones_like(u),u,np.tanh(1.8*u),np.sin(2.4*np.roll(u,2,axis=-1))],axis=-1)
p=s['panel_a']; a=cal(p['u']); co=[]
for j in range(4):
 d=a[:,j]; y=p['v'][:,j]; ii=np.arange(len(d))
 for _ in range(15):
  w=ls(d[ii],y[ii]); e=abs(d@w-y); ii=np.argsort(e)[:39]
 ii=e<.001; co.append(ls(d[ii],y[ii])); print('cal inliers',sum(ii))
co=np.array(co); q=np.array([root(lambda z: (cal(z)*co).sum(-1)-r,r,tol=1e-10).x for r in x]); print('cal max err',abs((cal(q)*co).sum(-1)-x).max())
b=s['panel_b']; c=s['panel_c']; u=c['u']; v=c['v2']; trans=lambda q:aff(np.cos(2.6*q)); w=trans(q)@ls(trans(b['u']), b['v']); c1=lambda x:aff(np.c_[x,pair(x)])
# Profile single-parameter transforms for the second C operator.
forms={'poly':lambda x,p:x+p*x*x,'rat':lambda x,p:x/(1+p*x),'absrat':lambda x,p:x/(1+p*abs(x)),'sqrat':lambda x,p:x/(1+p*x*x),'exp':lambda x,p:np.expm1(p*x)/p,'softplus':lambda x,p:np.logaddexp(0,p*x)/p,'trig':lambda x,p:np.sin(p*x)+np.cos(p*x),'tanh':lambda x,p:np.tanh(p*x)+.22*x*x,'pow':lambda x,p:(x+1.5)**p}
fits=[]
for k,fn in forms.items():
 def err(p):
  d=aff(fn(u,p[0])); return (d@ls(d,v)-v).ravel()
 lim=.7 if k=='rat' else 4
 candidates=[least_squares(err,[p],bounds=([.001],[lim])) for p in [.15,.55] if p<lim]; o=min(candidates,key=lambda o:o.cost); d=aff(fn(u,o.x[0])); er=np.sqrt(np.mean(err(o.x)**2)); fits.append((er,k,o.x[0])); print('v2',k,o.x[0],er)
print('v1 coef',ls(c1(u),c['v1']).round(4)); print('ranges',[(a[:72].min(),a[:72].max(),a[72:].min(),a[72:].max()) for a in [x,q,w]])
F={'x':x,'q':q,'w':w}
for k,z in list(F.items()):
 F[k+'C1']=c1(z)@ls(c1(u),c['v1']); F[k+'C2poly']=aff(np.c_[z,z*z,z**3])@ls(aff(np.c_[u,u*u,u**3]),v)
 for _,nm,p in sorted(fits)[:1]: F[k+'C2'+nm]=aff(forms[nm](z,p))@ls(aff(forms[nm](u,p)),v)
Y=np.c_[tr['y'],tr['traj']]; n=52
for name,z in F.items():
 print(name, 'raw', end=' ')
 for d in [aff(z),aff(np.c_[z,pair(z)]),aff(np.c_[z,z*z,pair(z)]),aff(np.c_[z,np.tanh(2.15*z),pair(z)]),aff(np.c_[z,np.sin(2.15*z),pair(z)]),aff(np.c_[z,z/(1+1.25*z*z),pair(z)]),aff(np.c_[z,np.cos(2.15*z),pair(z)])]:
  err=abs(d[n:72]@ls(d[:n],Y[:n])-Y[n:]); print('%.4f/%.4f'%(err[:,0].mean(),err[:,1:].mean()),end=' ')
 print()
