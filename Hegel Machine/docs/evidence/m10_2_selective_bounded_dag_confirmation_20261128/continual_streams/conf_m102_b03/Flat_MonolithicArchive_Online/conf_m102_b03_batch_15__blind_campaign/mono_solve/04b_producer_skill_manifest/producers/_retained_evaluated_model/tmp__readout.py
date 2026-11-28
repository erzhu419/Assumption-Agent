from identify import *
F=np.load('/tmp/probe_features.npz'); z,p,q=[F[k] for k in ('z','p','q')];p=p[0];q=q[0]; tr=s['train']; T=tr['traj']; y=tr['y']; t=s['eval_times']; n=52
u=s['panel_b']['u']; b=D(z)@ls(D(u),s['panel_b']['v']);dot=(p*q).sum(1); cross=(p[:,:,None]*q[:,None,:]).reshape(len(p),-1)
def h(k):return -np.expm1(-t[:,None]*k)
k=least_squares(lambda k:(h(k)@ls(h(k),T[:n].T)-T[:n].T).ravel(),[1.83,.42]).x
amp=ls(h(k),T.T).T
print('rates',k,'amp std',amp.std(0),'y std',y.std())
Y=np.c_[y,amp]; options={}
for name,a in [('raw',np.r_[tr['x'],s['test']['x']]),('z',z),('b',b),('p',p),('q',q),('pq',np.c_[p,q]),('zpqb',np.c_[z,p,q,b])]:
 options[name]=D(a)
 if name not in ('z','b','p','q'):continue
 options[name+' poly2']=D(np.c_[a,*[a[:,i]*a[:,j] for i,j in it.combinations_with_replacement(range(4),2)]])
for name,a in [('dot',dot),('diag',p*q),('cross',cross),('full2',np.c_[cross,*[g[:,i]*g[:,j] for g in [p,q] for i,j in it.combinations_with_replacement(range(4),2)]]),('extra',np.c_[dot,np.tanh(dot), np.linalg.norm(p,axis=1),np.linalg.norm(q,axis=1)])]:options['pq '+name]=D(np.c_[p,q,a])
for r in range(4): options[f'pq shift{r}']=D(np.c_[p,q,p*np.roll(q,r,axis=1)]);options[f'pq sumshift{r}']=D(np.c_[p,q,(p*np.roll(q,r,axis=1)).sum(1)])
for name,A in options.items():
 c=ls(A[:n],Y[:n]);res=A[:len(y)]@c-Y; print(name,A.shape[1], 'train rms',np.round(np.mean(res[:n]**2,axis=0)**.5,5),'valid mae',np.round(np.mean(abs(res[n:]),axis=0),5),'traj',np.mean(abs(res[n:,1:]@h(k).T)))
 if name in ('pq cross','pq dot'):print('coefs',np.round(c,3))
print('raw fits p,q,amp scale',np.quantile(np.c_[p[:72],q[:72],amp],[0,.5,1],axis=0))
