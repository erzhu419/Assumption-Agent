exec(open('/tmp/compose07.py').read().split('rows=[]')[0])
print('response range',y.min(),y.max(),'traj amplitude',T[:,-1].min(),T[:,-1].max());
for fn,p0 in [(lambda x,p:aff(x+p[1]*np.sin(p[0]*x)+p[2]*x**3), [2.9,-.4,0]),(lambda x,p:aff(x+p[1]*np.sin(p[0]*x)+p[2]*np.sin(2*p[0]*x)),[2.9,-.4,0]),(lambda x,p:aff(x+p[1]*np.sin(p[0]*x)+p[2]*x*x),[2.9,-.4,0]),(lambda x,p:aff(x+p[1]*np.tanh(p[0]*x)+p[2]*x**3),[1.4,-1,0]),(lambda x,p:np.c_[aff(x),np.sin(p[0]*x),x**3], [2.9])]:
 er,F=calibrate(b['u'],b['v'],fn,p0); print('new B',er); Bx=F(u); base['new'+str(len(base))]=np.c_[Bx,P(Bx),C2(u)]
_,_,vt=np.linalg.svd(T,full_matrices=False); time=vt[0]*np.sign(vt[0,-1]); amp=T@time; print('amp range',amp.min(),amp.max(),'corr',np.corrcoef(y,amp)[0,1]);
from scipy.optimize import least_squares
acts={'lin':lambda x:x,'tanh':np.tanh,'sin':np.sin,'sinh':lambda x:np.sinh(np.clip(x,-8,8)),'atan':np.arctan,'cubic':lambda x:x+x**3,'soft':lambda x:x/np.sqrt(1+x*x),'silu':lambda x:x/(1+np.exp(-np.clip(x,-50,50))), 'mix':lambda x:x+np.sin(x)}
parts=[np.arange(52,72)]+[np.arange(i,i+18) for i in range(0,72,18)]
sets={k:v for k,v in base.items() if 'B' in k or k.startswith('new')}; sets.update({'bp':np.c_[base['B0'],P(base['B0'])],'bq':np.c_[base['B0'],C2(u)],'all':np.c_[base['B0'],P(base['B0']),C2(u),u]})
for i,z0 in enumerate((y,amp)):
 rows=[]
 for name,x in sets.items():
  X=aff(x); X[:,1:]/=np.maximum(np.std(X[:,1:],axis=0),.001)
  for k,act in acts.items():
   for mode in (0,1):
    es=[]
    for h in parts:
     tr=np.setdiff1d(np.arange(n),h); A=X[tr]; v=z0[tr]; c0=fit(A,v); sd=np.std(v); 
     def pred(M,p):return p[-2]+p[-1]*act(M@p[:M.shape[1]])+(M@p[M.shape[1]:-2] if mode else 0)
     init=np.r_[c0/sd*(.65 if mode else 1),c0*.35 if mode else [],0,sd];
     opt=least_squares(lambda p:np.r_[pred(A,p)-v, .01*p],init,max_nfev=85).x; es.append(np.mean(abs(pred(X[h],opt)-z0[h]))*(1 if i==0 else time.mean()))
    rows.append((max(es),es[0],np.mean(es[1:]),name,k,mode))
 print('LINK',i,*sorted(rows)[:18],sep='\n')
