exec(open('/tmp/graph.py').read().split("print('B'")[0])
from scipy.optimize import least_squares
T=s['eval_times']; tr=s['train']['traj']; y=s['train']['y']; n=52
print('y vs observed state residual',np.std(y-D(tr)@ls(D(tr),y)))
print('summary',s['train']['summary'],[y.mean(),y.std(),tr[:,-1].mean(),tr[:,-1].std()]); print('z ranges',z.min(0),z.max(0))
U,S,V=np.linalg.svd(tr,full_matrices=False)
def basis(p, kind):
 h=-np.expm1(-T[:,None]*p[:2]); k=p[2] if len(p)>2 else 1
 f={'sin':np.sin(k*T),'exp':-np.expm1(-k*T),'lin':T,'const':np.ones(len(T)),'quad':h[:,0]*h[:,1],'dexp':T*np.exp(-k*T),'rat':T/(1+k*T)}
 return np.c_[h,f[kind]]
for kind in ['sin','exp','lin','const','quad','dexp','rat']:
 def res(p):
  H=basis(p,kind); return (H@ls(H,tr[:52].T)-tr[:52].T).ravel()
 ft=min([least_squares(res,np.array([a,b,c]),bounds=([.02,.02,.02],[8,8,8]),max_nfev=180) for a,b,c in [(1.8,.45,.2),(1.2,.25,2.8),(3,.9,1.7)]],key=lambda r:np.linalg.norm(r.fun));print(kind,np.round(ft.x,5),np.std(ft.fun))
print('y vs time PC correlations',np.corrcoef(np.c_[y,U[:,:3]].T)[0]); print('first 12 y and scores',np.round(np.c_[y,U[:,:3]*S[:3]][:12],3))
ops.update(I=lambda x:(x-b[0])@np.linalg.inv(b[1:]));
base={'z':z,'x':x,'bz':ops['B'](z),'iz':ops['I'](z),'bx':ops['B'](x),'ix':ops['I'](x)}
for key,v in list(base.items()):
 for name,w in [('tanh',np.tanh(1.25*v)),('sine',np.sin(1.25*v)),('rat',v/(1+.22*v*v)),('softsign',v/(1+.22*abs(v))),('sq',v*v)]:base[key+name]=w
Y=np.c_[y,U[:,:3]*S[:3]]; Y/=Y[:n].std(0)
r=[]
for name,v in base.items():
 P,Q=ops['P'](v),ops['Q'](v)
 for groups in [(('P',P),('Q',Q)),(('v',v),('P',P),('Q',Q)),(('P',P),('Q',Q),('PQ',P*Q)),(('PQsum',np.sum(P*Q,1)[:,None]),('P',P),('Q',Q)),(('P',P),('Q',Q),('PQall',(P[:,:,None]*Q[:,None,:]).reshape(-1,16)))]:
  mat=D(np.column_stack([val for _,val in groups])); coef=ls(mat[:n],Y[:n]); er=abs(mat@coef-Y);r.append((name,','.join(key for key,_ in groups),np.round(er[:n].mean(0),3),np.round(er[n:].mean(0),3)))
for j in range(4):print('MODE',j,*sorted(r,key=lambda t:t[2][j]+t[3][j])[:7],sep='\n')
