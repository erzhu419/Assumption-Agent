exec(open('/tmp/fit07.py').read().split('for name,b')[0])
m=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); a=s['panel_a']; X=m['_a'](a['u'],a['v'],s['train']['x']); W=w[:n]; ring=lambda a:a*np.roll(a,-1,axis=1)
Q=np.linalg.lstsq(np.c_[np.ones(len(u)),ring(u)],p['v1'],rcond=None)[0]; R=ring(W); V=np.c_[np.ones(n),R]@Q
B=np.c_[y,tr]; print('ring-linear coefficients', np.linalg.lstsq(np.c_[np.ones(n),R,W],B[:,[0,1,-1]],rcond=None)[0]); print('worst ring residuals',np.c_[W,B[:,[0,-1]],(B-np.c_[np.ones(n),R,W]@reg(np.c_[np.ones(n),R,W],B,0))[:,[0,-1]]][np.argsort(abs(B[:,0]-np.c_[np.ones(n),R,W]@reg(np.c_[np.ones(n),R,W],y,0)))[-12:]])
def ff(z):return np.c_[z,pair(z),z**3]
from sklearn.linear_model import lars_path
from scipy.linalg import solve
S={'w':W,'inv_a':X,'ring':R,'v1':V,'x':s['train']['x']}; ff0=[]; names=[]
def add(name,F):
 if F.ndim==1:F=F[:,None]
 ff0.append(F); names.extend([name+str(j) for j in range(F.shape[1])]); return F
for key,z in S.items():
 add(key,z);add(key+'sq',z*z)
 for i,j in itertools.combinations(range(4),2):
  for sign in [1,-1]:
   for c in [.5,1,2]:
    v=z[:,i]+sign*c*z[:,j];add(f'{key}{i}{j}:{sign*c}sum_',v)
    for g in [1,3,7]:
     add(f'{key}{i}{j}:{sign*c}:{g}act_', np.c_[np.tanh(g*v),np.sin(g*v),v/(1+(g*v)**2),1/(1+(g*v)**2),np.arcsinh(g*v),abs(v)])
  add(key+f'pair{i}{j}_',z[:,i]*z[:,j]);
 for h in [1,3,7]:
  add(key+f'act{h}_',np.c_[np.tanh(h*z),np.sin(h*z),z/(1+(h*z)**2),1/(1+(h*z)**2),np.arcsinh(h*z),abs(z)])
for zkey in ['w','ring','v1','inv_a']:
 z=S[zkey]; C=ff(z); add(zkey+'poly',C)
 add(zkey+'cubics',poly(z,3)[:,14:]);
 for g in [.5,2,8]:
  for j in range(4):
   add(f'{zkey}gat{g},{j}_',np.c_[W,R]/(1+g*z[:,j:j+1]**2))
A=np.column_stack(ff0); sd=np.maximum(A.std(0),1e-9); A=A/sd
for target in [y,tr[:,-1],tr@np.linalg.svd(tr,full_matrices=False)[2][0]]:
 print('target');
 for sel in [['w','ring','v1'],['w','ring'],['inv_a'],['w'],['v1'],list(S)]:
  ix=np.array([i for i,name in enumerate(names) if any(name.startswith(k) for k in sel)]); xs=A[:,ix]; a,v=folds[0]; train=xs[a]-xs[a].mean(0); yy=target[a]-target[a].mean(); _,act,coef=lars_path(train,yy,method='lasso',max_iter=55); act=np.array(act); out=[]
  for j in range(2,coef.shape[1]):
   cols=np.flatnonzero(coef[:,j]); F=np.c_[np.ones(n),xs[:,cols]];
   for regu in [0,.15,1]:
    e=np.mean(abs(F[v]@reg(F[a],target[a],regu)-target[v])); out.append((e,j,regu,[names[ix[k]] for k in cols][:16]));
  print(sel,sorted(out)[:1]);
