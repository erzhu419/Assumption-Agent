import numpy as np,pickle,sys,glob,importlib.util,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj'];
f=glob.glob('/tmp/hegel_continual_archive/*batch_09_solver.py')[0]; sp=importlib.util.spec_from_file_location('calmod',f); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
z=m._decode(np.r_[tr['x'],s['test']['x']],m._calibration(s['panel_a']))
af=lambda u: np.c_[np.ones(len(u)),u]; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
F={'sin':lambda u,p:np.sin(p*u),'tanh':lambda u,p:np.tanh(p*u),'cos':lambda u,p:np.cos(p*u),'atan':lambda u,p:np.arctan(p*u),'softsign':lambda u,p:u/(1+p*abs(u)), 'rat':lambda u,p:u/(1+p*u*u),'threshold':lambda u,p:np.sign(u)*np.maximum(abs(u)-p,0),'clip':lambda u,p:np.clip(u,-p,p),'asinh':lambda u,p:np.arcsinh(p*u)}
def panel(key,out):
 u=s[key]['u'];v=s[key][out]; results=[]
 def search(name,fn,starts,bounds):
  def residual(p, mask=slice(None)):
   a=af(fn(u,p))[mask]; return (a@fit(a,v[mask])-v[mask]).ravel()
  sol=min((least_squares(residual,p,bounds=bounds,max_nfev=100) for p in starts),key=lambda q:q.cost); p=sol.x; A=af(fn(u,p)); B=fit(A,v); e=np.std(A@B-v); results.append((e,name,p,lambda x,fn=fn,p=p,B=B:af(fn(x,p))@B))
 for k,f in F.items():
  for style in [0,1,2]:
   for phase in [False] if style!=2 else [True,False]:
    if style==0: fn=lambda u,p,f=f:f(u,p[0]); starts=[[p] for p in [.3,1,2,3,5]]; bounds=([.01],[7])
    elif style==1: fn=lambda u,p,f=f:np.c_[u,f(u,p[0])]; starts=[[p] for p in [.3,1,2,3,5]]; bounds=([.01],[7])
    elif not phase: fn=lambda u,p,f=f:u+p[1]*f(u,p[0]); starts=[[p,a] for p in [.3,1.25,2.15,3,5] for a in [-.6,.4]]; bounds=([.01,-5],[7,5])
    else: fn=lambda u,p,f=f:f(u+p[1],p[0]); starts=[[p,a] for p in [1,2.15,3] for a in [-.3,.2]]; bounds=([.01,-2],[7,2])
    search(k+str(style)+str(phase),fn,starts,bounds)
 for k,l in [('sin','tanh'),('sin','cos'),('sin','atan'),('tanh','cos'),('atan','cos')]:
  search(k+'+'+l, lambda u,p,k=k,l=l:np.c_[u,F[k](u,p[0]),F[l](u,p[1])],[[1.25,2.15],[2.15,1.25],[2.8,1.5],[1.5,2.8]],([.1,.1],[6,6]))
 results.sort(key=lambda j:j[0]); print(key,out,[(round(a,7),b,np.round(c,4)) for a,b,c,_ in results[:14]],flush=True); return results
B=panel('panel_b','v'); C=panel('panel_c','v1'); D=panel('panel_c','v2')
ops={k+str(i):f for k,res in [('B',B),('C',C),('D',D)] for i,(_,_,_,f) in enumerate(res[:4])}
spaces={'z':z,**{k:f(z) for k,f in ops.items()}}; spaces.update({i+j:ops[j](spaces[i]) for i,j in itertools.product(ops,repeat=2) if i[0]!=j[0] and i[1]=='0' and j[1]=='0'});
for k in list(spaces):
 if len(k)==4:
  for j in ['B0','C0','D0']:
   if j[0] not in k: spaces[k+j]=ops[j](spaces[k])
def quad(a): return np.column_stack([a[:,i]*a[:,j] for i in range(a.shape[1]) for j in range(i,a.shape[1])]);
def assess(a):
 d=af(a); c=fit(d[:52],np.c_[y,T][:52]); e=d[52:72]@c-np.c_[y,T][52:72]; return np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]))
R=[]
for k,a in spaces.items():
 for p,d in [('lin',a),('pair',np.c_[a,a[:,0]*a[:,2]+a[:,1]*a[:,3],a[:,0]*a[:,1]+a[:,2]*a[:,3],a[:,0]*a[:,3]+a[:,1]*a[:,2]]),('quad',np.c_[a,quad(a)])]: R.append((*assess(d),k,p))
for i,j in itertools.combinations(['z','B0','C0','D0','B1','C1','D1','B0C0','C0B0','D0B0','B0D0'],2):
 a,b=spaces[i],spaces[j];
 for p,d in [('lin',np.c_[a,b]),('pair',np.c_[a,b,a*b]),('cross',np.c_[a,b,(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)]),('quad',np.c_[a,b,quad(np.c_[a,b])])]:R.append((*assess(d),i+' '+j,p))
for l in [0,1]:print('ranking',l,sorted(R,key=lambda r:r[l])[:22]);
np.savez('/tmp/spaces.npz',**spaces)
