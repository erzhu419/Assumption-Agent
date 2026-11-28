import importlib.util,glob,pickle,numpy as np
from scipy.optimize import least_squares
m=importlib.util.module_from_spec(sp:=importlib.util.spec_from_file_location('arch',glob.glob('/tmp/hegel_continual_archive/*batch_03_solver.py')[0]));sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=m._calibrate(s['panel_a'],tr['x']); ls=m.lin; one=m.one
u=s['panel_c']['u']; unique=[('tanh',lambda w,q:np.tanh(q[0]*w)),('tanh+lin',lambda w,q:np.c_[w,np.tanh(q[0]*w)]),('cos',lambda w,q:np.cos(q[0]*w)),('cos+lin',lambda w,q:np.c_[w,np.cos(q[0]*w)]),('shiftcos',lambda w,q:np.cos(q[0]*w+q[1])),('shifttanh',lambda w,q:np.tanh(q[0]*w+q[1])),('sin',lambda w,q:np.sin(q[0]*w)),('softsign',lambda w,q: w/(1+q[0]*abs(w))+q[1]*w),('curvature',lambda w,q:np.c_[w,w*w,w*w/(1+q[0]*w*w)])]
ops=[]
for key in ['v1','v2']:
 Y=s['panel_c'][key]; best=[]
 for name,f in unique:
  q=[2] if name not in ['shiftcos','shifttanh','softsign'] else [2,0]
  def fun(p):
   d=one(f(u,p)); return (d@ls(d,Y)-Y).ravel()
  r=least_squares(fun,q,max_nfev=1000,gtol=1e-12); d=one(f(u,r.x)); err=np.std(r.fun);best.append((err,name,one(f(z,r.x))@ls(d,Y))); print(key,name,r.x,err)
 ops.append(min(best,key=lambda a:a[0])[2])
a=s['panel_a']; au=a['u']; av=a['v']; d=np.stack((np.ones_like(au),au,np.roll(au,-1,axis=1),au*np.roll(au,-1,axis=1),au**3),axis=-1);keep=np.linalg.norm(av,axis=1)<4
for j in range(4):
 w=np.stack([ls(d[keep,i],av[keep,i]) for i in range(4)]); r=np.linalg.norm(np.einsum('nik,ik->ni',d,w)-av,axis=1);keep=r<=max(.002,5*np.median(r))
print('calib',w,'resid',np.sort(r));print('z ranges',z.min(0),z.max(0));
b=s['panel_b']; b=one(z)@ls(one(b['u']),b['v']); g,h=ops; ag,ah=np.split(m._panel_maps(s['panel_c'],z),2,axis=1)
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def score(F):
 F=one(F);Y=np.c_[tr['y'],tr['traj']]; pred=F[52:]@ls(F[:52],Y[:52]); er=np.abs(pred-Y[52:]); fit=np.abs(F[:52]@ls(F[:52],Y[:52])-Y[:52]); return (np.linalg.matrix_rank(F),*np.round([er[:,0].mean(),er[:,1:].mean(),fit[:,0].mean(),fit[:,1:].mean()],6))
for name,G,H in [('archive',ag,ah),('best',g,h)]:
 base=np.c_[G,H]; p=G*H; print(name)
 for nm,d in [('lin',base),('dot',np.c_[base,p.sum(1)]),('diag',np.c_[base,p]),('cross',np.c_[base,cross(G,H)]),('pair',np.c_[base,cross(G,H),G**2,H**2]),('z',np.c_[base,z,p.sum(1)]),('zx',np.c_[base,z,cross(z,G),p.sum(1)]),('all',np.c_[base,cross(G,H),cross(G,G),cross(H,H)])]: print(nm,score(d))
