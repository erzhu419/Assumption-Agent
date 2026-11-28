exec(open('/tmp/spatial_probe.py').read().split('for mode')[0])
from scipy.optimize import minimize_scalar
ls=lambda f,y:np.linalg.lstsq(f,y,rcond=1e-10)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
def roll(x):return np.roll(x,-1,axis=1)
def cal(x):return np.stack([np.ones_like(x),x,roll(x),x*roll(x),x**3],axis=-1)
F=cal(u); mask=np.linalg.norm(v,axis=1)<5; w=np.array([ls(F[mask,i],v[mask,i]) for i in range(4)]); x=np.r_[t['x'],s['test']['x']]; q=np.array([least_squares(lambda zz:np.einsum('ij,ij->i',cal(zz[None])[0],w)-xx,xx,max_nfev=80,gtol=1e-12).x for xx in x]); print('inverse',np.max(abs(np.einsum('nij,ij->ni',cal(q),w)-x)),q.min(),q.max());
C=s['panel_c']; p=C['u'];
def c2(x,mode,k=.8):
 if mode=='cyc': return np.c_[aff(x),x*roll(x)]
 if mode=='cyc_rat':return np.c_[aff(x),x*roll(x),x*x/(1+k*x*x)]
 if mode=='cyc_sat':return np.c_[aff(x),x*roll(x),x/(1+k*abs(x))]
 if mode=='cyc_rat_mix':return np.c_[aff(x),x*roll(x)/(1+k*(x-roll(x))**2),x*x/(1+k*x*x)]
 if mode=='cyc_sqdiff':return np.c_[aff(x),x*roll(x),(x-roll(x))**2/(1+k*(x-roll(x))**2)]
 if mode=='cyc_cubes':return np.c_[aff(x),x*roll(x),x**3]
 if mode=='joint_denom':return np.c_[aff(x),x*roll(x)/(1+k*x*x),x*x/(1+k*x*x)]
def eval(co,of,yco=None):
 if yco is None:yco=co
 out=[]
 for z in [np.arange(52,72),*np.array_split(np.random.default_rng(889).permutation(72),4)]:
  tr=np.setdiff1d(np.arange(72),z); ids=np.r_[tr,z]; temp=n1({'__input__':{**s,'train':{k:t[k][tr] for k in ('x','y','traj')}}}); yh=a['_ridge'](yco[tr],t['y'][tr],yco[z],1e-6); res=n2({'node_00':{'common':co[ids],'offdiag':of[ids],'y_hat':yh},'node_01':temp});out.append([abs(yh-t['y'][z]).mean(),abs(res['state_pred']-t['traj'][z]).mean()]);
 return np.array(out)
r=opts[0](q); ss=opts[1](q); B=s['panel_b']; b=aff(q)@ls(aff(B['u']),B['v']); pairs=list(combinations(range(4),2)); prod=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in pairs]);
co,of=a['_features'](r,ss); print('archived feat with cubiccal',eval(co,of));
for mode in ['cyc','cyc_rat','cyc_sat','cyc_rat_mix','cyc_sqdiff','cyc_cubes','joint_denom']:
 def loss(k):
  f=c2(p,mode,k);return np.linalg.norm(C['v2']-f@ls(f,C['v2']))
 k=minimize_scalar(loss,bounds=(.0001,7),method='bounded').x; ss=c2(q,mode,k)@ls(c2(p,mode,k),C['v2']);
 co,of=a['_features'](r,ss); print(mode,k,loss(k),eval(co,of));
 print('extra',eval(np.c_[co,q,prod(q),r*r,ss*ss],np.c_[of,prod(q)]))
for name,co in [('q',q),('qr',np.c_[q,r]),('r',r),('b',b),('qprod',np.c_[q,prod(q),q*q]),('qprod_r',np.c_[q,prod(q),q*q,r,prod(r),r*r]),('rprod',np.c_[r,prod(r),r*r]),('rbasis',np.c_[q,q/(1+.843*abs(q)),prod(q)])]:print(name,eval(co,np.c_[prod(q),prod(r),prod(b)]))
