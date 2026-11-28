exec(open('/tmp/calibrate.py').read().split('for mode,z in us.items():')[0].replace("['affine','cyclic','cubic','both','full']","['cubic']"))
z=us['cubic']; y=s['train']['y']; tr=s['train']['traj']; st=dict(s,train={k:w[:52] for k,w in s['train'].items()},test={'x':s['train']['x'][52:]})
from scipy.optimize import minimize_scalar

def fit(u,v,f,lim):
 opt=minimize_scalar(lambda a:loo(f(u,a),v)[0],bounds=lim,method='bounded');a=opt.x; p=np.linalg.lstsq(f(u,a),v,rcond=None)[0]; print('fit',a,opt.fun,'coef',p.round(4));return lambda x:f(x,a)@p
b=lambda x,a:np.c_[aff(x),x**3/(1+a*x*x)]
c1=lambda x,a:np.c_[aff(x),np.arctan(a*x)]
c2=lambda x,a:np.c_[aff(x),np.expm1(-a*x*x)]
B=fit(s['panel_b']['u'],s['panel_b']['v'],b,(0,2)); C=fit(s['panel_c']['u'],s['panel_c']['v1'],c1,(.1,5));D=fit(s['panel_c']['u'],s['panel_c']['v2'],c2,(.02,1));
bz=B(z);cz=C(z);dz=D(z); cb=C(bz);db=D(bz)
base={'raw':xs,'z':z,'b':bz,'c':cz,'d':dz,'cb':cb,'db':db,'bc':B(cz),'bd':B(dz)}
def score(c):
 r=f({'__input__':st,'node_00':{'c':c[:52],'query':c[52:72]}}); t=g({'node_01':r})['state_pred']; return tuple(np.round([np.mean(abs(r['y_hat']-y[52:])),np.mean(abs(t-tr[52:]))],6))
for name,c in base.items(): print('score',name,score(c),c.min(),c.max())
for name,c in base.items():
 for nm,d in base.items():
  if name>=nm:continue
  for sn,cc in [('sum',c+d),('diff',c-d),('prod',c*d),('divide',c/(1+d*d)),('other divide',d/(1+c*c)),('tanhgate',c*np.tanh(d)),('rev tanhgate',d*np.tanh(c))]:
   a=aff(cc);p=np.linalg.lstsq(a[:52],tr[:52],rcond=None)[0]; tm=np.mean(abs(a[52:72]@p-tr[52:]));
   if tm<.12: print('combo',name,nm,sn,score(cc))
for ns in [('b','c','d'),('cb','db'),('z','b','c','d'),('z','b','cb','db')]:
 a=aff(np.column_stack([base[n] for n in ns])); p=np.linalg.lstsq(a[:52],tr[:52],rcond=None)[0];print('pooled traj',ns,np.mean(abs(a[52:72]@p-tr[52:])),np.mean(abs(a[:52]@p-tr[:52])))
