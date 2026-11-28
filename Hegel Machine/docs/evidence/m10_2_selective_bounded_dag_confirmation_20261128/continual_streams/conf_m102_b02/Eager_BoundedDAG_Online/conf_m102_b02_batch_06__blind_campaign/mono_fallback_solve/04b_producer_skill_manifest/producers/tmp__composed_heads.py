from probe import *
import importlib.util
from scipy.optimize import minimize_scalar
sp=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
tr=s['train']; z=old.coords(s,tr['x']); z=z[0] if isinstance(z,tuple) else z
p=s['panel_b']; fb=lambda a:D(np.sign(a)*np.maximum(abs(a)-.35,0)**1.5); wb=fit(fb(p['u']),p['v']); B=lambda a:fb(a)@wb
p=s['panel_c']; fc=lambda a:D(a*np.roll(a,-1,axis=1)); fd=lambda a:D(np.c_[a,np.cos(1.25*a)]); wc=fit(fc(p['u']),p['v1']); wd=fit(fd(p['u']),p['v2']); C=lambda a:fc(a)@wc; E=lambda a:fd(a)@wd
b=B(z); c=C(z); d=E(z); y=tr['y']; T=tr['traj']; t=s['eval_times']; print('times',t); print('time ratios',np.median(T/T[:,[-1]],axis=0)); print('traj example',T[np.argsort(T[:,-1])[::8]]);
print('cos fit',np.std(fd(p['u'])@wd-p['v2']));
g={'z':z,'b':b,'c':c,'d':d,'z+b':z+b,'z-b':z-b, 'products':z*b}
for nm,op in [('B',B),('C',C),('E',E)]:
 for kn,q in list(g.items())[:7]:
  if nm+kn not in ('Bz','Cz','Ez'):g[nm+kn]=op(q)
def ch(q):return (q-q.mean(0))/np.maximum(q.std(0),1e-6)
from itertools import combinations
for target in [y,T[:,-1],T[:,0]]:
 cand=[]
 for names in [(k,) for k in g]+list(combinations(g,2)):
  a=np.column_stack([ch(g[nm]) for nm in names]); F=D(a); fp=np.linalg.pinv(F[:52]);
  def check(h,inv):
   w=fp@h[:52]; e=inv(F@w)-target; return np.mean(abs(e[:52])),np.mean(abs(e[52:]))
  cand.append((*check(target,lambda x:x),names,'linear',0))
  for k,lo,hi in [('tanh',max(abs(target))*1.00001,10),('sinh',.15,10),('sin',max(abs(target))*1.00001,10),('erf',max(abs(target))*1.00001,10)]:
   from scipy.special import erfinv
   def run(scale):
    func,inv={'tanh':(np.arctanh,np.tanh),'sinh':(np.arcsinh,np.sinh),'sin':(np.arcsin,np.sin),'erf':(erfinv,erf)}[k]
    return check(func(target/scale),lambda p:scale*inv(p))
   m=minimize_scalar(lambda p:run(p)[0],bounds=(lo,hi),method='bounded',options={'maxiter':20});cand.append((*run(m.x),names,k,round(m.x,4)))
 print('TOP train',*[x for x in sorted(cand)[:9]],sep='\n'); print('TOP validation',*[x for x in sorted(cand,key=lambda x:x[1])[:7]],sep='\n')
