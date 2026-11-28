import pickle,numpy as np, importlib.util
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
t=s['train']; x=t['x']; y=t['y']; T=t['traj'];
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m101_a01_batch_02_solver.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
aug=old._aug; ls=old._ls
bias,gain,B,Bi,z,w=old._operators(s); d=(x-bias)/gain; b=aug(d)@B
if __name__=='__main__':
 np.set_printoptions(precision=4,suppress=True,linewidth=140)
 u=s['panel_c']['u']; print('panel C residuals',[(np.mean(abs(f(u)-s['panel_c'][k])),np.max(abs(f(u)-s['panel_c'][k]))) for k,f in [('v1',z),('v2',w)]])
 for k,f in [('v1',z),('v2',w)]:
  cl=dict(zip(f.__code__.co_freevars,[c.cell_contents for c in f.__closure__])); print(k,cl['p'],cl['coeff'])
 for n in (52,56):
  m=old.solve_batch(dict(s,train={k:v[:n] if k!='summary' else v for k,v in t.items()},test={'x':x[n:]})); print('old',n,np.mean(abs(m['y_hat']-y[n:])),np.mean(abs(m['state_pred']-T[n:])))
 def score(f,target):
  a=aug(f); p=a@ls(a[:52],target[:52]); return np.mean(abs(p[52:]-target[52:])),np.mean(abs(p[:52]-target[:52]))
 for label,a in [('raw',x),('d',d),('b',b),('z(d)',z(d)),('z(b)',z(b)),('w(d)',w(d)),('w(b)',w(b)),('d+invB(w(d))',d+(w(d)-B[0])@Bi),('z(d)+w(d)',z(d)+w(d)),('z(b)+w(b)',z(b)+w(b))]:
  print(label,[[score(old._poly(a,range(1,k+1)),target) for k in (1,2,3)] for target in (y,T)])
 print('y poly coefficients\n',ls(aug(old._poly(d,[1,2,3])),y))
