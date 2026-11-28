import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; b=s['panel_b']; c=s['panel_c']; a=s['panel_a']; k=np.linalg.norm(a['v'],axis=1)<4
u,v=a['u'][k],a['v'][k]; gain=np.mean((u-u.mean(0))*(v-v.mean(0)),0)/u.var(0); bias=v.mean(0)-gain*u.mean(0)
z=(s['train']['x']-bias)/gain; Y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; aff=lambda u:np.c_[np.ones(len(u)),u]; fit=lambda H,V:np.linalg.lstsq(H,V,rcond=None)[0]
def models(u):
 yield 'linear',u
 yield 'poly2',np.c_[u,*[u[:,i]*u[:,j] for i,j in it.combinations_with_replacement(range(4),2)]]
 for a in [.22,1.25,2.15,.5,1.,1.5,2.,3.,4.,.35]:
  for name,f in [('sin',lambda u:np.sin(a*u)),('tanh',lambda u:np.tanh(a*u)),('rational',lambda u:u/(1+a*u*u)),('arctan',lambda u:np.arctan(a*u)),('power',lambda u:np.sign(u)*abs(u)**a),('dead',lambda u:np.sign(u)*np.maximum(abs(u)-.22,0)**a),('cubic',lambda u:u+a*u**3)]:
   yield (name,a),f(u)
for panel in [b,c]:
 for label,V in list(panel.items())[1:]:
  u=panel['u']; vals=[]
  for name,h in models(u):
   A=aff(h); e=V-A@fit(A,V); loo=e/(1-(A*np.linalg.pinv(A).T).sum(1))[:,None]; vals.append((np.mean(abs(loo)),str(name),np.std(e)))
  print(label,sorted(vals)[:10])
F=lambda p:-np.expm1(-np.exp(p[:,None])*t); opt=least_squares(lambda p:(T@np.linalg.pinv(F(p))@F(p)-T).ravel(), np.log([.23,1.3]),gtol=1e-11); M=T@np.linalg.pinv(F(opt.x)); print('rates',np.exp(opt.x),'noise',np.std(M@F(opt.x)-T));print('correlations',np.corrcoef(np.c_[z,Y,M].T)[4:]);print('calib',gain,bias)
