import pickle,sys,numpy as np
from itertools import combinations
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ShuffleSplit
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal,solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; b,g=calibrate(s); z=(tr['x']-b)/g; t=s['eval_times']; Y=tr['traj']; y=tr['y']; p=s['panel_c']; u=p['u']; v=p['v2'];
lin=lambda z:np.c_[np.ones(len(z)),z]
cyc=lambda z:np.column_stack([z[:,i]*z[:,j] for i,j in [(0,1),(0,3),(1,2),(2,3)]])
S=lambda x:x+.55*x*x/np.sqrt(1+.39*x*x)
W=np.linalg.lstsq(lin(S(u)),v,rcond=None)[0]; B=np.linalg.lstsq(lin(s['panel_b']['u']),s['panel_b']['v'],rcond=None)[0]; C=np.linalg.lstsq(lin(cyc(u)),p['v1'],rcond=None)[0]; d=lin(S(z))@W; bz=lin(z)@B; cz=lin(cyc(z))@C
P=PolynomialFeatures(2).fit_transform(z); E=np.c_[lin(S(z)),cyc(z)]; rng=np.random.default_rng(128)
for train,test in [(np.arange(52),np.arange(52,72)),*list(ShuffleSplit(2,train_size=52,random_state=11).split(z))]:
 q,a=temporal(t,Y[train]); A=np.linalg.lstsq(q,Y.T,rcond=None)[0].T; T=np.c_[y,A]; st=dict(s,train={k:(v[train] if k!='summary' else v) for k,v in tr.items()},test={'x':tr['x'][test]}); out=solve_batch(st); print('BASE',np.mean(abs(out['y_hat']-y[test])),np.mean(abs(out['state_pred']-Y[test]))); print('rates',-np.log1p(-q[0])/t[0]);
 for nm,F in [('D',lin(S(z))),('D+C',E),('D+C+Z',np.c_[E,z]),('P+D',np.c_[P,S(z)])]:
  fit=F@np.linalg.lstsq(F[train],T[train],rcond=1e-10)[0]; print(nm,np.round(np.mean(abs(fit[test]-T[test]),0),4),np.round(np.std(fit[train]-T[train],0),4));
 if test[0]!=52: continue
 for nm,H in [('linear',z),('B',bz),('C',cz),('D',d),('ZC',np.c_[z,cz]),('ZD',np.c_[z,d])]:
  for typ in ['tanh','rad','sin','prod','rational']:
   F0=E; k=H.shape[1];
   def f(par):
    h=H@par[:k]+par[k];
    if typ=='tanh': w=np.tanh(h)
    if typ=='rad': w=h/np.sqrt(1+h*h)
    if typ=='sin': w=np.sin(h)
    if typ=='prod': w=h*(z@par[k+1:])
    if typ=='rational': w=h/(1+(z@par[k+1:])**2)
    return np.c_[F0,w]
   def calc(par,ret=False):
    F=f(par); W=np.linalg.lstsq(F[train],T[train],rcond=1e-9)[0]; return F@W-T
   best=None
   for i in range(5):
    p0=rng.normal(0,.6,k+1+(4 if typ in ('prod','rational') else 0)); opt=least_squares(lambda v:calc(v)[train].ravel(),p0,bounds=(-5,5),max_nfev=100); e=calc(opt.x); score=np.mean(e[train]**2)
    if best is None or score<best[0]: best=(score,e,opt.x)
   print(nm,typ,np.round(np.mean(abs(best[1][train]),0),4),np.round(np.mean(abs(best[1][test]),0),4),round(np.mean(abs(best[1][test,1:]@q.T)),4),np.round(best[2],3))
