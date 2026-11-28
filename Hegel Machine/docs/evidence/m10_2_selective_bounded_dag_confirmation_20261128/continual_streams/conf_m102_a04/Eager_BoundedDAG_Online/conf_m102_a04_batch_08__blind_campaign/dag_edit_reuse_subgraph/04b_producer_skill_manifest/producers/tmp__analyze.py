import pickle, numpy as np, runpy, glob
from scipy.optimize import least_squares
from itertools import combinations
np.set_printoptions(precision=4,suppress=True,linewidth=150)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; old=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); lin=old['lin']; one=old['one']; off,sc=old['_align'](s['panel_a']); print('ALIGN',off,sc); a=s['panel_a']; print('alignment residual quantiles',np.quantile(abs(a['v']-off-a['u']*sc),[0,.5,.8,.9,1],axis=0)); z=(s['train']['x']-off)/sc; print('z extent',z.min(0),z.max(0)); y=np.c_[s['train']['y'],s['train']['traj'][:,[3,-1]]];
def inter(x):return np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(x.shape[1]),2)])
def score(fun,u,v,mask=None):
 f=fun(u); r=f@lin(f,v)-v; return np.sqrt(np.mean(r*r))
def forms(x):
 return [one(x),one(inter(x)),np.c_[one(x),inter(x)],np.c_[one(x),x*x,inter(x)],np.c_[one(x),x*x],one(np.tanh(x)),one(x/(1+abs(x))),np.c_[one(x),x*x/(1+.3*x*x)]]
for p in ['panel_b','panel_c']:
 d=s[p]; u=d['u'];
 for k,v in d.items():
  if k=='u': continue
  print(p,k,'baseline rms',np.std(v,axis=0)); print('basis RMSE',[score(f,u,v) for f in [lambda x,i=i:forms(x)[i] for i in range(8)]]); print('lin coefs',lin(one(u),v)); print('inter coefs',lin(np.c_[one(u),inter(u)],v))
  for fam,guess in [(lambda x,q:one(x/(1+q[0]*abs(x))+q[1]*x),[.85,0]),(lambda x,q:np.c_[one(x),x*x,x*x/(1+q[0]*x*x)],[.3]),(lambda x,q:np.c_[one(x),inter(x)/(1+q[0]*inter(x)**2)],[.3]),(lambda x,q:one(np.tanh(q[0]*x)),[1.]),(lambda x,q:np.c_[one(x),inter(x),x*x/(1+q[0]*x*x)],[.3])]:
   def res(q):
    f=fam(u,q); return (f@lin(f,v)-v).ravel()
   opt=least_squares(res,guess,bounds=([.001]+[-1]*(len(guess)-1),[10]*len(guess))); print('fit',opt.x,np.std(opt.fun))

def kernel(w,idx,te):
 w=(w-w[idx].mean(0))/np.maximum(w[idx].std(0),.1); G=w@w.T/w.shape[1]; d=np.maximum(G.diagonal()[:,None]+G.diagonal()-2*G,0); P=(1+.25*G)**2; return np.mean([K[np.ix_(te,idx)]@np.linalg.solve(K[np.ix_(idx,idx)]+r*np.eye(len(idx)),y[idx]) for K,r in [(P,3e-5),(P+np.exp(-.06*d),3e-4)]],0)
idx=np.arange(52); te=np.arange(52,72); c,d=np.hsplit(old['_panel_maps'](s['panel_c'],z),2);
for name,w in [('z',z),('c',c),('d',d),('cd',np.c_[c,d]),('cc',inter(c)),('zcc',np.c_[z,inter(c)]),('c+cc',np.c_[c,inter(c)]),('cd+cc',np.c_[c,d,inter(c)]),('zcd',np.c_[z,c,d]),('z+zz',np.c_[z,inter(z)])]: print(name,np.mean(abs(kernel(w,idx,te)-y[te]),axis=0))
