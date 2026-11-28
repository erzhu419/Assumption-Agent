import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=3,suppress=True)
a,c,b=s['panel_a'],s['panel_c'],s['panel_b']; u=a['u'];v=a['v'];F=np.c_[np.ones(len(u)),u];p=least_squares(lambda q:(F@q.reshape(5,4)-v).ravel(),np.linalg.lstsq(F,v,rcond=None)[0].ravel(),loss='cauchy',f_scale=.18).x.reshape(5,4); e=v-F@p
print('A multivar parameters',p); print('A large row count',np.sum(abs(e)>2,axis=1));print('A sorted columns, paired x and residual',*[np.c_[u[:,j],e[:,j]][u[:,j].argsort()][::2] for j in range(4)],sep='\n');print('A residual abs quantile',np.quantile(abs(e),[.25,.5,.75,.8,.9],axis=0))
for p in (b,c):
 u=p['u']; y=p.get('v2',p.get('v'));print('P pairs sorted',*[np.c_[u[:,j],y[:,j]][u[:,j].argsort()] for j in range(4)],sep='\n');print('correlation',np.corrcoef(np.c_[u,y].T)[:4,4:]);
 for fun in [lambda x:x, np.tanh,lambda x:x*x,lambda x:np.abs(x),lambda x:np.sin(3*x),lambda x:np.cos(3*x)]: print(np.corrcoef(np.c_[fun(u),y].T)[:4,4:])
