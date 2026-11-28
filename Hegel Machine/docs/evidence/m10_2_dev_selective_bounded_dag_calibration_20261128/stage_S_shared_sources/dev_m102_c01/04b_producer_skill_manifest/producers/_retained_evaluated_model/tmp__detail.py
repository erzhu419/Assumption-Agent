exec(open('/tmp/probe.py').read().split('ops=[]')[0])
from scipy.optimize import least_squares
from itertools import combinations
for k in ['panel_a','panel_b','panel_c']:
 p=s[k];u=p['u']; F=PolynomialFeatures(2); a=F.fit_transform(u)
 for v in [p[j] for j in p if j!='u']:
  w=lstsq(a,v)[0]; print(k,list(zip(F.get_feature_names_out(),w.round(3)))); print('res for linear', (v-a[:,:5]@lstsq(a[:,:5],v)[0])[:10].round(3))
T=s['eval_times']; _,d,V=np.linalg.svd(t[:52],full_matrices=False)
print('V',V[:2]); print('target increment mean/std',np.mean(np.diff(t,axis=1),axis=0),np.std(np.diff(t,axis=1),axis=0));
fs={'exp+lin':lambda p:np.array([-np.expm1(-p[0]*T),T]), 'two exp':lambda p:np.array([-np.expm1(-p[0]*T),-np.expm1(-p[1]*T)]), 'exp+log':lambda p:np.array([-np.expm1(-p[0]*T),np.log1p(p[1]*T)]), 'exp+exp2':lambda p:np.array([-np.expm1(-p[0]*T),(-np.expm1(-p[1]*T))**2]),'log+lin':lambda p:np.array([np.log1p(p[0]*T),T]), 'log+log':lambda p:np.array([np.log1p(p[0]*T),np.log1p(p[1]*T)]), 'exp+sin':lambda p:np.array([-np.expm1(-p[0]*T),np.sin(p[1]*T)]), 'exp+sat':lambda p:np.array([-np.expm1(-p[0]*T),T/(1+p[1]*T)])}
for nam,fn in fs.items():
 def e(p):
  A=fn(p); return t[:52]-lstsq(A.T,t[:52].T)[0].T@A
 p=least_squares(lambda p:e(p).ravel(),[2.,.5],bounds=(.00001,10.)).x; A=fn(p); c=lstsq(A.T,t.T)[0].T; print(nam,p,np.std(e(p)),np.std(t-c@A),np.quantile(c,[0,.5,1],axis=0))
for name,z in [('x',x),('a',x*np.array([1.12812,1.02248,1.02085,1.1123])+[-.0988,-.00079,.03321,.01353])]:
 F=PolynomialFeatures(3); a=F.fit_transform(z); w=lstsq(a[:52],np.c_[y,t@V[:2].T][:52])[0]; print(name,'cubic coefficients\n',list(zip(F.get_feature_names_out(),w.round(3))));
