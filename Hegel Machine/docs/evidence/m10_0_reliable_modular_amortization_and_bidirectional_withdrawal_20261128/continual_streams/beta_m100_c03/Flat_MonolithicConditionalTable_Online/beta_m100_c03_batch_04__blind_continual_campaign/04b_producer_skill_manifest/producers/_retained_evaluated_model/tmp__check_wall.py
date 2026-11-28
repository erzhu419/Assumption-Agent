exec(open('/tmp/wall_probe.py').read().split('folds=')[0])
from scipy.optimize import minimize
A=axes(('sealed','sealed')); B=axes(('periodic','periodic')); R=axes(('robin','robin')); allix=np.arange(len(y))
def responses(p):
 return response._responses(s,[a+p[j]*(b-a)+p[j+2]*(r-a) for j,(a,b,r) in enumerate(zip(A,B,R))])[0]
def evaluate(p, ix):
 S=responses(p); w,g=fit(S,ix,1e-8); return y-S@w-z@g
for p in ([0]*4,[1,0,0,0],[0,1,0,0],[1,1,0,0]):
 e=evaluate(p,allix); print('corner',p,'RMSE',np.sqrt(np.mean(e**2)),'bank',np.sqrt(np.mean(e.reshape(-1,4)**2,axis=0)),'rosette means',e.reshape(-1,4).mean(0))
for excluded in (None,0,1,2):
 tr=allix if excluded is None else allix[allix//ns!=excluded]; te=allix if excluded is None else allix[allix//ns==excluded]
 o=minimize(lambda p: np.mean(evaluate(p,tr)[tr]**2), [0.4,0.4,0.015,0.015],bounds=[(0,1),(0,1),(0,1),(0,1)],method='L-BFGS-B',options={'gtol':1e-10,'ftol':1e-13}); print('omit shift',excluded,'fit interpolation',o.x,'test RMSE',[(p,float(np.sqrt(np.mean(evaluate(p,tr)[te]**2)))) for p in ([0]*4,[1,0,0,0],o.x)])
