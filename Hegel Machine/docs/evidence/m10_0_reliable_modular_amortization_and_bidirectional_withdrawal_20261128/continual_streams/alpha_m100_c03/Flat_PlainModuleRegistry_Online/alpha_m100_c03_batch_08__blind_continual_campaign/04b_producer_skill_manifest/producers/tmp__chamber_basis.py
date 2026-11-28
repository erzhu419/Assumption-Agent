import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/chamber_probe.py').read())
from scipy.optimize import least_squares
P,H=load(di); A=np.column_stack((H,P)); X=m._features(tr,di)[n:]; D=th[2]+X@th[3:]; Y=np.column_stack((di['instant_rate_obs'],y(di))); T=np.r_[0,t]

def relax(q): return np.r_[1/q,-np.expm1(-t/q)]
def drift(q): return q*relax(q)
L=np.r_[1,t]

def describe(name,pred):
 e=Y-pred;print(name,'rate/curve median',np.round([np.median(abs(e[:,0])),np.median(abs(e[:,1:]))],5),'rmse',np.round(np.sqrt(np.mean(e**2,axis=0))[[0,1,4,8,12]],4))

def proj(U,V):
 c=np.linalg.lstsq(U,Y,rcond=None)[0]; b=np.linalg.lstsq(V.T,c.T,rcond=None)[0].T
 return U@b@V,b

for base in (np.column_stack((np.ones(len(H)),A,D)),np.column_stack((np.ones(len(H)),A,X)),np.column_stack((np.ones(len(H)),H,P[:,-1],D))):
 V=np.array([relax(taus[0]),relax(taus[1]),L]); pred,b=proj(base,V); describe('fixed branches and linear drift',pred); print('coeff',np.round(b,3))
 fun=lambda p: (proj(base,np.array([relax(p[0]),relax(p[1]),drift(p[2])]))[0]-Y).ravel()
 f=least_squares(fun,[*taus,1],bounds=(.08,25),max_nfev=150);print('free',f.x); pred,b=proj(base,np.array([relax(f.x[0]),relax(f.x[1]),drift(f.x[2])]));describe('free branches and curved drift',pred);print('coeff',np.round(b,3))
 print('unrestricted first and last time',np.round(np.linalg.lstsq(base,Y,rcond=None)[0][:,[0,1,12]],3))
print('H and P quantiles',np.quantile(A,[0,.5,1],axis=0));print('diag features missing',np.isnan(di['coupon_features_raw']).sum(axis=0));print('asset keys',list(__import__('json').load(open('/holdout_task/public_asset_index.json'))))
