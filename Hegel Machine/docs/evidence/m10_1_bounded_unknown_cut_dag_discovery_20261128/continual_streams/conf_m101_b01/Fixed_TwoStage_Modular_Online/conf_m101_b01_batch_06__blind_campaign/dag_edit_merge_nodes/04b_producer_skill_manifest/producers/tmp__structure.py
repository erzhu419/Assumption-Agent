from panels import *
from scipy.optimize import least_squares
r=dict(np.load('/tmp/reg.npz')); u=r['u']; Y=r['Y']; z=s['train']['traj']; t=s['eval_times'];print('times',t,'telemetry',s['telemetry_vec'])
p=s['panel_c']; L=lambda w:np.c_[np.ones(len(w)),w]; C1=lambda w:np.c_[L(w),features(w)['cross']]; C2=lambda w:np.c_[L(w),np.log1p(.365*w*w)];print('C1',np.linalg.lstsq(C1(p['u']),p['v1'],rcond=None)[0].round(4));print('C2',np.linalg.lstsq(C2(p['u']),p['v2'],rcond=None)[0].round(4));
p=s['panel_b']; U,V=p['u'],p['v']; E=V-L(U)@np.linalg.lstsq(L(U),V,rcond=None)[0];print('B residual singular values',np.linalg.svd(E,compute_uv=False))
fams={ 'rational':(lambda w,t:(w+t[0]*w**3)/(1+t[1]*w*w),[.1,.8],[-2,.02],[2,6]), 'tanh_mix':(lambda w,t:np.tanh(w*t[0])+t[1]*np.sin(w*t[2]),[1,.2,2],[.1,-3,.1],[5,3,6]), 'sin_mix':(lambda w,t:np.sin(w*t[0])+t[1]*np.sin(w*t[2]),[1,.2,2],[.1,-3,.1],[6,3,6]), 'sat_offset':(lambda w,t:np.tanh(w*t[0]+t[1]),[1,0],[.1,-2],[6,2]), 'rat_offset':(lambda w,t:(w+t[1])/(1+t[0]*(w+t[1])**2),[1,0],[.02,-2],[6,2]), 'asinh':(lambda w,t:np.arcsinh(w*t[0]+t[1]),[1,0],[.1,-2],[6,2]),'erf':(None,[1,0],[.1,-2],[6,2]) }
from scipy.special import erf
fams['erf']=(lambda w,t:erf(w*t[0]+t[1]),*fams['erf'][1:]); best=(1,None)
for k,(g,init,lo,hi) in fams.items():
 for extra in ['none','lin','cross','square']:
  def mat(w,t):return np.column_stack([np.ones(len(w)),g(w,t)]+([] if extra=='none' else [dict(lin=w,cross=features(w)['cross'],square=w*w)[extra]]))
  def res(t):
   f=mat(U,t);P=np.linalg.pinv(f);return ((f@(P@V)-V)/np.maximum(.025,1-(f*P.T).sum(1))[:,None]).ravel()
  opt=least_squares(res,init,bounds=(lo,hi),max_nfev=100);sc=np.sqrt(np.mean(opt.fun**2)); print('B',k,extra,round(sc,6),opt.x.round(4))
  if sc<best[0]:best=(sc,mat(u,opt.x)@np.linalg.lstsq(mat(U,opt.x),V,rcond=None)[0])
np.save('/tmp/bnew.npy',best[1]); print('profile Vh',np.linalg.svd(z,full_matrices=False)[2][:3].round(3));
for name,w in [('u',u),('b',r['b']),('c1',r['c1']),('c2',r['c2'])]:
 print(name,'max correlations by response',[(float(np.max(np.abs(np.corrcoef(np.c_[F,Y].T)[:-2,-2:]),axis=0)[j])) for F in [w,w*w,w**3,np.sin(3*w),features(w)['cross']] for j in range(2)])
