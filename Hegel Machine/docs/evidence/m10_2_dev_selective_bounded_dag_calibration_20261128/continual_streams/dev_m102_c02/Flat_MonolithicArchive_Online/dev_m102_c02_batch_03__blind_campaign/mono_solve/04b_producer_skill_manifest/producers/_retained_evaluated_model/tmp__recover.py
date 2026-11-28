exec(open('/tmp/probe.py').read().split('u,v=')[0])
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
from itertools import combinations
P=PolynomialFeatures(3).fit(X)
u,v=s['panel_a']['u'],s['panel_a']['v']; keep=np.max(abs(v),axis=1)<3
B=ls(P.transform(u[keep]),v[keep]); print('A nonzero',[(k,w.round(5)) for k,w in zip(P.get_feature_names_out(),B) if max(abs(w))>.001]); B[abs(B)<.001]=0; print('A residual',np.max(abs(P.transform(u[keep])@B-v[keep])))
powers=P.powers_; nz=np.any(B!=0,axis=1); bp=B[nz]; pw=powers[nz];
def cal(z): return np.prod(z[...,None,:]**pw,axis=-1)@bp
xx=np.r_[X,s['test']['x']]; z=np.array([least_squares(lambda q:cal(q)-x, x, max_nfev=200).x for x in xx]); print('inverse residual',np.max(abs(cal(z)-xx)),'range',z.min(0),z.max(0)); print('inverse worst', sorted(zip(np.linalg.norm(cal(z)-xx,axis=1),range(len(z))),reverse=True)[:8]);
q=s['panel_c']['u']; a=s['panel_c']['v1']; d=s['panel_c']['v2'];
for name,h in [('quad',PolynomialFeatures(2).fit_transform(q)),('diag2',np.c_[aff(q),q*q]),('diag3',np.c_[aff(q),q*q,q**3]),('diag4',np.c_[aff(q),q*q,q**3,q**4]),('tanh',np.c_[aff(q),np.tanh(q)]),('sin',np.c_[aff(q),np.sin(q)]),('diag2+a',np.c_[aff(q),q*q,a])]:
 print(name, np.std(h@ls(h,a)-a),np.std(h@ls(h,d)-d))
 if name in ('quad','diag3'):print('c1',ls(h,a).round(4),'c2',ls(h,d).round(4))
np.save('/tmp/coords.npy',z)
