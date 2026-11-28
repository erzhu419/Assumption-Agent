from operators import *
import pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; inv,B,C,D=operators(s); x=s['train']['x']; z=inv(x); c=C(z);d=D(z); v=B(z); T=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; n=52
p=lambda h,a: h@fit(h[:n],a[:n]); mae=lambda r: np.mean(abs(r[n:]),axis=0)
print('times',t,'SVD',np.linalg.svd(T,compute_uv=False));
for nm,u in [('C,D',np.c_[c,d]),('D',d),('C',c),('C,D,z',np.c_[c,d,z]),('C,D,Bquad',np.c_[c,d,v*v]),('C,D,CB,DB',np.c_[c,d,C(v),D(v)])]:
 h=one(u); r=T-p(h,T); a=np.c_[y,T]; print(nm,'trajectory',mae(r).round(4),'res SVD',np.linalg.svd(r,compute_uv=False).round(5)); print('y direct',mae(y-p(h,y))); print('y with traj',mae(y-p(np.c_[h,T],y)))
 if nm=='C,D':
  w=fit(h[:n],T[:n]); print('coef T',w.round(4)); print('coef y direct',fit(h[:n],y[:n]).round(4));
  print('res y T correlations',np.corrcoef(np.c_[y-p(h,y),r].T)[0].round(4)); print('y temporal diffs')
  for j in range(8):
   u2=np.c_[h,T[:,j]]; print(j,mae(y-p(u2,y)),fit(u2[:n],y[:n]).round(4));
q=lambda g:-np.expm1(-t[:,None]*g); g=least_squares(lambda g:(q(g)@fit(q(g),T[:n].T)-T[:n].T).ravel(),[.4,1.8]).x; a=fit(q(g),T.T).T; h=one(np.c_[c,d]); print('amp coef C,D',fit(h[:n],a[:n]).round(4));print('y via amp coef',fit(np.c_[h,a][:n],y[:n]).round(4)); print('sample residuals (amps vs y)\n',np.c_[a-p(h,a),y-p(h,y)][:20].round(4));
