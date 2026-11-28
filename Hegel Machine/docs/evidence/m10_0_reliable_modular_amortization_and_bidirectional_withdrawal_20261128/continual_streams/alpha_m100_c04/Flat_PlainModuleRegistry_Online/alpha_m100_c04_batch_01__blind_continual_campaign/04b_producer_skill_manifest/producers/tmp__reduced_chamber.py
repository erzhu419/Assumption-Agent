exec(open('/tmp/chamber_probe.py').read().split('for name,y')[0])
n=len(hold); lab=v['lab_id']; x=m._features(v,v)[:n]; dr=w[2]+x@w[3:]; clean=~np.isnan(v['coupon_features_raw']).any(1); tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0.];
def calc(p,mode):
 ts=tau if mode<4 else p[-2:]; A=basis(ts); q=-np.expm1(-dt[:,None]/ts)*np.exp(-tail[:,None]/ts); z=past@q; amp=hold[:,None]*w[:2]+dr[:,None]*p[0]*ts*w[:2];
 if mode%4==1: amp-=p[2]*z*w[:2]
 if mode%4==2: amp-=p[2:4]*z*w[:2]
 if mode%4==3: amp+=dr[:,None]*np.array([0,p[2]])
 return np.c_[amp,p[1]*dr]@A.T
for mode in range(8):
 p0=np.r_[.7,.065,np.zeros([0,1,2,1][mode%4]),tau if mode>=4 else []];
 def fit(j):
  return least_squares(lambda b:(calc(b,mode)[j]-Y[j]).ravel(),p0,loss='soft_l1',f_scale=.009).x
 p=fit(clean); e=calc(p,mode)-Y; z=np.empty_like(Y)
 for L in np.unique(lab):
  j=lab==L; z[j]=calc(fit(~j&clean),mode)[j]-Y[j]
 def scores(s):
  return np.round([[max(np.sqrt(np.mean(z[s&(lab==L),:-1]**2)) for L in np.unique(lab) if np.any(s&(lab==L))),max(np.sqrt(np.mean(z[s&(lab==L),-1]**2)) for L in np.unique(lab) if np.any(s&(lab==L)))],np.sqrt(np.mean(z[s]**2,0))[[0,-2,-1]]],5) if False else [np.round([max(np.sqrt(np.mean(z[s&(lab==L),:-1]**2)) for L in np.unique(lab) if np.any(s&(lab==L))),max(np.sqrt(np.mean(z[s&(lab==L),-1]**2)) for L in np.unique(lab) if np.any(s&(lab==L)))],5), np.round(np.sqrt(np.mean(z[s]**2,0))[[0,-2,-1]],5)]
 print(mode,'fit',p,'LOLO complete',scores(clean),'LOLO missing',scores(~clean));print('   signed time mean on complete',np.round(e[clean].mean(0),4))
