exec(open('/tmp/chamber_models.py').read().split("summarize('benchtop'")[0])
basekeys=['f0','f1','steady','m0','m1']
for k in ['d2','d3','drift-offset','sq','q']:
 summarize('memory + '+k,basekeys+[k])
for k,v in {'hold-linear':h,'hold-drift':h*drift,'unit':np.ones_like(h),'abs-drift':abs(drift),'crossstate':z[:,0]-z[:,1]}.items():
 features[k+'0']=D(v)*g[0]; features[k+'1']=D(v)*g[1]; features[k+'s']=H(v)
 summarize(k+' transients',basekeys+[k+'0',k+'1']); summarize(k+' slope',basekeys+[k+'s'])
summarize('separate rate sensor offset',basekeys+['d-rate'])
features['rate1']=np.broadcast_to(np.r_[np.zeros(len(t)),1.],y.shape)
summarize('constant rate sensor offset',basekeys+['rate1'])
# A chamber drift lag approaches its steady value with a separate time constant.
for keys in (['steady'],['steady','m0','m1'],basekeys):
 def lagged(q0):
  c=np.exp(np.clip(q0,-4,4)); return D(drift)*np.r_[-c*(-np.expm1(-t/c)),-1.]
 # Allow the lag amplitude to be fitted, as well as its time scale.
 aa=basis(keys)
 def pred(p):return base0+aa@p[:-2]+p[-2]*lagged(p[-1])
 def fit(m):
  return min((least_squares(lambda p:(pred(p)-y)[m].ravel(), np.r_[np.zeros(len(keys)),.9,l],loss='soft_l1',f_scale=.014,max_nfev=160) for l in [-1,0,1]),key=lambda v:v.cost).x
 w=fit(np.ones(len(y),bool)); errors=[(y-pred(fit(~m)))[m] for m in labm]; print('separate drift lag',keys,np.round(w,4),'worst',np.round(np.max([[abs(e[:,:3]).mean(),abs(e[:,-4:-1]).mean(),abs(e[:,-1]).mean()] for e in errors],axis=0),4))
