exec(open('/tmp/probe.py').read().split("for name,z in")[0])
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
p=PolynomialFeatures(2); F=p.fit_transform(w); print('terms',p.get_feature_names_out()); print('coef y, T',np.round(ls(F,np.c_[y,T]),4));
for name,z in [('u',u),('w',w)]:
 for k,v in (c.items() if name=='u' else [('y',y),('T',T)]):
  if k=='u':continue
  print(name,k,[(d,np.round(np.std((A:=PolynomialFeatures(d).fit_transform(z))@ls(A,v)-v,axis=0),5)) for d in (1,2,3)])
print('panel C quadratic coefficients'); print(np.round(ls(p.fit_transform(u),np.c_[c['v1'],c['v2']]),3));
for seed in range(4):
 es=[]
 for a,b in KFold(6,shuffle=True,random_state=seed).split(F): es.append(np.mean(abs(F[b]@ls(F[a],np.c_[y,T][a])-np.c_[y,T][b]),axis=0))
 es=np.array(es); print('CV',seed,es[:,0].mean(),es[:,0].max(),es[:,1:].mean(),es[:,1:].max())
for kind in ['lin','rat','exp2','log','sin','exp-sin','exp-cos']:
 def basis(q):
  a,b=q
  g=-np.expm1(-a*t)
  h={'lin':lambda:t,'rat':lambda:t/(1+b*t),'exp2':lambda:-np.expm1(-b*t),'log':lambda:np.log1p(b*t),'sin':lambda:np.sin(b*t),'exp-sin':lambda:np.exp(-a*t)*np.sin(b*t),'exp-cos':lambda:1-np.exp(-a*t)*np.cos(b*t)}[kind]()
  return np.array([g,h])
 def loss(q):
  B=basis(q);return (ls(B.T,T.T).T@B-T).flatten()
 r=min((least_squares(loss,[a,b],bounds=(.001,8)) for a in (.3,1.4,3) for b in (.25,1.1,3)),key=lambda r:np.mean(r.fun**2)); print('time',kind,np.round(r.x,5),np.sqrt(np.mean(r.fun**2)))
