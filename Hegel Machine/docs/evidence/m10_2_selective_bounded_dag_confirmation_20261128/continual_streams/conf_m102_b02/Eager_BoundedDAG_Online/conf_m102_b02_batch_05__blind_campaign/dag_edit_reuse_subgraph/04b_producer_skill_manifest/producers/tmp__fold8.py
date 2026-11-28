import pickle,numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u,v=map(s['panel_b'].get,('u','v'))
for label,fn in [('erf',erf),('tanh',np.tanh),('sin',np.sin),('exp',lambda t:t*np.exp(-t*t)),('soft',lambda t:t/np.sqrt(1+t*t))]:
 for typ in ['unit','separate','scale','shift','mix']:
  def fit(q,U=u):
   if typ=='unit': a,b=q;c=0
   elif typ=='separate':a,b=q[:4],q[4:8];c=0
   elif typ=='scale':a,b=q[:4],q[4];c=0
   elif typ=='shift':a,b,c=q[:4],q[4:8],q[8:12]
   else:a,b,d=q;c=0
   f=fn(a*U+c)+b*U
   if typ=='mix': f=f+d*np.sin(2*U)
   F=np.c_[np.ones(len(U)),f]; C=np.linalg.lstsq(F,v,rcond=1e-10)[0]; return F,C,F@C-v
  sols=[]
  for a in [.7,1.3,2,3]:
   b=-.95*a
   q={'unit':[a,b],'separate':[a]*4+[b]*4,'scale':[a]*4+[b],'shift':[a]*4+[b]*4+[0]*4,'mix':[a,b,0]}[typ];lo=np.full(len(q),-10.);lo[:(1 if typ in ['unit','mix'] else 4)]=.06
   sols.append(least_squares(lambda q:fit(q)[2].ravel(),q,bounds=(lo,np.full(len(q),10.)),max_nfev=260))
  o=min(sols,key=lambda o:o.cost); F,C,R=fit(o.x);print(label,typ,np.round(o.x,4),np.mean(R*R)**.5,'C',np.round(C,3).tolist() if typ=='unit' else '')
