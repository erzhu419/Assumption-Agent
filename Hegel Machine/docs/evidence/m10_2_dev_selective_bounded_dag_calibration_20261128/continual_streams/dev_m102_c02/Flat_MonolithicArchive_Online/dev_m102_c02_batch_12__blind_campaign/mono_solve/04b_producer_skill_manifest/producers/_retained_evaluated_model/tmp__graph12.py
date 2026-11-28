exec(open('/tmp/probe12.py').read().split('for panel in [b,c]:')[0])
np.set_printoptions(precision=5,suppress=True)
def prof(u,V,fn,p0,lo,hi):
 def res(p):
  A=aff(fn(u,p));return (A@fit(A,V)-V).ravel()
 r=least_squares(res,p0,bounds=(lo,hi),max_nfev=250,gtol=1e-12); A=aff(fn(u,r.x)); W=fit(A,V); l=1-np.sum(A*np.linalg.pinv(A).T,1); return np.mean(abs(r.fun.reshape(V.shape)/l[:,None])),np.std(r.fun),r.x,lambda q:aff(fn(q,r.x))@W
B=prof(b['u'],b['v'],lambda q,p:np.cos(q*p[0]),[2.6],[.5],[5])[-1]; w=B(z)
forms=[('atan',lambda q,p:np.arctan(q*p[0]+p[1]),[2.15,0],[.1,-4],[8,4]),('atan4',lambda q,p:np.arctan(q*p),[2.15]*4,[.1]*4,[8]*4),('atanlin',lambda q,p:np.arctan(q*p[0])+p[1]*q,[2.15,0],[.1,-2],[8,2]),('atan3',lambda q,p:np.arctan(q*p[0]+p[1]*q**3),[2.15,0],[.1,-2],[8,2]),('atan+lin',lambda q,p:np.c_[q,np.arctan(q*p[0])],[2.15],[.1],[8]),('softplus',lambda q,p:np.logaddexp(0,q*p[0]+p[1]),[2.15,0],[.1,-4],[15,4]),('softplus+lin',lambda q,p:np.c_[q,np.logaddexp(0,q*p[0]+p[1])],[2.15,0],[.1,-4],[15,4]),('exp',lambda q,p:np.exp(q*p[0]),[1],[.01],[5]),('sq+lin',lambda q,p:np.c_[q,q*q],[1],[0],[2]),('cos+lin',lambda q,p:np.c_[q,np.cos(q*p[0])],[1.27],[.1],[5]),('rat',lambda q,p:q/(1+p[0]*abs(q)**p[1])**p[2],[.62,1.07,1.57],[.0001,.1,.1],[8,6,8]),('rat+lin',lambda q,p:np.c_[q,q*q/(1+p[0]*q*q)],[.1],[0],[8]),('sqrt+lin',lambda q,p:np.c_[q,np.sqrt(1+p[0]*q*q)],[.3],[.0001],[8]),('power+lin',lambda q,p:np.c_[q,abs(q)**p[0]],[2],[.1],[5]),('softsig',lambda q,p:q/(1+np.exp(-p[0]*q-p[1])),[2,0],[.1,-4],[10,4])]
Ms=[]
for key in ['v1','v2']:
 vals=[]
 for name,fn,p0,lo,hi in forms:
  r=prof(c['u'],c[key],fn,p0,lo,hi); vals.append((*r[:2],name,*r[2:])); print(key, name,*r[:3])
 Ms.append(min(vals)[-1]); print('BEST',min(vals)[:4])
C,D=Ms; print('time, tele',t,s['telemetry_vec']); print('ranges',z.min(0),z.max(0),w.min(0),w.max(0))
F=-np.expm1(-np.array([.41101653,1.81952481])[:,None]*t); M=T@np.linalg.pinv(F); R=np.c_[Y,M]; a=s['telemetry_vec'][0]
groups={'z':z,'b':w,'c':C(z),'d':D(z),'cb':C(w),'db':D(w)}
quad=lambda q:np.stack([q[:,i]*q[:,j] for i,j in it.combinations_with_replacement(range(q.shape[1]),2)],1)
for k,q in list(groups.items()):groups[k+'2']=quad(q)
for k,l in it.combinations(list(groups)[:6],2):
 q,p=groups[k],groups[l]; groups[k+l]=q*p;groups[k+'/'+l]=q/(1+a*p*p)
for m in range(3):
 def err(H):
  H=aff(H); e=H@fit(H[:52],R[:52,m])-R[:,m]; return np.mean(abs(e[52:])),np.std(e[:52])
 vals=[(*err(H),k) for k,H in groups.items()]; vals+= [(*err(np.c_[H,G]),k+'+'+l) for (k,H),(l,G) in it.combinations(groups.items(),2)]
 print('target',m,sorted(vals)[:18]); print('plain',[(k,err(groups[k])) for k in list(groups)[:6]])
