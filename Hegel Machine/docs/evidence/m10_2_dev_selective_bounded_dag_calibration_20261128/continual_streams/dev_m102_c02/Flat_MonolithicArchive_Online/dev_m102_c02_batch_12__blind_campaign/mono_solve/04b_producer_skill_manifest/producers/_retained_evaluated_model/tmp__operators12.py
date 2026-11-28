exec(open('/tmp/probe12.py').read().split('for panel in [b,c]:')[0])
np.set_printoptions(precision=3,suppress=True)
pairs=list(it.combinations(range(4),2))
for P,key in [(b,'v'),(c,'v2')]:
 u,V=P['u'],P[key]; typ='B' if P is b else 'C'; print(typ)
 for name,H in [('even', np.c_[u*u,*[u[:,i]*u[:,j] for i,j in pairs]]),('linear',u),('quad',np.c_[u,u*u,*[u[:,i]*u[:,j] for i,j in pairs]])]:
  A=aff(H); W=fit(A,V); print(name,np.std(A@W-V),W.T)
 def score(name,fn,p):
  A=aff(fn(u,p)); W=fit(A,V); E=A@W-V; L=1-np.sum(A*np.linalg.pinv(A).T,axis=1); return np.mean(abs(E/L[:,None])),np.std(E),name,p
 def forms(u):
  l,r=u[:,[i for i,j in pairs]],u[:,[j for i,j in pairs]]
  return [('self',u),('pair',l*r),('plus',l+r),('minus',l-r),('sumprod',np.c_[u,l*r]),('cycp',u+np.roll(u,-1,axis=1)),('cycm',u-np.roll(u,-1,axis=1))]
 vals=[]
 for i,(name,_) in enumerate(forms(u)):
  for tag,g in [('cos',lambda q,k:np.cos(k*q)),('sin',lambda q,k:np.sin(k*q)),('atan',lambda q,k:np.arctan(k*q)),('rat',lambda q,k:q/(1+k*q*q)),('inv',lambda q,k:1/(1+k*q*q)),('abs',lambda q,k:abs(q)**k),('sqrat',lambda q,k:q*q/(1+k*abs(q)))]:
   for append in [False,True]:
    def fn(u,p):
     H=g(forms(u)[i][1],p[0]); return np.c_[u,H] if append else H
    def res(p):
     A=aff(fn(u,p)); return (A@fit(A,V)-V).ravel()
    for start in [.22,1.1,2.15,4.]:
     p=least_squares(res,[start],bounds=([.001],[8]),max_nfev=60).x
     vals.append(score((name,tag,append),fn,float(p[0])*np.ones(1)))
 print(*sorted(vals)[:20],sep='\n')
u=c['u'];V=c['v1'];p=least_squares(lambda p: (aff(np.arctan(p*u))@fit(aff(np.arctan(p*u)),V)-V).ravel(),[2.15]); print('C1 atan fitted',p.x,np.std(p.fun))
