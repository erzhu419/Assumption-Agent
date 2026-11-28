exec(open('/tmp/explore.py').read().split('ops={}')[0])
from scipy.optimize import least_squares
p=s['panel_b']; B=lstsq(aff(p['u']),p['v'])[0]; w=aff(z)@B
for name,a in [('raw',z),('B',w),('centered B',w-B[0])]:
 L=np.c_[aff(a),a[:,0]*a[:,2],a[:,1]*a[:,3]]
 def stat(D):
  r=D@lstsq(D[:52],y[:52])[0]-y
  return np.round([np.mean(abs(r[:52])),np.mean(abs(r[52:]))],6)
 print(name,'cubes',stat(np.c_[L,a**3]),'pure cubic+cubic pairs',stat(np.c_[L,a**3,*[a[:,i]**2*a[:,j] for i in range(4) for j in range(4) if i!=j]]))
 P=poly(a,3); c=lstsq(P[:52],y[:52])[0]; print('cubic coeff',c)
 labels=[str(ix) for d in (2,3) for ix in combinations_with_replacement(range(4),d)]
 D=L
 for it in range(9):
  R=P[:,5:]-D@lstsq(D[:52],P[:52,5:])[0]; ry=y-D@lstsq(D[:52],y[:52])[0]; sc=abs(R[:52].T@ry[:52])/np.maximum(np.linalg.norm(R[:52],axis=0),1e-12); j=np.argmax(sc);D=np.c_[D,P[:,5+j]];print('add',labels[j],stat(D))
 for tag,f,lo,hi in [('sat',lambda a,h:a/(1+h*a*a),-.15,4),('sqrt',lambda a,h:a/np.sqrt(1+h*a*a),-.15,4),('cube',lambda a,h:a+h*a**3,-1,2),('tanh',lambda a,h:np.tanh(h*a),.01,4),('sin',lambda a,h:np.sin(h*a),.01,3),('asinh',lambda a,h:np.arcsinh(h*a),.01,6)]:
  for mode in (0,1,2):
   def design(q):
    b=f(a,q[0]); d=b if mode!=1 else a
    return np.c_[aff(b), d[:,0]*d[:,2],d[:,1]*d[:,3]] if mode!=2 else np.c_[aff(b),np.sum(d[:,:2]*d[:,2:],axis=1)]
   def res(q):
    D=design(q)[:52]; return D@lstsq(D,y[:52])[0]-y[:52]
   r=least_squares(res,[.22],bounds=(lo,hi)); print(tag,mode,r.x,stat(design(r.x)))
