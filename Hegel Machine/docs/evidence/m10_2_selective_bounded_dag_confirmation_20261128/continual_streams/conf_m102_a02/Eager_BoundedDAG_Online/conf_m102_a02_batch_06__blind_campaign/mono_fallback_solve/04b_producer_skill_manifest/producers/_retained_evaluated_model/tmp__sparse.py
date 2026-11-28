exec(open('/tmp/graph.py').read().split('D={}')[0]); z=F['inv']; p=s['panel_b']; u=p['u']; V=p['v']; h=np.c_[y,T]
func.update({'erf':__import__('scipy').special.erf,'sqrt':lambda a:a/np.sqrt(1+a*a),'rational_cube':lambda a:a**3/(1+a*a),'tanh_sq':lambda a:a*np.tanh(a)**2,'saturating_cube':lambda a:a-np.tanh(a),'sin_cubic':lambda a:a-np.sin(a),'tanh_cube':lambda a:np.tanh(a)**3,'exp_cubic':lambda a:a*(-np.expm1(-a*a))})
for name in ['rat','tanh','erf','sqrt','rational_cube','tanh_sq','saturating_cube','sin_cubic','tanh_cube','exp_cubic']:
 f=func[name]; e,w,B=panel_fit(p,'v',f,int(name in ('rational_cube','tanh_sq','saturating_cube','sin_cubic','tanh_cube','exp_cubic'))); j=int(len(B)==5)
 def calc(w,z): return np.c_[np.ones(len(z)) if j else aff(z),f(w*z)]
 g=calc(w,z)@B
 def stat(A):
  c=fit(A[:52],h[:52]); r=h-A@c
  return np.round([abs(r[:52,0]).mean(),abs(r[52:,0]).mean(),abs(r[:52,1:]).mean(),abs(r[52:,1:]).mean(),abs(r[52:,-1]).mean()],5),r,c
 print(name,round(e,7),round(w,5));
 for lag in (1,-1,0,2):
  v=z*np.roll(g,lag,axis=1); A=aff(np.c_[g,v]); vout,r,C=stat(A); print('lag',lag,vout,end='; ')
  if lag==1: A0=A
 print(); print('z, z^2, B^2, shared',stat(np.c_[A0,z])[0],stat(np.c_[A0,z,z*z])[0],stat(np.c_[A0,g*g])[0],stat(np.c_[A0,z,np.sin(z)])[0]);
 if name in ('rat','tanh','erf'):
  a,r,c=stat(A0); print('readouts',c[:,[0,-1]].T); print('resid correlations',np.corrcoef(np.c_[r[:,0],r[:,-1],z,z*z,g,g*g].T)[:2])
