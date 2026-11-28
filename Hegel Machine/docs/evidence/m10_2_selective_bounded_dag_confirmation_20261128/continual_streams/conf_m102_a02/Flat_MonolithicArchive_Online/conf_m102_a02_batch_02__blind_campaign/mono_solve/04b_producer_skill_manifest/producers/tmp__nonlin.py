exec(open('/tmp/grammar.py').read().split('for k,z in Z.items():')[0].replace("print('A',A,'B',B)",''))
from scipy.optimize import minimize_scalar

def evalfit(F,Y=y,I=I,J=J):
 c=np.linalg.lstsq(F[I],Y[I],rcond=None)[0]; r=Y-F@c; return np.mean(abs(r[J])),np.sqrt(np.mean(r[I]**2)),c
z=Z['cal']; L=aff(z); Q=terms(z,p2); C=terms(z,p3)
print('dominant cubic',[(p,round(c,4)) for p,c in zip(p3,evalfit(np.c_[L,Q,C])[2][15:])]); print('y noise guesses')
r=[]
for name,z in Z.items():
 l=aff(z); q=terms(z,p2); qs=[l[:,1+j] for j in range(4)]+[z[:,i]+sg*z[:,j] for i,j in pr for sg in [-1,1]]
 for k,w in enumerate(qs):
  for form,f in [('rat',lambda u,a:1/(1+a*u*u)),('exp',lambda u,a:np.exp(-a*u*u)),('cos',lambda u,a:np.cos(np.sqrt(a)*u))]:
   for mode in range(3):
    def feat(a):
     g=f(w,a)[:,None]-1
     return [np.c_[l,q,g*l],np.c_[l,q,g*np.c_[l,q]],np.c_[l,q,(l[:,1:]+0.1)*g]][mode]
    a=minimize_scalar(lambda a:evalfit(feat(a))[1],bounds=(.001,3.),method='bounded').x
    r.append((*evalfit(feat(a))[:2],name,k,form,mode,round(a,5)))
print('best training',sorted(r,key=lambda a:a[1])[:18]);print('best cv',sorted(r)[:18]);
p=s['panel_c']; u=p['u']; L=aff(u); Q=terms(u,p2); C=terms(u,p3)
for key in ['v1','v2']:
 V=p[key]; print('panel',key,'range',V.min(),V.max())
 for name,vinv in [('identity',V),('asinh',np.arcsinh(V)),('atan',np.arctan(V)),('atanh1.6',np.arctanh(V/1.6)),('atanh2',np.arctanh(V/2)),('invquad',V*(1+0.22*V*V))]:
  res=[]
  for a in [L,np.c_[L,Q],np.c_[L,u**3],np.c_[L,u**2,u**3],np.c_[L,u**2],np.c_[L,Q,u**3]]:
   h=a@np.linalg.pinv(a); res.append(round(np.sqrt(np.mean(((vinv-h@vinv)/(1-np.diag(h))[:,None])**2)),6))
  print(name,res)
 print('quadratic',np.linalg.lstsq(np.c_[L,Q],V,rcond=None)[0].round(3))
