exec(open('/tmp/explore15.py').read().split("for k in ['panel_b'")[0])
from scipy.optimize import minimize_scalar
B=s['panel_b']; C=s['panel_c']; t=s['telemetry_vec']; w=o['w']; n=len(tr['y'])
def r(u):return u*np.roll(u,-1,axis=1)
coef=fit(aff(B['u']),B['v']); u=np.linalg.solve(coef[1:].T,(w-coef[0]).T).T
print('handoff condition',np.linalg.cond(coef[1:]))
def prof(x,y,fn,lo,hi):
 def cost(g):
  a=aff(fn(x,g)); return np.mean((y-a@fit(a,y))**2)
 grid=np.linspace(lo,hi,25); j=np.argmin([cost(g) for g in grid]); return minimize_scalar(cost,bounds=(grid[max(j-1,0)],grid[min(j+1,len(grid)-1)]),method='bounded').x
bf=lambda x,g:np.cos(g*x)
g=prof(B['u'],B['v'],bf,0.4,5); bc=fit(aff(bf(B['u'],g)),B['v']); b=lambda x:aff(bf(x,g))@bc
cc=fit(aff(r(C['u'])),C['v1']); c=lambda x:aff(r(x))@cc
v2={'expm1':lambda x,g:np.expm1(g*x),'x+quad':lambda x,g:x+g*x*x,'x+quad-rat':lambda x,g:x+.45*x*x/(1+g*x*x),'x+sinquad':lambda x,g:x+.45*np.sin(g*x*x)/g,'x+cos':lambda x,g:x+.22*(1-np.cos(g*x)), 'x+softplus':lambda x,g:x+.45*np.log(np.cosh(g*x))*2/g**2, 'x+abs':lambda x,g:x+.45*2*(np.sqrt(1+g*x*x)-1)/g, 'x+atanquad':lambda x,g:x+.45*np.arctan(g*x*x)/g}
x=C['u']; y=C['v2'];
for k in range(1,6):
 d=aff(np.concatenate([x**j for j in range(1,k+1)],axis=1)); co=fit(d,y); print('v2 additive polynomial',k,err(d,y),np.round([np.sum(co[1:5]*co[1+4*j:5+4*j])/np.sum(co[1:5]**2) for j in range(k)],5))
best=[]
for k,fn in v2.items():
 g2=prof(x,y,fn,.001,4); A=aff(fn(x,g2)); sc=err(A,y); print('v2',k,g2,sc); best.append((sc[1],k,g2,fit(A,y)))
_,k,g2,co=min(best); d=lambda x:aff(v2[k](x,g2))@co
z=b(u); maps={'u':u,'b':z,'c(u)':c(u),'c(b)':c(z),'d(u)':d(u),'d(b)':d(z),'d(c(b))':d(c(z)),'c(d(b))':c(d(z)),'c(b)+d(b)':c(z)+d(z),'c(b)-d(b)':c(z)-d(z),'c(b)+.22d(b)':c(z)+.22*d(z),'c(b)-.22d(b)':c(z)-.22*d(z),'b+.22c(b)':z+.22*c(z),'b+c(b)':z+c(z),'b+d(b)':z+d(z),'bc':np.c_[z,c(z)],'bd':np.c_[z,d(z)],'cd':np.c_[c(z),d(z)],'bcd':np.c_[z,c(z),d(z)]}
fs={'lin':lambda x:x,'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'asinh':np.arcsinh,'atan':np.arctan,'inv3':lambda x:np.sinh(np.arcsinh(x)/3),'rat':lambda x:x/(1+x*x),'rat2':lambda x:x*x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x)}
for target in [tr['y'],tr['traj']]:
 scores=[]
 for k,Z in maps.items():
  x=Z[:n]
  for name,fn in fs.items():
   for g0 in [.45,1.25,2.15,2.6,4]:
    F=fn(g0*x)
    for typ in [0,1,2,3]:
     if typ==0:A=aff(F)
     elif typ==1:A=aff(np.c_[x,F]) if name!='lin' else aff(x)
     elif typ==2:A=poly(F,2) if x.shape[1]<=8 else aff(np.c_[x,F,F**2])
     else:A=aff(np.c_[x,F,poly(F,2)[:,1+F.shape[1]:]]) if x.shape[1]<=8 else aff(np.c_[x,F,r(F)])
     if A.shape[1]>=48:continue
     score=err(A,target)[1]; scores.append((score,k,name,g0,typ))
 print('head',target.shape, sorted(scores)[:32])
np.savez('/tmp/map15.npz',**maps)
