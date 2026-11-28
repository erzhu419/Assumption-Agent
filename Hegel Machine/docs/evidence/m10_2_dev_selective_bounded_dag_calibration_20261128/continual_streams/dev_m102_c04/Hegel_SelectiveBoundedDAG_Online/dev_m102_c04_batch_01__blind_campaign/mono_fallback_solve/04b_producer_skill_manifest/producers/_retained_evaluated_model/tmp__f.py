from d import *
u,v=[s['panel_a'][k] for k in ('u','v')];m=np.linalg.norm(v-u,axis=1)<2
wd=np.array([ls(A(u[m])[:,[0,i+1]],v[m,i]) for i in range(4)]).T
b=ls(A(s['panel_b']['u']),s['panel_b']['v']);q=(x-wd[0])/wd[1];z=A(q)@b;zi=np.linalg.solve(b[1:].T,(q-b[0]).T).T

def loo(f,Y):
 U,S,V=np.linalg.svd(f,full_matrices=False); m=S>1e-9*S[0]; U,S,V=U[:,m],S[m],V[m];E=U@(U.T@Y)-Y; h=np.sum(U*U,axis=1);return np.mean(abs(E),axis=0),np.mean(abs(E/np.maximum(1-h,1e-12)[:,None]),axis=0)

def grams(p):
 F=poly(p,2);P=poly(p,3);a=p*p
 yield 'linear',A(p);yield 'quadratic',F;yield 'cubic',P
 for i,k in enumerate((.22,1.25)):
  for j,g in enumerate((1+k*a,1+k*abs(p),np.sqrt(1+k*a),np.cosh(k*p))):
   for lab,h in [('odd',p/g),('even',a/g)]:
    yield f'q + {i},{j},{lab}',np.c_[F,h]
   yield f'quad odd {i},{j}',poly(p/g,2)
  for lab,h in [('sin',np.sin(k*p)),('tanh',np.tanh(k*p)),('exp',np.exp(k*p)),('cos',np.cos(k*p))]:yield f'q + {i},{lab}',np.c_[F,h]
 for i,k in enumerate((.22,1.25)):
  for lab,g in [('norm',np.sum(a,axis=1)),('L1',np.sum(abs(p),axis=1))]:
   G=1+k*g[:,None];yield f'{lab} div {k}',F/G;yield f'{lab} div+ {k}',np.c_[F,F/G]
 for i in range(4):
  for j in range(i):
   for h,l in [(p[:,i]+p[:,j],'+'),(p[:,i]-p[:,j],'-'),(p[:,i]*p[:,j],'*')]:
    yield f'single {i}{l}{j}',np.c_[F,np.sin(1.25*h),np.cos(1.25*h),h/(1+.22*h*h),h*h/(1+.22*h*h)]

def panel():
 u=s['panel_c']['u'];U={'u':u,'B(u)':A(u)@b,'invB(u)':np.linalg.solve(b[1:].T,(u-b[0]).T).T}; Y=np.c_[s['panel_c']['v1'],s['panel_c']['v2']]; print('C cross corr',np.corrcoef(np.c_[u,Y].T)[:4,4:]); print('C polynomial coefficients',ls(poly(u,2),Y));
 for name,p in U.items():
  r=[]
  for lab,f in grams(p):
   a,c=loo(f,Y);r.append((lab,a,c))
  for k in range(8):
   print(name,k,[(a,round(b[k],4),round(c[k],4)) for a,b,c in sorted(r,key=lambda v:v[2][k])[:3]])

def train():
 Y=np.c_[y,T[:,0],T[:,-1]];out=[]
 for name,p in [('q',q),('z',z),('zi',zi)]:
  for lab,f in grams(p):
   a,c=loo(f[:52],Y[:52]); e=np.mean(abs(f[52:]@ls(f[:52],Y[:52])-Y[52:]),axis=0);out.append((name,lab,a,c,e))
 for j in range(3):
  print('target',j, [(n,l,round(a[j],4),round(c[j],4),round(e[j],4)) for n,l,a,c,e in sorted(out,key=lambda v:v[3][j])[:13]])
if __name__=='__main__':panel();train()
