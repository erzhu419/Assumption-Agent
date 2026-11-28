import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe_state import *
from scipy.linalg import cho_factor,cho_solve
# Compare parsimonious saturation laws with a smooth residual on the paired readout.
D=np.c_[base,paired]
Ds=np.c_[d(c),paired.sum(1)]
I0=np.arange(52)
Q=poly(np.c_[a,c],2)
perts={'a':a,'c':c,'z':z,'a+c':np.c_[a,c]}
for k,w in perts.items():
 perts[k]=w/np.sqrt(np.mean(w*w,axis=0))

def sat(F, A, I, kind, w=None):
 x=F@fit(F[I],A[I]); h0=x.std()
 if kind=='single':
  def predict(p):return np.tanh(F@p[:-1])*np.exp(p[-1])
  p=np.r_[fit(F[I],A[I])/(2*h0),np.log(2*h0)]
 elif kind=='radial':
  v=(w*w).sum(1)/w.shape[1]
  def predict(p):
   G=F/(1+p[0]*v[:,None]); return G @ fit(G[I],A[I])
  p=[.05]
 elif kind=='component':
  def predict(p):
   G=np.c_[D[:,:5],np.tanh(p[0]*c)/p[0],paired]; return G@fit(G[I],A[I])
  p=[.4]
 elif kind=='mixed':
  def predict(p):
   G=F.copy();G[:,5:9]=c/(1+p[0]*c*c);return G@fit(G[I],A[I])
  p=[.06]
 bounds=(0,10) if kind in ('radial','mixed') else ((.001,5) if kind=='component' else (-np.inf,np.inf))
 p=least_squares(lambda p:predict(p)[I]-A[I],p,bounds=bounds,max_nfev=150).x
 return predict(p),p

def resid_gp(A,I,w,length,ridge):
 dist=np.sum((w[:,None]-w[None,:])**2,axis=-1)/w.shape[1]
 R=np.sqrt(5*dist)/length; K=(1+R+R*R/3)*np.exp(-R)
 F=cho_factor(K[np.ix_(I,I)]+ridge*np.eye(len(I)))
 DI=D[I]; P=cho_solve(F,DI); V=cho_solve(F,A[I]); beta=fit(DI.T@P,DI.T@V)
 return D@beta+K[:,I]@(V-P@beta)

settings={'paired':lambda A,I:(D@fit(D[I],A[I]),[]),'singleC2':lambda A,I:sat(Ds,A,I,'single'),'singlePair':lambda A,I:sat(D,A,I,'single'),'tanhC2':lambda A,I:sat(D,A,I,'component'),'rationalC2':lambda A,I:sat(D,A,I,'mixed')}
for nm,w in perts.items():
 settings['radial '+nm]=lambda A,I,w=w:sat(D,A,I,'radial',w)
 for le in (.6,1.25,2.5):
  for reg in (.003,.04):
   settings[f'GP {nm} {le} {reg}']=lambda A,I,w=w,le=le,reg=reg:(resid_gp(A,I,w,le,reg),[])
results={k:[] for k in settings}
for I,J in splits:
 rates=least_squares(lambda l:(H(l)@fit(H(l),T[I].T)-T[I].T).ravel(),np.log([.4,2.]),bounds=(-5,3)).x
 h=H(rates); A=fit(h,T.T).T; fast=D @ fit(D[I],A[I,1]);
 for k,method in settings.items():
  slow,pv=method(A[:,0],I); pred=np.c_[slow,fast]@h.T
  results[k].append((np.mean(abs(pred[J]-T[J])),np.mean(abs(slow[I]-A[I,0])),pv))
for k,res in sorted(results.items(),key=lambda kv:max(r[0] for r in kv[1]))[:20]:
 print(k,np.round([r[0] for r in res],5),round(res[0][1],5),np.round(res[0][2],3))
print('slow baseline and nonlinear candidates')
for k in list(settings)[:9]:print(k,[round(r[0],5) for r in results[k]],np.round(results[k][0][2],3))
