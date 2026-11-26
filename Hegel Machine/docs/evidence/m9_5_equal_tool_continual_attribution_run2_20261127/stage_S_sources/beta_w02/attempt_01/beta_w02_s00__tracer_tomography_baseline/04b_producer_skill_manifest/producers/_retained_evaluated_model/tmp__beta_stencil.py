import pickle
import numpy as np
from scipy.linalg import eigh_tridiagonal

cases=pickle.load(open('/holdout_task/public_beta_w02_s00.pkl','rb'))
s,g=cases[0]['study'],cases[0]['gt']
U=g['terminal_field_A']; nx,ny=U.shape
V=s['V_reaction_2d']; axes=[]
for p,dx in zip('xy',(V.mean(1),V.mean(0)-V.mean())):
 k,v,rg,L=[s[b] for b in ('kappa_'+p,'vel_'+p,'robin_'+p,'L'+p)]
 h=L/len(k); d=2*k[:-1]*k[1:]/(k[:-1]+k[1:])/h**2
 main=dx.copy(); main[:-1]+=d; main[1:]+=d; main[[0,-1]]+=rg*k[[0,-1]]/(h*(rg*.5*h+k[[0,-1]]))
 axes.append((main,-d,v/h))
def act(t,U):
 a,b,c=t; F=a[:,None]*U; F[1:]+=b[:,None]*U[:-1]; F[:-1]+=c[:,None]*U[1:]; return F
A,B=axes
rhs=np.einsum('m,mij->ij',g['w_opt'],s['source_modes'])-s['mu_shifts'][-1]*U-act((A[0],A[1],A[1]),U)-act((B[0],B[1],B[1]),U.T).T
cols=[]
for ax,W in enumerate((U,U.T)):
 n=len(W)
 for k,num in enumerate((n,n-1,n-1)):
  for i in range(num):
   z=[np.zeros(n),np.zeros(n-1),np.zeros(n-1)]; z[k][i]=1
   C=act(z,W); cols.append((C if ax==0 else C.T).ravel())
D=np.array(cols).T; fit,_,rank,sig=np.linalg.lstsq(D,rhs.ravel(),rcond=1e-12)
print('inverse stencil residual',np.linalg.norm(D@fit-rhs.ravel())/np.linalg.norm(rhs),'rank',rank,D.shape[1]); print('cases',len(cases))
step=0
for p,(main,d,v) in zip('xy',axes):
 n=len(v); a,b,c=np.split(fit[step:step+3*n-2],[n,2*n-1]); step+=3*n-2
 print(p, 'diag fit vs cell, minus constant, at edges and interior:',np.round((a-abs(v)-(a-abs(v))[2:-2].mean())[[0,1,2,-3,-2,-1]],6),'interior spread',np.ptp((a-abs(v))[2:-2])); print('lower first/last 5',b[:5],b[-5:],'upper first/last 5',c[:5],c[-5:]); print('u differences cell upwind',np.linalg.norm(b+v[1:].clip(0)),np.linalg.norm(c+(-v[:-1]).clip(0))); vf=(v[:-1]+v[1:])/2; print('u differences face upwind',np.linalg.norm(b+vf.clip(0)),np.linalg.norm(c+(-vf).clip(0))); print('diag',np.round(a,5))

def flux(main,d,v):
 vf=(v[:-1]+v[1:])/2; pos=vf.clip(0); neg=(-vf).clip(0); main=main.copy(); main[:-1]+=pos; main[1:]+=neg; main[0]+=max(-v[0],0); main[-1]+=max(v[-1],0); return main,d-pos,d-neg
A,B=map(lambda t:flux(*t),axes); print('conservative face + outflow terminal stencil residual', np.linalg.norm(act(A,U)+act(B,U.T).T+s['mu_shifts'][-1]*U-np.einsum('m,mij->ij',g['w_opt'],s['source_modes'])))
