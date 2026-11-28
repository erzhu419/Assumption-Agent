from graph08 import *
D=setup(s)(s['train']['x']);z,b,p,r=[D[k] for k in ('z','b','p','r')]; t=s['eval_times']; y=s['train']['y']; T=s['train']['traj']; N=52
idx=sorted({tuple(sorted((i,j,k))) for i in range(4) for j,k in [(0,1),(0,3),(1,2),(2,3)]}); C=np.column_stack([np.prod(z[:,i],axis=1) for i in idx]);dot=np.sum(p*r,axis=1); cross=p[:,0]*r[:,1]; groups=[np.c_[p,dot],np.c_[r,cross],np.c_[z,C]]
F=aff(*groups); W=ls(F[:N],np.c_[y,T][:N]); print('reduced cv',np.mean(abs((F@W-np.c_[y,T])[N:]),axis=0));print('coefficients',np.round(W,5))
for G in groups:
 E=T[:N]-np.c_[np.ones(N),*[a[:N] for a in groups if a is not G]]@ls(np.c_[np.ones(N),*[a[:N] for a in groups if a is not G]],T[:N]); _,s0,H=np.linalg.svd(E,full_matrices=False); h=H[0]; print('group residual sv/time',s0[:4],h/h[-1]);print('single exp',least_squares(lambda k:(1-np.exp(-k*t))/(1-np.exp(-k*t[-1]))-h/h[-1],[1]).x)
def fit(G,ix,r0,offset):
 def des(r,ii):
  h=-np.expm1(-t[None,:]*r[:,None]); A=np.column_stack([(g[ii,None,:]*hh[None,:,None]).reshape(len(ii)*len(t),-1) for g,hh in zip(G,h)]);return np.tile(np.eye(len(t)),(len(ii),1)) if len(G)==0 else (np.c_[A,np.tile(np.eye(len(t)),(len(ii),1))] if offset else A)
 def err(r):
  A=des(r,ix); return A@ls(A,T[ix].ravel())-T[ix].ravel()
 opt=least_squares(err,r0,bounds=(.01,12),ftol=1e-12,gtol=1e-10);A=des(opt.x,ix);w=ls(A,T[ix].ravel());return opt.x,(des(opt.x,np.arange(len(T)))@w).reshape(T.shape),w
ix=np.arange(N)
for label,G,r0 in [('groups',groups,[1.9,.55,1.3]),('swap cross',[np.c_[p,dot,cross],r,groups[2]],[1.9,.55,1.3]),('cross third',[groups[0],r,np.c_[z,C,cross]],[1.9,.55,1.3]),('cross separate',[groups[0],r,groups[2],cross[:,None]],[1.9,.55,1.3,.55])]:
 for off in [False,True]:
  rr,pr,w=fit(G,ix,r0,off); print(label,off,rr,'train',np.std((pr-T)[:N]),'test',np.mean(abs((pr-T)[N:]),axis=0),'weights',np.round(w,4))
# If the cubic and linear terms share a spatial latent with y, estimate it from the scalar data alone.
for ii in [np.arange(52),np.arange(72)]:
 Wy=ls(F[ii,:-0 or F.shape[1]],y[ii]); latent=groups[-1]@Wy[-groups[-1].shape[1]:]; rr,pr,w=fit([groups[0],groups[1],latent[:,None]],ii,[1.9,.55,1.3],False); print('ytied',len(ii),rr,np.std((pr-T)[ii]),np.mean(abs((pr-T)[52:]),axis=0),np.round(w,4))
