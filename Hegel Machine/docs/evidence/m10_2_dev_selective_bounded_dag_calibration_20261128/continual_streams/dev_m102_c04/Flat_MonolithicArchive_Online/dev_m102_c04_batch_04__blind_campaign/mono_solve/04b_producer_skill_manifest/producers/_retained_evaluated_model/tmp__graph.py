exec(open('/tmp/probe.py').read().split('for d in range')[0])
aff=lambda z:np.c_[np.ones(len(z)),z]
U=s['panel_c']['u']; V=s['panel_c']['v1']; W=s['panel_c']['v2']; c1=lambda z:np.array([z[:,i]*z[:,j] for i,j in [(0,1),(0,3),(1,2),(2,3)]]).T
w1=ls(aff(c1(U)),V,rcond=None)[0]
F1=lambda z:aff(c1(z))@w1
B=s['panel_b']; bw=ls(aff(B['u']),B['v'],rcond=None)[0]
Bf=lambda z:aff(z)@bw
Bi=lambda z:(z-bw[0])@np.linalg.inv(bw[1:])
H=-np.expm1(-t[:,None]*[1.795,.42]); Z=np.c_[y,ls(H,T.T,rcond=None)[0].T]
# Panel C2 uses a common coefficient on the even-to-linear ratio.
def de(z,k,c):return aff(z+c*z*z/(1+k*z*z))
def err(kc):
 F=de(U,*kc); return (F@ls(F,W,rcond=None)[0]-W).ravel()
res=least_squares(err,[.16,.54]); print('C2 shared k, ratio, SD',res.x,np.std(res.fun)); w2=ls(de(U,*res.x),W,rcond=None)[0]
F2=lambda z:de(z,*res.x)@w2
frames={'raw':x,'Ainv':q,'A':p,'B(Ainv)':Bf(q),'B(raw)':Bf(x),'Bi(Ainv)':Bi(q),'Bi(raw)':Bi(x)}
def metrics(F):
 w=ls(F[:52],Z[:52],rcond=1e-9)[0]; e=F@w-Z
 return np.array([np.mean(abs(e[:52,0])),np.mean(abs(e[52:,0])),np.mean(abs(e[:52,1:]@H.T)),np.mean(abs(e[52:,1:]@H.T))])
rows=[]
for a,za in frames.items():
 for b,zb in frames.items():
  P,R=F1(za),F2(zb); D=np.einsum('ni,nj->nij',P,R).reshape(-1,16)
  F=aff(np.c_[P,R]); fields={'main':F,'dot':np.c_[F,D[:,::5].sum(1)],'diag':np.c_[F,D[:,::5]],'full':np.c_[F,D],'raw+full':np.c_[F,zb,D],'crossInput':np.c_[F,np.einsum('ni,nj->nij',P,zb).reshape(-1,16)]}
  for h,G in fields.items():rows.append((metrics(G),a,b,h))
for row in sorted(rows,key=lambda r:max(r[0][1],r[0][3]))[:35]:print(row)
for a in ['raw','Ainv','A']:
 P,R=F1(frames[a]),F2(frames[a]); F=aff(np.c_[P,R]); w=ls(F[:52],Z[:52],rcond=None)[0]; print('main residual',a,metrics(F)); print(w)
