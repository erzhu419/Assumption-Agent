exec(open('/tmp/m99_probe.py').read().split("for d,c in")[0])
y,dy=obs(di); Y=np.c_[y,dy]
def basis(u): return np.r_[np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1][None,:]]
sol=least_squares(lambda p:(lambda a:(np.linalg.lstsq(a,Y.T,rcond=None)[0].T@a.T-Y).ravel())(basis(np.exp(p))),np.log([.5,2.5])); u=np.exp(sol.x); Z=np.linalg.lstsq(basis(u),Y.T,rcond=None)[0].T
p,h,d=cd
print('effective taus',u,'relative',u/tau)
print('raw-feature missing',np.isnan(di['coupon_features_raw']).sum(0))
print('ranges',np.min(np.c_[p,h,d],axis=0).round(3),np.max(np.c_[p,h,d],axis=0).round(3))
def gaps(u):
 z=np.zeros((len(h),len(u)))
 for j,s in enumerate(dt): z+=-np.expm1(-s/u)*(p[:,j,None]-z)
 return h[:,None]-z
g1,g2=gaps(tau),gaps(u)
D=d@th[2:]
P={'hold':h,'pmean':p@dt/sum(dt),'plast':p[:,-1],'pfirst':p[:,0],'g0':g1[:,0],'g1':g1[:,1],'q0':g2[:,0],'q1':g2[:,1],'drift':D,'strain0':di['strain_0'],**{f'x{j}':d[:,j+1] for j in range(4)},**{f'p{j}':p[:,j] for j in range(len(dt))}}
def regress(A):
 A=np.column_stack((np.ones(len(h)),A)); co=np.linalg.lstsq(A,Z,rcond=None)[0]; R=A@co-Z; leverage=np.einsum('ij,ji->i',A,np.linalg.pinv(A)); e=R/(1-leverage[:,None]); return np.sqrt(np.mean(e*e,axis=0)),co
for name,a in P.items():
 E,B=regress(a); print(name,np.round(E,4),np.round(B.T,3))
for names in [('hold','g0','g1'),('hold','q0','q1'),('hold','q0','q1','drift'),('hold','q0','q1','x0','x1','x2','x3'),('hold',*[f'p{j}' for j in range(len(dt))]),('x0','x1','x2','x3')]:
 E,B=regress(np.column_stack([P[k] for k in names])); print(names,E.round(5),B.T.round(3))
print('per-branch gap ratios',np.c_[Z,g2,Z[:,:2]/g2,D][:8].round(4))
