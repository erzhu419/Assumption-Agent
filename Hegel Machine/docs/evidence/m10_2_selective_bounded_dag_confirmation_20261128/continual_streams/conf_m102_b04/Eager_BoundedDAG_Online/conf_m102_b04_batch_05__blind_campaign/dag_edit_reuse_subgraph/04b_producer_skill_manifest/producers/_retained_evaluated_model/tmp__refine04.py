exec(open('/tmp/search04.py').read().split('f=np.c_')[0])
a=s['panel_a']; u,v=a['u'],a['v']; lin=poly(u,1); good=np.argsort(np.linalg.norm(v-lin@np.linalg.lstsq(lin,v,rcond=None)[0],axis=1))[:-3]; u,v=u[good],v[good]; lin=poly(u,1)
print('panel A leave-one-out, residual by coordinate')
for name,F in [('linear',lin),('quad',poly(u,2)),('cubic',poly(u,3)),('addpoly',np.c_[lin,u*u,u**3,u**4])]+[(f'sin{w}',np.c_[lin,np.sin(w*u),np.cos(w*u)]) for w in (1,2,3,4,5,6)]:
 B=np.linalg.pinv(F); r=(v-F@(B@v))/(1-np.einsum('ij,ji->i',F,B))[:,None];print(name,np.sqrt(np.mean(r*r,axis=0)))
print('y vs traj',np.sqrt(np.mean((y-np.c_[np.ones(n),z]@np.linalg.lstsq(np.c_[np.ones(n),z],y,rcond=None)[0])**2)))
regs=tuple(np.geomspace(.0003,.1,12)); rs=[]
for h in np.linspace(.35,1.1,16):
 w=q/np.sqrt(1+h*q*q)
 for degree in (3,4,5,6):
  fx=poly(w,degree)
  ex=list(j for d in range(1,degree+1) for j in cwr(range(4),d)); num=np.array([0]+[len(set(j)) for j in ex]); deg=np.array([0]+[len(j) for j in ex]);
  for interact in (.08,.25,.5,1,2,4):
   for decay in (.5,1,2):
    X=fx*interact**(np.maximum(num-1,0)/2)*decay**(np.maximum(deg-2,0)/2); K=X@X.T; e=evaluate(K,regs);rs.append((h,degree,interact,decay,*e))
print('top scalar');print(*sorted(rs,key=lambda r:r[-2])[:8],sep='\n');print('top trajectory');print(*sorted(rs,key=lambda r:r[-1])[:15],sep='\n')
