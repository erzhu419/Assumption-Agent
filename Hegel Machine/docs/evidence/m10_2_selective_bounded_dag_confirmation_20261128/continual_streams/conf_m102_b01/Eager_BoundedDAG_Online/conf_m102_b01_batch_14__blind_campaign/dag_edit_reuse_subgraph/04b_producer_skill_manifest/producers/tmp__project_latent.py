exec(open('/tmp/compose.py').read().split('for j in [')[0]); from scipy.optimize import least_squares
z=V['b']; ii=np.where(~idx)[0]; jj=np.where(idx)[0]; print('sorted by b2'); print(np.round(np.c_[z,Y,T[:,0],T[:,-1]][np.argsort(z[:,2])],3));
for k,h in [('c1',C),('c2',D)]: print(k,'at B',np.linalg.lstsq(ab(h(z)),Y,rcond=None)[0], 'at Z matrix',np.round(h(np.vstack([np.zeros((1,4)),np.eye(4),-np.eye(4)])),3));
F={'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'asinh':np.arcsinh,'sat':lambda v:v/np.sqrt(1+v*v),'inv3':lambda v:np.sinh(np.arcsinh(v)/3),'rat':lambda v:v/(1+v*v),'gauss':lambda v:np.exp(-v*v),'clip':lambda v:np.clip(v,-1,1),'soft':lambda v:np.sqrt(1+v*v)}
results=[]
for kind,f in F.items():
 for key,d in {'z':ab(z),'zc':ab(np.c_[z,C(z),D(z)]),'C':ab(C(z)),'t':ab(np.c_[z,np.arctan(2.15*z)]),'q':ab(np.c_[np.arctan(2.15*z),Q(np.arctan(2.15*z))])}.items():
  for m in [0,1,2,3,4,5]:
   latent=ab(z) if m==5 else (ab(C(z)) if m==4 else ab(z[:,m:m+1])); l=latent.shape[1];
   def design(p):return np.c_[d,f(latent@p)]
   def opt(p):
    a=design(p)[jj]; return a@np.linalg.lstsq(a,Y[jj],rcond=1e-8)[0]-Y[jj]
   g=np.linalg.lstsq(latent[jj],Y[jj],rcond=None)[0]; g/=max(np.std(latent[jj]@g),.03)
   best=None
   for start in [.6,1.6,4,8]:
    for shift in [0,.7,-.7]:
     p0=start*g; p0[0]+=shift
     p=least_squares(opt,p0,max_nfev=70)
     if best is None or p.cost<best.cost: best=p
   a=design(best.x); pred=a@np.linalg.lstsq(a[jj],Y[jj],rcond=1e-8)[0]; results.append((np.abs(pred[ii]-Y[ii]).mean(),np.abs(best.fun).mean(),kind,key,m,best.x.round(3)))
print('projection top cv',sorted(results,key=lambda r:r[0])[:18]);print('projection top train',sorted(results,key=lambda r:r[1])[:18]);
