exec(open('/tmp/compose.py').read().split('for j in [')[0]); from scipy.optimize import minimize_scalar
z=V['b']; print('times',s['eval_times']); print('z,y,T',np.round(np.c_[z,Y,T[:,[0,-1]]][:30],3)); P,Sv,H=np.linalg.svd(T,full_matrices=False); print('temporal vectors',np.round(H[:2],5)); t=s['eval_times'];
for k in [0.22,2.15,.8,1.2,2.6]:
 for a0 in (np.ones_like(t),t):
  d=np.c_[1-np.exp(-k*t),a0]; print('temporal',k,a0[0],np.mean(abs((d@np.linalg.lstsq(d,T.T,rcond=None)[0]).T-T)))
F={'linear':lambda g,x:x,'sin':lambda g,x:np.sin(g*x),'cos':lambda g,x:np.cos(g*x),'tanh':lambda g,x:np.tanh(g*x),'asinh':lambda g,x:np.arcsinh(g*x),'atan':lambda g,x:np.arctan(g*x),'rat':lambda g,x:g*x/(1+(g*x)**2),'sqrt':lambda g,x:np.sqrt(1+(g*x)**2),'cbrt':lambda g,x:np.cbrt(1+(g*x)**2),'log':lambda g,x:np.log1p((g*x)**2),'cosq':lambda g,x:np.cos(g*x*x),'sinq':lambda g,x:np.sin(g*x*x),'signeds':lambda g,x:np.sign(x)*np.sqrt(abs(x))}
for target in (Y,T[:,0],T[:,-1],P[:,1]*Sv[1]):
 results=[]
 for tag,fun in F.items():
  for lin in (False,True):
   def design(g):return ab(np.c_[z,fun(g,z)] if lin else fun(g,z))
   def solve(g):
    d=design(g); q=np.linalg.lstsq(d[idx],target[idx],rcond=1e-9)[0]; return d@q-target,q
   grid=np.linspace(.12,10,61); los=lambda g:np.mean(solve(g)[0][idx]**2); j=np.argmin([los(g) for g in grid]); g=minimize_scalar(los,bounds=(grid[max(0,j-1)],grid[min(j+1,len(grid)-1)]),method='bounded').x; r,q=solve(g); results.append((np.mean(abs(r[~idx])),np.mean(abs(r[idx])),tag,lin,g,q.round(3)))
 print('fits',sorted(results,key=lambda p:p[0])[:8])
