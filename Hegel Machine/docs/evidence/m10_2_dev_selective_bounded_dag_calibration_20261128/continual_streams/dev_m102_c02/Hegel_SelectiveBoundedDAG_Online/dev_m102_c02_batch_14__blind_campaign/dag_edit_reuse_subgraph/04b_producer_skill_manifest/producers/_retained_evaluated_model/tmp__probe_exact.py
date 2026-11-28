exec(open('/tmp/probe.py').read().split('m=old._panel_maps')[0])
from scipy.optimize import least_squares,root
from itertools import product
u,v=a['u'],a['v']; kl=np.arange(4); opp=(kl+2)%4; si=np.linalg.lstsq(lin(u),v,rcond=None)[0]; err=np.linalg.norm(v-lin(u)@si,axis=1); keep=err<3; U,V=u[keep],v[keep]; print('A rows',len(U)); print('input keys',s.keys()); print('B/C rows',[(k,len(s[k]['u'])) for k in ['panel_b','panel_c']]);
F={'lin':lambda z,h:z,'poly':lambda z,h:z**3,'rat':lambda z,h:z/(1+h*z*z),'sqrt':lambda z,h:z/np.sqrt(1+h*z*z),'tanh':lambda z,h:np.tanh(h*z),'atan':lambda z,h:np.arctan(h*z),'sin':lambda z,h:np.sin(h*z),'gauss':lambda z,h:z*np.exp(-h*z*z),'soft':lambda z,h:z/(1+h*abs(z))};
results=[]
for own,cross in product(F,F):
 def dd(z,j,p):
  w=z[:,j]; q=z[:,opp[j]]; return np.column_stack([np.ones(len(z)),w]+([F[own](w,p[0])] if own!='lin' else [])+[F[cross](q,p[1])]);
 def fit(p,ret=False):
  ds=[dd(U,j,p) for j in kl]; co=[ls(d,V[:,j]) for j,d in enumerate(ds)]; rr=np.column_stack([d@c-V[:,j] for j,(d,c) in enumerate(zip(ds,co))]);return co if ret else rr.ravel();
 ini=[0.8,1.5]; p=least_squares(fit,ini,bounds=([.025,.025],[5.,5.]),max_nfev=55).x; results.append((np.sqrt(np.mean(fit(p)**2)),own,cross,p,fit(p,True)))
U2,V2=b['u'],b['v']; D=lambda z,h:np.c_[lin(z/(1+h*z*z)),z]; ob=lambda h:np.mean((D(U2,h)@ls(D(U2,h),V2)-V2)**2); h=minimize_scalar(ob,bounds=(.1,3),method='bounded').x; bb=ls(D(U2,h),V2); print('B fitted shape',h,ob(h)**.5)
y=s['train']['y']; t=s['train']['traj']; n=52
for rm,own,cross,p,beta in sorted(results,key=lambda r:r[0])[:18]:
 def forward(z):return np.array([beta[j]@np.array([1,z[j]]+([F[own](z[j],p[0])] if own!='lin' else [])+[F[cross](z[opp[j]],p[1])]) for j in kl]);
 inv=[root(lambda z:forward(z)-xx,xx) for xx in x]; z=np.stack([i.x for i in inv]); c=D(z,h)@bb; yp=sink._scalar_readout(c[:n],c[n:],y[:n]); tp=sink._trajectory(c[:n],c[n:],t[:n],s['eval_times']); print('A',rm,own,cross,p.round(5),'coef',beta[0].round(5),'inverse',sum(i.success for i in inv),np.max(abs(z)),'CV',np.mean(abs(yp-y[n:])),np.mean(abs(tp-t[n:])));
