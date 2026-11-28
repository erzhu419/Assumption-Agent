exec(open('/tmp/explore15.py').read().split("for k in ['panel_b'")[0])
from scipy.optimize import minimize_scalar
B=s['panel_b']; C=s['panel_c']; t=s['telemetry_vec'];
for p,key in [(B,'v'),(C,'v1'),(C,'v2')]:
 u,v=p['u'],p[key]; pairs=np.column_stack([u[:,i]*u[:,j] for i,j in [(0,1),(1,2),(2,3),(3,0)]]); print('\n',key); print('even+ intercept, ring, square LOOCV',err(aff(u*u),v),err(aff(pairs),v),err(aff(np.c_[u*u,pairs]),v));
 funs={'cos':lambda x,g:np.cos(g*x),'sin':lambda x,g:np.sin(g*x),'tanh':lambda x,g:np.tanh(g*x),'asinh':lambda x,g:np.arcsinh(g*x),'atan':lambda x,g:np.arctan(g*x),'rational2':lambda x,g:x*x/(1+g*x*x),'rational1':lambda x,g:x/(1+g*x*x),'exp':lambda x,g:np.expm1(g*x),'abs':lambda x,g:np.sqrt(x*x+g),'cosh':lambda x,g:np.cosh(g*x),'x2+sin':lambda x,g:x*x+0.22*np.sin(g*x),'sin+cos':lambda x,g:np.sin(g*x)-0.22*np.cos(g*x),'x+cos':lambda x,g:x-0.22*np.cos(g*x),'x+x2':lambda x,g:x+g*x*x,'x+sin':lambda x,g:x+0.22*np.sin(g*x),'sin+x2':lambda x,g:np.sin(g*x)+0.22*x*x};
 for name,fn in funs.items():
  def cost(g):
   A=aff(fn(u,g));return np.mean((v-A@fit(A,v))**2)
  q=minimize_scalar(cost,bounds=(.005,6),method='bounded'); cand=min([q.x,*t],key=cost); print(name,round(cand,4),np.round(err(aff(fn(u,cand)),v),6))
 print('resid by row, quad',np.round(np.linalg.norm(v-poly(u,2)@fit(poly(u,2),v),axis=1),4)); print('inputs output',np.round(np.c_[u,v][:10],3))
a=f._calibrate(s['panel_a'],t[0]); uu=f._invert(np.concatenate((tr['x'],s['test']['x'])),a); print('frozen params',a[:-1]); print('A median calibration:',np.median(np.abs(f._forward(s['panel_a']['u'],a)-s['panel_a']['v']))); A=aff(B['u']); coef=fit(A,B['v']); print('linear B singular',np.linalg.svd(coef[1:],compute_uv=False),'u recovery error',np.max(abs(uu-((w-coef[0])@np.linalg.inv(coef[1:])))))
