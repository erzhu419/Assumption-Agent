exec(open('/tmp/discover.py').read().split('maps=[]')[0])
from scipy.special import erf
odd={
 'softsign_abs':lambda x,a:x/(1+a*abs(x)),
 'sqrt_abs':lambda x,a:x/np.sqrt(1+a*abs(x)),
 'exp_abs':lambda x,a:np.sign(x)*(-np.expm1(-a*abs(x))),
 'sqrt_subtract':lambda x,a:np.sign(x)*(np.sqrt(1+a*abs(x))-1),
 'log_square':lambda x,a:x*np.log1p(a*x*x),
 'tanh':lambda x,a:np.tanh(a*x),
}
even={
 'sqrt_subtract':lambda x,a:np.sqrt(1+a*x*x)-1,
 'log':lambda x,a:np.log1p(a*x*x),
 'rat':lambda x,a:x*x/(1+a*x*x),
 'sqrt_rat':lambda x,a:x*x/np.sqrt(1+a*x*x),
}
for entries,Y in ((odd,v),(even,w)):
 for name,f in entries.items():
  for add in ((False, True) if Y is v else (True,)):
   for dim in (1,4):
    def gr(x,p):
     h=f(x,np.exp(p)); return aug(np.c_[x,h] if add else h)
    def res(p):
     g=gr(u,p);return (g@fit(g,Y)-Y).ravel()
    r=least_squares(res,np.zeros(dim),bounds=(-5,5),ftol=1e-12,gtol=1e-12); print('odd' if Y is v else 'even',name,add,dim,loo(gr(u,r.x),Y),np.exp(r.x))
print('summary',s['train']['summary'],'telemetry',s['telemetry_vec'])
