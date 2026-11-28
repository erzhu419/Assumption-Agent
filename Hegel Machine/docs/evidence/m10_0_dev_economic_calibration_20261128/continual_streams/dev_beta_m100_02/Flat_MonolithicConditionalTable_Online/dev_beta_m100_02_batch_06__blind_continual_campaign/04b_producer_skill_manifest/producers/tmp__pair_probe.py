exec(open('/tmp/rosette_probe.py').read().split('for j in range(4):')[0])
from scipy.optimize import least_squares
from itertools import product
x,y=x[clean],y[clean]; iso=np.count_nonzero(x,axis=1)==1
f=lambda z:z+.25*np.sin(2.5*z)
for k in range(4):
 j=(k+2)%4; q=iso&(x[:,j]!=0); print('cross',k,np.c_[x[q,j],y[q,k]-np.median(y[iso&(x[:,j]==0)&(x[:,k]==0),k])].round(7))
# Separate additive cross response from effects depending on simultaneous excitation.
for kind in ['poly','trig','pairpoly','pairtrig','allpoly']:
 for degree in [3,5,7]:
  rr=[]; hold=[]
  for k in range(4):
   a=x[:,k]; b=x[:,(k+2)%4]; t=y[:,k]; c,d=x[:,(k+1)%4],x[:,(k+3)%4]; cols=[np.ones(len(a)),f(a)]
   if kind=='trig' or kind=='pairtrig': cols += [np.sin(2.5*b),1-np.cos(2.5*b)]
   cols += [b**i for i in range(1,degree+1)]
   if kind.startswith('pair') or kind=='allpoly': cols += [a**i*b**j for i in range(1,degree) for j in range(1,degree-i+1)]
   if kind=='pairtrig': cols += [f(a)*b,a*np.sin(2.5*b),b*np.sin(2.5*a),np.sin(2.5*a)*np.sin(2.5*b),np.cos(2.5*a)*b,np.cos(2.5*b)*a]
   if kind=='allpoly': cols += [c,d,c*d,a*c,a*d,b*c,b*d,c*c,d*d]
   D=np.column_stack(cols); r=D@np.linalg.lstsq(D,t,rcond=None)[0]-t; rr.append(np.max(abs(r)))
   cv=[]
   for i in range(5):
    q=np.arange(len(x))%5==i; cv.extend((D[q]@np.linalg.lstsq(D[~q],t[~q],rcond=None)[0]-t[q]).tolist())
   hold.append(np.max(np.abs(cv)))
  print(kind,degree,'fit max',np.round(rr,7),'CV max',np.round(hold,7))
# Test simple physical composition: nonlinear response to a shared pairwise mixture, plus independently nonlinear cross leakage.
a=x; b=np.roll(x,2,axis=1)
for name,make,initial in [
 ('f(a+cb)+db+eb^2',lambda a,b,p:[f(a+p[0]*b),b,b*b],[-.1]),
 ('f(a)+f(a+cb)+db+eb^2',lambda a,b,p:[f(a),f(a+p[0]*b),b,b*b],[-.2]),
 ('f(a+cb+db^2)+eb+fb^2',lambda a,b,p:[f(a+p[0]*b+p[1]*b*b),b,b*b],[-.12,.04]),
 ('f(a)+f(cb+db^2)+eb+fb^2',lambda a,b,p:[f(a),f(p[0]*b+p[1]*b*b),b,b*b],[-.5,.2]),
 ('f(a)+linear+sin/cos b',lambda a,b,p:[f(a),b,b*b,np.sin(p[0]*b),1-np.cos(p[0]*b)],[2.5]),
 ('f(a)+linear+rational b',lambda a,b,p:[f(a),b/(1+p[0]*b*b),b*b/(1+p[0]*b*b),b],[.4])]:
 rr=[]; pp=[]
 for k in range(4):
  def residual(p):
   D=np.column_stack([np.ones(len(x)),*make(a[:,k],b[:,k],p)]); return D@np.linalg.lstsq(D,y[:,k],rcond=None)[0]-y[:,k]
  m=least_squares(residual,initial,max_nfev=100); rr.append(max(abs(m.fun)));pp.append(m.x.round(4))
 print(name,np.round(rr,7),pp)
