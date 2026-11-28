exec(open('/tmp/spatial_probe.py').read().split('pa=s[')[0])
from scipy.optimize import minimize
pa=s['panel_a'];m=np.linalg.norm(pa['v']-pa['u'],axis=1)<4
for tag,x,y in [('A',pa['u'][m],pa['v'][m]),('C2',s['panel_c']['u'],s['panel_c']['v2'])]:
 def report(n,a):
  w=fit(a,y); z=a@w-y; h=np.sum(np.linalg.svd(a,full_matrices=False)[0][:,:np.linalg.matrix_rank(a)]**2,axis=1); print(tag,n,round(np.std(z),8),round(np.std(z/(1-h[:,None])),8),np.round(np.linalg.norm(w,axis=1),4))
 report('Q+cubic',np.c_[affine(x),pair(x),x**3]);report('Q+cubic+sq',np.c_[affine(x),pair(x),x**3,x*x]);report('Q+cubic+trip',np.c_[affine(x),pair(x),x**3,trip(x)]);report('Q+quartic',np.c_[affine(x),pair(x),x**4]);report('Q+fifth',np.c_[affine(x),pair(x),x**5]);
 for f,op in [('sin',lambda x,k:np.sin(k*x)),('odd rational',lambda x,k:x/(1+k*x*x)),('signed rational',lambda x,k:x/(1+k*abs(x))),('square rational',lambda x,k:x*x/(1+k*x*x))]:
  print(tag,f,scan(x,y,lambda x,k:np.c_[pair(x),op(x,k)],.02,6))
 a,b=x[:,[i for i,j in tri]],x[:,[j for i,j in tri]]
 for n,den in [('x2',x*x),('a2',a*a),('b2',b*b),('sum2',(a+b)**2),('dif2',(a-b)**2),('rad',a*a+b*b),('prod2',(a*b)**2)]:
  print(tag,'cubes with rat',n,scan(x,y,lambda x,k:np.c_[x**3,pair(x)/(1+k*den)] if n!='x2' else np.c_[pair(x),x**3/(1+k*den)],.001,5))
