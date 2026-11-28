exec(open('/tmp/joint14.py').read().split('for tag,z in')[0]);from sklearn.linear_model import Ridge
from scipy.optimize import minimize_scalar
print('times',s['eval_times']);print('a',[(k,np.shape(v)) for k,v in s['panel_a'].items()]);print('ranges',u.min(0),u.max(0),U.min(0),U.max(0))
def tied(p,k,fn,fb,wb):
 X=p['u']; O=p[k]
 def dd(X,w,f):return aff(X+w*fn(f*X))
 def e(q):
  D=dd(X,*q);return (D@fit(D,O)-O).ravel()
 from scipy.optimize import least_squares
 for init in [(0.4,1.25),(-.4,1.25),(.4,2.15),(-.4,2.15)]:
  o=least_squares(e,init,bounds=([wb[0],fb[0]],[wb[1],fb[1]]));print(k,fn.__name__,o.x,np.std(o.fun))
  if init==(0.4,1.25) or np.std(o.fun)<best[0]:best=(np.std(o.fun),o.x)
 return lambda Z:dd(Z,*best[1])@fit(dd(X,*best[1]),O)
bs=tied(b,'v',np.sin,(.15,4.5),(-2,2));bt=tied(b,'v',np.tanh,(.15,4.5),(-2,2));ct=tied(C,'v1',np.tanh,(.15,4.5),(-2,2));cc=tied(C,'v2',np.cos,(.15,4.5),(-2,2));
from itertools import combinations
for bn,z in [('bs',bs(u)),('bt',bt(u)),('B2',B2(u))]:
 v=ct(z); w=cc(z); q=np.c_[z,v,w]; base=aff(q); groups=[('base',base),('qz',np.c_[base,poly(z,2)]),('qv',np.c_[base,poly(v,2)]),('qw',np.c_[base,poly(w,2)]),('qall',np.c_[base,poly(z,2),poly(v,2),poly(w,2)]),('qjoint',np.c_[base,poly(q,2)]),('cross',np.c_[base,z*v,z*w,v*w]),('cz',np.c_[base,poly(z,2),poly(z,3)]),('tanh',np.c_[base,poly(z,2),np.tanh(2.15*z),np.cos(1.25*z),np.tanh(1.25*v),np.tanh(1.25*w)])];print(bn)
 for name,d in groups:print(name,[met(d,r) for r in [0,.002,.03,.3]])
