import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/chamber98.py').read())
from scipy.optimize import least_squares
X=m._features(a,c)[n:]; drift=th[2]+X@th[3:]; o=np.ones(len(H)); V=np.c_[Y,2*R]; slope,off=M[lab].T
HC=c['hold_channel_raw']; PC=c['past_channel_raw']; print('hold channels',np.round(HC[:5],3)); print('channel correlations with hold',np.round(np.corrcoef(np.c_[H,HC].T)[0],3))
ZC=np.einsum('nsc,sb->ncb',PC,W)
fields={'H,d':np.c_[H,drift], 'H,d,1':np.c_[H,drift,o], 'H,d,X':np.c_[H,o,X], 'H,d,d2':np.c_[H,drift,drift**2,o], 'H,d,Hd':np.c_[H,drift,H*drift,o], 'H,d,Z':np.c_[H,drift,Z,o], 'H,d,cmd':np.c_[H,drift,(H-off)/slope,o], 'H,d,invcmd':np.c_[H,drift,H*slope+off,o], 'H,d,Hchan':np.c_[HC,drift,o], 'H,d,Hchan,Zchan':np.c_[HC,drift,o,ZC.reshape(-1,8)]}
def fit(F,outer,start):
 def predict(q):
  ta=np.exp(q); T=np.vstack([np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1]*2]); Q=np.linalg.lstsq(F[outer],V[outer],rcond=None)[0]; C=np.linalg.lstsq(T,Q.T,rcond=None)[0].T; return F@C@T.T,C
 if start is None: q=np.log(tau)
 else: q=least_squares(lambda q:(predict(q)[0][outer]-V[outer]).ravel(),np.log(start),bounds=(np.log([.15,.8]),np.log([.8,15]))).x
 p,C=predict(q); return p,C,np.exp(q)
for nm,F in fields.items():
 for start in (None,[.475,3.]):
  pred,C,ta=fit(F,o.astype(bool),start); oo=np.zeros_like(V)
  for l in np.unique(lab):
   v=lab==l; oo[v]=fit(F,~v,start)[0][v]
  def score(E): return np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2))/2
  print(nm,bool(start),'tau',np.round(ta,4),'fit',np.round(score(pred-V),5),'worst',np.round(np.max([score((oo-V)[lab==l]) for l in np.unique(lab)],0),5),'C',np.round(C,4).tolist() if F.shape[1]<=5 else '')
