exec(open('/tmp/w03_b04_eval.py').read().split('pred=np.empty_like')[0])
def rm(e): return np.sqrt(np.mean(e*e))
def score(label, out):
  e=out['predicted_strain']-gt['te_clean']; r=out['initial_rate_pred']-gt['te_true_rate0']
  miss=np.isnan(te['coupon_features_raw']).any(1); act=(abs(h[n:])>y)|(abs(z[n:]).max(1)>y)
  groups=[(te['lab_id']==l)&(act==q)&(miss==a) for l in range(4) for q in [False,True] for a in [False,True]]
  print(label,'overall/worst-slice curve',rm(e),max(rm(e[j]) for j in groups if j.any()),'rate',rm(r),max(rm(r[j]) for j in groups if j.any()))
  for a,b in [('estimated_taus','taus'),('hereditary_weights','theta'),('viscoplastic_weights','vp_theta')]: print(a,'max absolute error',np.max(abs(out[a]-gt[b])))
base=m.solve_alpha_w03_batch_04(d); score('public',base)
C=d['calibration_load_cells']; labs=C['lab_id']; ids=np.arange(len(labs)); truth=np.c_[gt['stress_gains'],gt['stress_biases']]
for wing in (0,1):
 for mode in ('absent','one-clean-one-EMI','three-EMI-in-sweep'):
  affected=np.isin(labs,[2*wing,2*wing+1]); keep=(~affected) | ((ids%10<2) if mode=='one-clean-one-EMI' else (affected if mode=='three-EMI-in-sweep' else False)); tab={k:v[keep].copy() for k,v in C.items()}; bad=np.isin(tab['lab_id'],[2*wing,2*wing+1])
  if mode=='one-clean-one-EMI': tab['true_stress'][bad & (ids[keep]%10==1)]+=.6
  if mode=='three-EMI-in-sweep': tab['true_stress'][bad & (ids[keep]%10<3)]+=[.45,-.6,.55]*2 if False else np.tile([.45,-.6,.55],2)
  fits=m._calibrate(tab,np.arange(4)); err=np.max(abs(np.array(list(fits.values()))-truth)); print(wing,mode,'worst calibration error',err); assert err<.003
  score(f'wing{wing} {mode}',m.solve_alpha_w03_batch_04(dict(d,calibration_load_cells=tab)))
tau = base['estimated_taus']
step = 1e-6
c, r = m._design(tau, x, h, z, dt, np.array([0., step/2, step]), y)
assert np.all(c[:, 0] == 0)
forward = c[:, 2] / step
# The first-order error of a branch with rate r and time constant tau
# is bounded by |r| * step / (2*tau). Linear columns are exact.
truncation = np.zeros_like(r)
truncation[:, [0, 1, 8]] = abs(r[:, [0, 1, 8]]) * step / (2 * tau[[0, 1, 0]])
roundoff = 1e-12 * np.maximum(1., abs(r))
assert np.all(abs(forward - r) <= truncation + roundoff)
richardson = 2 * c[:, 1] / (step/2) - forward
relative_error = np.max(abs(richardson - r) / np.maximum(1., abs(r)))
print('onset derivative: raw max error', np.max(abs(forward-r)), 'Richardson scaled max error', relative_error)
assert relative_error < 1e-10

sub=dict(d,yield_stress=1e6)
c,r=m._design(base['estimated_taus'],x[:n],h[:n],z[:n],dt,t,sub['yield_stress']); th=gt['theta']; sub['train']=dict(tr,strain_curves_obs=start[:,None]+c[:,:,:7]@th,instant_rate_obs=r[:,:7]@th); out=m.solve_alpha_w03_batch_04(sub); assert np.allclose(out['viscoplastic_weights'],0,atol=1e-12); print('subyield weights',out['viscoplastic_weights'])
