exec(open('/tmp/creep04_oof.py').read().split('for linear in')[0])
c=d['calibration_load_cells']; full={l:s._checks(c,l) for l in range(4)}; ref={l:s._supported_line(*full[l]) for l in full}
rng=np.random.default_rng(404)
for label in ('one spot','two random spots','two wide spots','one EMI','no spots on one lab','one spot across wing'):
 errors=[]
 for ell in full:
  x,y=full[ell]
  for i in range(40):
   checks=full.copy()
   if label=='one spot': ids=[i%len(x)]
   elif label=='two random spots': ids=rng.choice(len(x),2,replace=False)
   elif label=='two wide spots': ids=[np.argmin(x),np.argmax(x)]
   elif label=='one EMI': ids=np.arange(len(x))
   elif label=='no spots on one lab': ids=[]
   else:
    ids=[i%len(x)]
    peer=ell ^ 1
    checks[peer]=tuple(v[:0] for v in full[peer])
   xx, yy=x[ids],y[ids].copy()
   if label=='one EMI': yy[i%len(x)]+=(-1)**i*.35
   checks[ell]=xx,yy
   fits={l:s._supported_line(*xy) for l,xy in checks.items()}
   prior=s._archive_calibration(checks,fits) if any(v is None for v in fits.values()) else {}
   for l in (ell,ell^1):
    fit=fits[l] if fits[l] is not None else s._sparse_line(*checks[l],prior[l]); delta=fit-ref[l]
    errors.append(np.max(np.abs(delta[0]*np.array([-3.,3.])+delta[1])))
 print(label,'worst physical stress deviation',round(max(errors),6))
 assert max(errors)<.005, (label,max(errors))
