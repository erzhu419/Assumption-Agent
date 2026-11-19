"""Analytic algebra controls only: no Hegel or external physics execution.

Run with Python + SymPy. JSON values are finite and output rejects NaN/Infinity.
"""
import argparse
import json
import math
from pathlib import Path
import sympy as sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='thermo_contract_verified.json')
    args = ap.parse_args()
    s, kp, km = sp.symbols('s kp km', positive=True)
    alpha, beta, gamma, mobility = sp.symbols('alpha beta gamma mobility', positive=True)
    x, u, udot = sp.symbols('x u udot', real=True)
    F = alpha*x*x/2-beta*x*u+gamma*u*u/2
    Fx = sp.diff(F,x)
    p = sp.diff(F,u)
    dx = -mobility*Fx
    storage_residual = sp.simplify(Fx*dx+p*udot-(p*udot-mobility*Fx*Fx))
    chemical_transfer = sp.factor(1-kp/(s+kp+km))
    general_transfer = sp.factor(1-mobility*beta*beta/(gamma*(s+mobility*alpha)))
    astar, bstar, RT = sp.symbols('astar bstar RT', positive=True)
    chemical_sub = {alpha:RT*(1/astar+1/bstar), beta:RT/astar,
                    gamma:RT/astar, mobility:kp*astar/RT}
    bound_transfer = sp.factor(general_transfer.subs(chemical_sub).subs(bstar,kp*astar/km))
    L,R = sp.symbols('L R', positive=True)
    em_transfer = sp.factor(general_transfer.subs({alpha:1/L,beta:1/L,gamma:1/L,mobility:R}))
    out = {
      'scope':'Analytic symbolic identities and finite evaluations; not Hegel, LLM, or physical lab runs.',
      'symbolic': {
        'storage_identity_residual':str(storage_residual),
        'chemical_A_over_D':str(chemical_transfer),
        'general_normalized_p_over_u':str(general_transfer),
        'chemical_mapping_residual':str(sp.simplify(bound_transfer-chemical_transfer)),
        'em_normalized_p_over_u':str(em_transfer),
        'chemical_step_kappa':str(sp.factor((beta**2/(alpha*gamma)).subs(chemical_sub))),
      },
      'initial_conditions':'All transfer ratios are zero-increment-initial-state maps around the specified equilibrium.',
      'step_rows':[],
      'schur_examples':[],
      'chemical_readout_linearization':[]
    }
    for name, a,b,c,m,D in [('chemical',1.25,1.0,1.0,2.0,0.05),('em',0.5,0.5,0.5,3.0,0.05)]:
      for t in [0,0.1,0.5,1.0,3.0]:
        xx=b*D/a*(1-math.exp(-m*a*t))
        dxx=m*b*D*math.exp(-m*a*t)
        fx=a*xx-b*D
        pp=c*D-b*xx
        res=fx*dxx+m*fx*fx
        out['step_rows'].append({'model':name,'t':t,'x':xx,'p':pp,
          'p_over_frozen_response':pp/(c*D),
          'kappa':b*b/(a*c),'decay_rate':m*a,
          'storage_residual_after_step':res})
    for name,a,b,c in [('joint_convex',2.0,1.0,3.0),('internal_stable_only',1.0,2.0,1.0)]:
      disc=math.sqrt((a-c)**2+4*b*b)
      out['schur_examples'].append({'case':name,'F_xx':a,'K_frozen':c,
       'K_relaxed':c-b*b/a,'joint_eigenvalues':[(a+c-disc)/2,(a+c+disc)/2]})
    for delta_a in [0.01,0.1,1.0]:
      out['chemical_readout_linearization'].append({'a_star':1.0,'delta_a':delta_a,
       'exact_delta_mu_over_RT':math.log1p(delta_a),'linear_delta_mu_over_RT':delta_a})
    assert out['symbolic']['storage_identity_residual']=='0'
    assert out['symbolic']['chemical_mapping_residual']=='0'
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(target),'symbolic_controls':out['symbolic'],
     'max_abs_storage_residual':max(abs(r['storage_residual_after_step']) for r in out['step_rows'])},ensure_ascii=False,allow_nan=False))

if __name__=='__main__':
    main()
