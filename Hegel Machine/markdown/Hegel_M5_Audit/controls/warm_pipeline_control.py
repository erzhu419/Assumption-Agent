import copy
import json
import random
from unittest.mock import patch

import hegel_machine.cognition.m5_temporal_multiscale_and_base_subtree as m5
from hegel_machine.cognition.hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex

def row(u, s):
    return {
        "u_past": 1.0, "u_now": u, "x_now": 0.2, "mem_conv": 0.0,
        "s_grad": s, "H": 0.6,
        "y": 7.8 * (1 - 0.14 * s / (1 + s)) * u / (1.2 + u) + 0.32 * u ** 0.5,
    }

train = [row(u, 1 + .01 * t) for u, t in ((.5,-1), (1.1,.6), (1.7,-.2), (2.3,.2), (3.,-.6), (3.6,1))]
confirm = [row(u,s) for u,s in ((.5,.25), (1.5,2.2), (3.5,.4))]
heldout = [row(u,s) for u,s in ((.8,2.0), (2.0,.3), (2.7,1.8), (3.7,.65), (1.2,1.6))]
extra = [row(u,s) for u,s in ((.55,.995), (1.35,1.006), (2.55,.998), (3.65,1.002))]
ds = m5.M5DisjointTaskDataset(
    task_id="audit_warm_confirmation_escape", suite="audit", subfamily="C_shared_structure_hetero_params",
    is_out_of_scope=False,
    probe_signatures=["sig:base_field_C1", "sig:sink_gradient_C2", "sig:monotone_concave"],
    coarse_channel_available=False, horizon_H=.6, tau_scale=1.,
    pilot_rows=train[:5], extra_dynamic_rows=extra,
    probe_pool={
        "nuisance_high_var_aux":{"id":"nuisance_high_var_aux", "measurement_cost":3.5, "reset_cost":0., "horizon_wait_cost":0., "row":None},
        "open_branch_discriminating_probe":{"id":"open_branch_discriminating_probe", "measurement_cost":2., "reset_cost":0., "horizon_wait_cost":0., "row":train[5]},
    },
    confirm_rows=confirm, heldout_rows=heldout,
)
out = {}
for warm in (True, False):
    forest, pb = m5._build_seed_forest_and_priors()
    if not warm:
        pb["siphon_warm_prior"]["warm_start_box"] = None
    original = m5._fit_candidate_model_class
    trace = []
    def recorded(model_class, train_rows, **kwargs):
        fit = original(model_class, train_rows, **kwargs)
        trace.append({
            "model_class":model_class,
            "train_rmse":m5._rmse([fit["predict_fn"](r) for r in train_rows], [r["y"] for r in train_rows]),
            "confirm_rmse":m5._rmse([fit["predict_fn"](r) for r in confirm], [r["y"] for r in confirm]),
            "used_warm_start":fit.get("used_warm_start"),
            "escaped_to_wide_fallback":fit.get("escaped_to_wide_fallback"),
        })
        return fit
    with patch.object(m5, "_fit_candidate_model_class", recorded):
        receipt = m5._run_genuine_solver_pipeline(
            ds, "Hegel_M5_Full_TemporalSubtree_Forest", forest=forest, priors_bundle=pb,
            hippo_idx=HippoRAGWormholeIndex(forest), flat_idx=FlatEqualMetadataIndex(forest),
            rng=random.Random(733),
        )
    out["warm_enabled" if warm else "wide_from_start"] = {"receipt":receipt, "fit_trace":trace}
print(json.dumps(out,indent=2,sort_keys=True))
