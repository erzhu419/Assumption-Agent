"""Compare one or two executable research visits in response-loss units."""

from itertools import combinations

from .research_resources import research_resource_share


def choose_research_visit(offers, remaining, budget, *, allocation_for,
                          knowledge_tie_tolerance=0.0):
    """Buy only the first visit of the best affordable single or pair.

    Allocations are execution limits, including confirmation and later work
    allowed by that visit. The companion is an affordable option, not a second
    purchase: actual feedback and unspent resources must price it again.
    Comparing at most two visits gives a bounded opportunity cost, not an
    optimum over arbitrary research portfolios. An unsupported queue returns
    None without spending. A supported queue with no purchasable improvement
    returns a decision whose offer is None.
    """
    count = len(offers)
    if not count or any(offer.get("loss_kind") != "squared_error"
            or "expected_loss_reduction" not in offer for offer in offers):
        return None
    # Visit allocation depends on which kind executes first. Both directed
    # orders must be evaluated before calling a two-visit option affordable.
    compute_cost = count + count * (count - 1)
    if compute_cost > remaining["compute_cost"] or not budget.spend(compute_cost):
        return None
    available = {**remaining, "compute_cost": remaining["compute_cost"] - compute_cost}

    def allocate(index, others, resources):
        offer = offers[index]
        allocation = allocation_for(offer, [offers[other] for other in others], resources)
        if allocation is None or any(not (
                offer["minimum_resources"][key] <= allocation[key] <= resources[key])
                for key in resources):
            return None
        return dict(allocation)

    gains = [max(0., offer["expected_loss_reduction"]) for offer in offers]
    # Measured downstream knowledge value breaks ties only. It never becomes
    # fabricated immediate loss reduction or overrules a materially better
    # current experiment.
    knowledge_scores = [float(offer.get("knowledge_return_score", 0.)) for offer in offers]
    portfolios = []

    def portfolio_resource_share(allocations):
        # A tie should preserve more of the scarce pool for later visits.  Use
        # the allocations that can actually execute this portfolio, rather
        # than a forecast that may omit confirmation work.
        total = {key: sum(item[key] for item in allocations) for key in available}
        return research_resource_share(total, available)

    for index, gain in enumerate(gains):
        if gain <= 0.:
            continue
        allocation = allocate(index, [], available)
        if allocation is not None:
            portfolios.append({"indices": [index], "allocations": [allocation], "gain": gain,
                               "knowledge_score": knowledge_scores[index],
                               "resource_share": portfolio_resource_share([allocation])})
    for left, right in combinations(range(count), 2):
        if (gains[left] <= 0. or gains[right] <= 0.
                or offers[left]["task_position"] == offers[right]["task_position"]):
            continue
        orders = []
        for first, second in ((left, right), (right, left)):
            first_allocation = allocate(first, [second], available)
            if first_allocation is None:
                continue
            after_first = {key: value - first_allocation[key] for key, value in available.items()}
            second_allocation = allocate(second, [], after_first)
            if second_allocation is not None:
                orders.append({"indices": [first, second],
                    "allocations": [first_allocation, second_allocation],
                    "gain": gains[first] + gains[second],
                    "knowledge_score": knowledge_scores[first] + knowledge_scores[second],
                    "resource_share": portfolio_resource_share([first_allocation, second_allocation])})
        portfolios.extend(orders)

    decision = {"basis": "bounded_two_visit_squared_error", "candidate_count": count,
        "compute_cost": compute_cost, "remaining_resources": available,
        "portfolio_expected_loss_reduction": 0., "selected_indices": [],
        "companion": None, "opportunity_cost": 0.}
    if not portfolios:
        decision["reason"] = "no_affordable_positive_portfolio"
        return {"offer": None, "index": None, "allocation": None, "decision": decision}

    if knowledge_tie_tolerance < 0.:
        raise ValueError("knowledge_tie_tolerance must be nonnegative")
    max_gain = max(item["gain"] for item in portfolios)
    gain_tie_tolerance = float(knowledge_tie_tolerance) * max(abs(max_gain), 1.)
    gain_tied = [item for item in portfolios
                 if max_gain - item["gain"] <= gain_tie_tolerance]

    def first_visit_density(item):
        # Both orders of a feasible pair have the same total return. Preserve
        # the denser first purchase in the final comparison, where knowledge
        # value and total allocation may otherwise tie exactly.
        return gains[item["indices"][0]] / research_resource_share(item["allocations"][0], available)

    if knowledge_tie_tolerance:
        # Knowledge value can resolve differences below the forecast's stated
        # resolution, while a materially better immediate return still wins.
        best = max(gain_tied, key=lambda item: (item["knowledge_score"],
                                                 -item["resource_share"],
                                                 first_visit_density(item),
                                                 -item["indices"][0]))
        selection_basis = "measured_return_within_gain_resolution"
    else:
        best = max(portfolios, key=lambda item: (item["gain"], item["knowledge_score"],
                                                  -item["resource_share"],
                                                  first_visit_density(item),
                                                  -item["indices"][0]))
        selection_basis = "largest_affordable_single_or_pair_return"
    index = best["indices"][0]
    without = max((item["gain"] for item in portfolios if index not in item["indices"]), default=0.)
    retained = max((item["gain"] - gains[index] for item in portfolios
                    if item["indices"][0] == index), default=0.)
    opportunity_cost = max(0., without - retained)
    companion = None
    if len(best["indices"]) == 2:
        other = best["indices"][1]
        companion = {"index": other, "task_position": offers[other]["task_position"],
            "basis": offers[other].get("basis"), "expected_loss_reduction": gains[other],
            "allocation": best["allocations"][1]}
    decision.update(reason="largest_affordable_single_or_pair_return",
        portfolio_expected_loss_reduction=best["gain"], selected_indices=best["indices"],
        selected_knowledge_return_score=best["knowledge_score"],
        selected_resource_share=best["resource_share"],
        gain_tied_resource_shares={str(item["indices"]): item["resource_share"]
                                  for item in gain_tied},
        selected_expected_loss_reduction=gains[index], companion=companion,
        alternative_expected_loss_reduction=without,
        retained_companion_expected_loss_reduction=retained,
        opportunity_cost=opportunity_cost, net_expected_loss_reduction=gains[index] - opportunity_cost,
        knowledge_tie_tolerance=float(knowledge_tie_tolerance),
        gain_tie_tolerance=gain_tie_tolerance,
        gain_tied_portfolios=[item["indices"] for item in gain_tied],
        selection_basis=selection_basis)
    return {"offer": offers[index], "index": index,
            "allocation": best["allocations"][0], "decision": decision}
