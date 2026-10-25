"""Price research against the resources still available for that decision."""


def allocate_complete_research(minimum_resources, pending_minima, remaining):
    """Share the pool without shrinking a payable visit below its full minimum."""
    if any(remaining[key] < amount for key, amount in minimum_resources.items()):
        return None
    payable = [minimum for minimum in pending_minima
               if all(remaining[key] >= amount for key, amount in minimum.items())]
    return {key: min(value, max(minimum_resources[key], value // (1 + len(payable))))
            for key, value in remaining.items()}


def research_resource_share(costs, remaining):
    """The most constrained resource determines a complete action's pool share."""
    return max(*(costs[resource] / max(1, remaining[resource])
                 for resource in ("observation_cost", "measurement_cost", "compute_cost")
                 if remaining[resource] is not None), 1e-12)
