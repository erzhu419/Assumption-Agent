"""Price research against the resources still available for that decision."""


def research_resource_share(costs, remaining):
    """The most constrained resource determines a complete action's pool share."""
    return max(*(costs[resource] / max(1, remaining[resource])
                 for resource in ("observation_cost", "measurement_cost", "compute_cost")
                 if remaining[resource] is not None), 1e-12)
