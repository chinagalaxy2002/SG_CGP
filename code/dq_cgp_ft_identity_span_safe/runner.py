"""Runner alias for the identity-start, span-safe experiment."""

from code.dq_cgp_ft_improved.runner import DifferentialLRMomentRetrievalRunner


class IdentitySpanSafeRunner(DifferentialLRMomentRetrievalRunner):
    """Use strict Baseline inheritance and the proven differential LR schedule."""


__all__ = ["IdentitySpanSafeRunner"]
