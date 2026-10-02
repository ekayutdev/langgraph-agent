"""Ready-made workflow patterns. Each returns spec data, not a runtime."""

from lgkit.patterns._fragment import Fragment
from lgkit.patterns.approval_gate import approval_gate
from lgkit.patterns.plan_execute_review import plan_execute_review

__all__ = ["Fragment", "approval_gate", "plan_execute_review"]
