"""TeammateState definitions for LangGraph state machine."""

from typing import Any
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


class TeammateState(TypedDict):
    """Core agent state managed by LangGraph."""

    task_id: str
    task_description: str
    task_budget: int
    messages: list[Any]
    current_context: str
    current_step: str
    sub_agents_spawned: list[str]
    requires_approval: bool
    approval_granted: bool
    # Economy/Cost tracking fields
    total_cost: float  # Accumulated cost in USD
    cost_limit: float  # Budget limit in USD
    cost_history: list[dict]  # Per-request cost records
    # Privacy/Security fields
    privacy_violation: bool  # PII detected in LLM response
    intervention_reason: str  # Reason for intervention (budget/PII)


class TeammateStateModel(BaseModel):
    """Pydantic model for state validation and serialization."""

    task_id: str = Field(..., description="Unique identifier for the task")
    task_description: str = Field(..., description="Human-readable task description")
    task_budget: int = Field(default=1000, description="Token budget for the task")
    messages: list[Any] = Field(default_factory=list, description="Conversation history")
    current_context: str = Field(default="", description="Current working context")
    current_step: str = Field(default="init", description="Current graph node")
    sub_agents_spawned: list[str] = Field(default_factory=list, description="Spawned sub-agents")
    requires_approval: bool = Field(default=False, description="Needs human approval")
    approval_granted: bool = Field(default=False, description="Approval status")
    # Economy/Cost tracking fields
    total_cost: float = Field(default=0.0, description="Accumulated cost in USD")
    cost_limit: float = Field(default=0.50, description="Budget limit in USD")
    cost_history: list[dict] = Field(
        default_factory=list,
        description="Per-request cost records with model, tokens, and cost"
    )
    # Privacy/Security fields
    privacy_violation: bool = Field(
        default=False,
        description="PII detected in LLM response"
    )
    intervention_reason: str = Field(
        default="",
        description="Reason for intervention (budget/PII)"
    )

    def to_typeddict(self) -> TeammateState:
        """Convert to TypedDict for LangGraph compatibility."""
        return TeammateState(
            task_id=self.task_id,
            task_description=self.task_description,
            task_budget=self.task_budget,
            messages=self.messages,
            current_context=self.current_context,
            current_step=self.current_step,
            sub_agents_spawned=self.sub_agents_spawned,
            requires_approval=self.requires_approval,
            approval_granted=self.approval_granted,
            total_cost=self.total_cost,
            cost_limit=self.cost_limit,
            cost_history=self.cost_history,
            privacy_violation=self.privacy_violation,
            intervention_reason=self.intervention_reason,
        )
