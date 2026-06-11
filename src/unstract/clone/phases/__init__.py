"""Per-entity clone phases.

Each phase implements ``run(report)``, uses ``ctx.source`` / ``ctx.target``
to drive HTTP, records ``ctx.remap`` entries for downstream phases.

Dependency order is owned by ``orchestrator.clone`` — phases must NOT
call each other directly.
"""

from unstract.clone.phases.adapter import AdapterPhase
from unstract.clone.phases.api_deployment import APIDeploymentPhase
from unstract.clone.phases.base import Phase
from unstract.clone.phases.connector import ConnectorPhase
from unstract.clone.phases.custom_tool import CustomToolPhase
from unstract.clone.phases.files import FilesPhase
from unstract.clone.phases.group import GroupPhase
from unstract.clone.phases.pipeline import PipelinePhase
from unstract.clone.phases.tag import TagPhase
from unstract.clone.phases.tool_instance import ToolInstancePhase
from unstract.clone.phases.workflow import WorkflowPhase
from unstract.clone.phases.workflow_endpoint import WorkflowEndpointPhase

__all__ = [
    "APIDeploymentPhase",
    "AdapterPhase",
    "ConnectorPhase",
    "CustomToolPhase",
    "FilesPhase",
    "GroupPhase",
    "Phase",
    "PipelinePhase",
    "TagPhase",
    "ToolInstancePhase",
    "WorkflowEndpointPhase",
    "WorkflowPhase",
]
