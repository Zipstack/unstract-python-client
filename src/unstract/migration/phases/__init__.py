"""Per-entity migration phases.

Each phase implements ``run(report)``, uses ``ctx.source`` / ``ctx.target``
to drive HTTP, records ``ctx.remap`` entries for downstream phases.

Dependency order is owned by ``orchestrator.migrate`` — phases must NOT
call each other directly.
"""

from unstract.migration.phases.adapter import AdapterPhase
from unstract.migration.phases.api_deployment import APIDeploymentPhase
from unstract.migration.phases.base import Phase
from unstract.migration.phases.connector import ConnectorPhase
from unstract.migration.phases.custom_tool import CustomToolPhase
from unstract.migration.phases.files import FilesPhase
from unstract.migration.phases.pipeline import PipelinePhase
from unstract.migration.phases.tag import TagPhase
from unstract.migration.phases.tool_instance import ToolInstancePhase
from unstract.migration.phases.workflow import WorkflowPhase
from unstract.migration.phases.workflow_endpoint import WorkflowEndpointPhase

__all__ = [
    "APIDeploymentPhase",
    "AdapterPhase",
    "ConnectorPhase",
    "CustomToolPhase",
    "FilesPhase",
    "Phase",
    "PipelinePhase",
    "TagPhase",
    "ToolInstancePhase",
    "WorkflowEndpointPhase",
    "WorkflowPhase",
]
