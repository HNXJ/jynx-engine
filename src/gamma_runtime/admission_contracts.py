from dataclasses import dataclass
from typing import Literal, Optional

@dataclass(frozen=True)
class BackendContract:
    provider_id: str
    role: Literal["office_mac_lms_players", "windows_lms_guard_judge"]
    base_url: str
    auth_mode: Literal["none", "session_env_only"]
    api_style: Literal["openai_compatible"]

@dataclass(frozen=True)
class SlotContract:
    slot_id: str
    agent_id: str
    model_id: str
    backend_id: str
    role: Literal["player", "judge", "observer"]
    truth_mode: Literal["truth_safe_unverified"]

@dataclass(frozen=True)
class SlotVerificationResult:
    inventory_status: Literal["PASS", "FAIL"]
    model_present: bool
    smoke_attempted: bool
    http_status: Optional[int]
    response_present: bool
    adherence_status: Literal["PASS", "WARN", "FAIL", "NOT_CHECKED"]
    slot_verdict: Literal["PASS", "PARTIAL", "FAIL"]
