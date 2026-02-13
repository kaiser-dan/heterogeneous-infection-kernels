from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field

class DataPrepConfig(BaseModel):
    name: str
    source: str
    backboning_method: str = "none" # e.g., 'threshold', 'none', 'disparity'
    backboning_params: Dict[str, Union[float, str]] = Field(default_factory=dict)
    recency: Dict[str, Union[bool, int]] = Field(default_factory=lambda: {"enabled": False, "window_hours": 24})
    kmax: int = 20

class InferenceConfig(BaseModel):
    name: str
    model: Dict[str, Union[str, int, float]]
    data_config: str
    q: float = 1.0
    exposure_metric: str = "kappa" # 'k' or 'kappa'
    wandb_enabled: bool = True
    
    # Internal: path to processed data
    processed_data_path: Optional[str] = None
