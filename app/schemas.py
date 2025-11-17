from pydantic import BaseModel
from typing import Optional, List


class FaceBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class FaceDetectResponse(BaseModel):
    status: str
    count: int
    boxes: Optional[List[List[float]]] = None


class FaceMatchResponse(BaseModel):
    status: str
    person_id: Optional[str] = None
    filename: Optional[str] = None
    distance: Optional[float] = None


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
