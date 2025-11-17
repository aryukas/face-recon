from pydantic import BaseModel


class FaceDetectResponse(BaseModel):
    status: str
    faces_detected: int


class FaceMatchResponse(BaseModel):
    status: str
    name: str | None = None
    distance: float | None = None


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
