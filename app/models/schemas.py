from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    sessionId: str = Field(..., min_length=1, description="Unique session identifier")
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    reply: str
    tokensUsed: int = 0
    retrievedChunks: int = 0


class ErrorResponse(BaseModel):
    error: str


class HealthResponse(BaseModel):
    status: str = "healthy"
