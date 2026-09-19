from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .segmenter import Config, solve

app = FastAPI(title="字幕断句", version="1.3.0")

STATIC_DIR = Path(__file__).parent / "static"


class WordIn(BaseModel):
    text: str
    start: float
    end: float


class BoundaryIn(BaseModel):
    """词间隙边界约束: index = k 表示词 k-1 与词 k 之间的间隙(1 <= k <= len(words)-1)。"""
    index: int
    type: Literal["must_break", "no_break"]


class LockIn(BaseModel):
    """锁定段: 固定词范围 [word_range[0], word_range[1]](含端点)与内部换行。

    line_break = k 表示段内换行落在词 k-1 与词 k 之间(必须是段内词间隙,
    且不得位于不可拆短语内部); 为 null 表示锁定为单行。
    """
    word_range: list[int] = Field(min_length=2, max_length=2)
    line_break: int | None = None


class SegmentRequest(BaseModel):
    words: list[WordIn]
    phrases: list[list[int]] = Field(default_factory=list)
    shots: list[float] = Field(default_factory=list)
    boundaries: list[BoundaryIn] = Field(default_factory=list)
    locks: list[LockIn] = Field(default_factory=list)
    alternatives: int = Field(default=1, ge=1, le=5)
    config: dict = Field(default_factory=dict)


@app.post("/api/segment")
def segment(req: SegmentRequest):
    cfg_fields = {k: v for k, v in req.config.items() if k in Config.__dataclass_fields__}
    cfg = Config(**cfg_fields)
    try:
        return solve([w.model_dump() for w in req.words], req.phrases, req.shots, cfg,
                     boundaries=[b.model_dump() for b in req.boundaries],
                     alternatives=req.alternatives,
                     locks=[lk.model_dump() for lk in req.locks])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
