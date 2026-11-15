"""Request/response schemas for the corpus API."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Annotated

from .db import MAX_ARTICLES, MAX_TEXT_LEN, MIN_TEXT_LEN

ArticleId = Annotated[str, Field(min_length=1, max_length=128)]
ArticleText = Annotated[str, Field(min_length=MIN_TEXT_LEN, max_length=MAX_TEXT_LEN)]


class ArticleUpsert(BaseModel):
    id: ArticleId
    side: Literal["train", "test"]
    text: ArticleText


class BatchRequest(BaseModel):
    """One atomic corpus modification: additions/corrections plus deletions.

    The whole batch is validated first and applied in a single transaction;
    if anything fails, nothing takes effect.
    """

    upserts: list[ArticleUpsert] = Field(default_factory=list, max_length=MAX_ARTICLES)
    deletes: list[ArticleId] = Field(default_factory=list, max_length=MAX_ARTICLES)


class BatchResponse(BaseModel):
    changed: bool
    generation: int
    published_generation: int
    upserted: int
    deleted: int
    articles: int


# ---------------------------------------------------------------------------
# dual-source ingestion (采集双源核验)
# ---------------------------------------------------------------------------

SourceName = Literal["A", "B"]
IngestStatus = Literal["approved", "pending", "conflict", "retracted"]


class IngestReport(BaseModel):
    """One report from one crawl pipeline for one article.

    A content report carries the original text and its train/test side; a
    retraction (撤稿) carries neither. Revisions increase monotonically per
    (article, source): re-sending the same revision with the same content is
    an idempotent no-op, while the same revision with different content and
    any older revision are rejected.
    """

    source: SourceName
    id: ArticleId
    revision: Annotated[int, Field(ge=1)]
    retracted: bool = False
    side: Literal["train", "test"] | None = None
    text: ArticleText | None = None

    @model_validator(mode="after")
    def _consistent_payload(self):
        if self.retracted:
            if self.side is not None or self.text is not None:
                raise ValueError("retraction report must not carry side or text")
        else:
            if self.side is None or self.text is None:
                raise ValueError("content report requires both side and text")
        return self


class IngestReportResponse(BaseModel):
    accepted: bool
    replayed: bool  # true when the exact same revision+content was re-sent
    id: str
    source: SourceName
    revision: int
    status: IngestStatus
    corpus_changed: bool  # only approved-corpus changes bump the generation
    generation: int
    published_generation: int


class SourceVersion(BaseModel):
    revision: int
    kind: Literal["content", "retraction"]
    side: Literal["train", "test"] | None
    text: str | None
    updated_at: float


class IngestSources(BaseModel):
    A: SourceVersion | None = None
    B: SourceVersion | None = None


class IngestArticleStatus(BaseModel):
    """Both pipelines' latest versions of one article plus the verdict."""

    id: str
    status: IngestStatus
    in_corpus: bool
    conflicts: list[str]  # subset of ["text", "side"] when status == conflict
    sources: IngestSources
    approved_side: Literal["train", "test"] | None
    approved_text: str | None


class IngestArticleSummary(BaseModel):
    id: str
    status: IngestStatus
    in_corpus: bool
    revisions: dict[str, int]  # per-source latest revision, e.g. {"A": 3, "B": 2}
