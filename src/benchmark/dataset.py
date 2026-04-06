"""Dataset loader for MediFlow benchmark documents.

Provides functions for loading Hebrew source documents from the structured
data directory into DatasetDoc instances, split by eval/train role and
optionally filtered by document type.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetDoc:
    """A single benchmark document loaded from disk.

    Attributes:
        doc_id: Unique identifier, e.g. "Form_001" or "Prescript_035".
        doc_type: Document category; one of "form", "summary", "prescript", "referral".
        hebrew_text: Full UTF-8 text content of the source document.
    """

    doc_id: str       # e.g. "Form_001", "Prescript_035"
    doc_type: str     # "form" | "summary" | "prescript" | "referral"
    hebrew_text: str


_DOC_CONFIGS = [
    {
        "doc_type": "form",
        "subdir": Path("informed_consent_forms") / "text" / "he",
        "prefix": "Form",
        "eval_nums": range(1, 11),    # 001–010
        "train_nums": range(11, 19),  # 011–018
    },
    {
        "doc_type": "summary",
        "subdir": Path("summaries"),
        "prefix": "Summary",
        "eval_nums": range(1, 11),
        "train_nums": range(11, 19),
    },
    {
        "doc_type": "prescript",
        "subdir": Path("prescripts"),
        "prefix": "Prescript",
        "eval_nums": range(35, 45),   # 035–044
        "train_nums": range(45, 53),  # 045–052
    },
    {
        "doc_type": "referral",
        "subdir": Path("referrals"),
        "prefix": "Referral",
        "eval_nums": range(68, 78),   # 068–077
        "train_nums": range(78, 86),  # 078–085
    },
]


def _read_doc(path: Path, doc_id: str, doc_type: str) -> DatasetDoc:
    """Read a single document from disk and return a DatasetDoc.

    Args:
        path: Absolute path to the UTF-8 text file.
        doc_id: Identifier string to assign (e.g. "Form_001").
        doc_type: Document category string (e.g. "form").

    Returns:
        DatasetDoc populated with the file's text content.

    Raises:
        FileNotFoundError: If the file at path does not exist.
        OSError: If the file cannot be read.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Document '{doc_id}' not found. "
            f"Expected file at: {path}"
        )
    return DatasetDoc(
        doc_id=doc_id,
        doc_type=doc_type,
        hebrew_text=path.read_text(encoding="utf-8"),
    )


def _load_docs(data_dir: Path, nums_key: str) -> list[DatasetDoc]:
    """Load documents for all configured doc types using the given number range key.

    Args:
        data_dir: Root data directory containing all document subdirectories.
        nums_key: Key into each _DOC_CONFIGS entry selecting the range to load;
                  either "eval_nums" or "train_nums".

    Returns:
        List of DatasetDoc for every configured doc type in the requested range.
    """
    docs = []
    for cfg in _DOC_CONFIGS:
        for num in cfg[nums_key]:
            doc_id = f"{cfg['prefix']}_{num:03d}"
            path = data_dir / cfg["subdir"] / f"{doc_id}_HE.txt"
            docs.append(_read_doc(path, doc_id, cfg["doc_type"]))
    return docs


def load_eval_docs(data_dir: Path) -> list[DatasetDoc]:
    """Load all evaluation documents across every doc type.

    Args:
        data_dir: Root data directory containing all document subdirectories.

    Returns:
        List of DatasetDoc for every configured doc type's eval range.
    """
    return _load_docs(data_dir, "eval_nums")


def load_train_docs(data_dir: Path) -> list[DatasetDoc]:
    """Load all training documents across every doc type.

    Args:
        data_dir: Root data directory containing all document subdirectories.

    Returns:
        List of DatasetDoc for every configured doc type's train range.
    """
    return _load_docs(data_dir, "train_nums")


def load_eval_docs_by_type(data_dir: Path, doc_type: str) -> list[DatasetDoc]:
    """Load evaluation documents filtered to a single doc type.

    Args:
        data_dir: Root data directory containing all document subdirectories.
        doc_type: One of "form", "summary", "prescript", "referral".

    Returns:
        List of DatasetDoc for the requested type's eval range.

    Raises:
        ValueError: If doc_type does not match any configured doc type.
    """
    matching = [cfg for cfg in _DOC_CONFIGS if cfg["doc_type"] == doc_type]
    if not matching:
        valid = [cfg["doc_type"] for cfg in _DOC_CONFIGS]
        raise ValueError(
            f"Unknown doc_type '{doc_type}'. Expected one of: {valid}"
        )
    cfg = matching[0]
    docs = []
    for num in cfg["eval_nums"]:
        doc_id = f"{cfg['prefix']}_{num:03d}"
        path = data_dir / cfg["subdir"] / f"{doc_id}_HE.txt"
        docs.append(_read_doc(path, doc_id, cfg["doc_type"]))
    return docs


def load_summary_train_docs(data_dir: Path) -> list[DatasetDoc]:
    """Load Summary training documents for S7: Summary_011 through Summary_030.

    This range is intentionally larger than the global summary train range in
    _DOC_CONFIGS (011–018) and is used exclusively by the S7 bootstrap scenario.

    Args:
        data_dir: Root data directory containing the summaries subdirectory.

    Returns:
        List of 20 DatasetDoc entries for Summary_011–Summary_030.
    """
    summary_cfg = next(cfg for cfg in _DOC_CONFIGS if cfg["doc_type"] == "summary")
    docs = []
    for num in range(11, 31):  # Summary_011–Summary_030 (20 docs)
        doc_id = f"Summary_{num:03d}"
        path = data_dir / summary_cfg["subdir"] / f"{doc_id}_HE.txt"
        docs.append(_read_doc(path, doc_id, "summary"))
    return docs
