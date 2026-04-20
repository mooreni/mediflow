"""Data loader for MediFlow production pipeline.

Loads Hebrew source documents from the structured data directory into
DatasetDoc instances. Optionally filtered by document type.

File naming convention: <Prefix>_<NNN>_HE.txt  (e.g. Form_001_HE.txt).
The '_HE' suffix identifies the Hebrew (source) file; other suffixes
(e.g. '_RU') would denote translated versions but are not loaded here.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetDoc:
    """A single document loaded from the dataset.

    Attributes:
        doc_id: Unique identifier, e.g. "Form_001" or "Prescript_035".
        doc_type: Document category; one of "form", "summary", "prescript",
            "referral".
        hebrew_text: Full UTF-8 text content of the source document.
    """

    doc_id: str       # e.g. "Form_001", "Prescript_035"
    doc_type: str     # "form" | "summary" | "prescript" | "referral"
    hebrew_text: str


# Document type configurations: each entry maps a type to its subdirectory,
# filename prefix, and the numeric range used in production evaluation.
_DOC_CONFIGS = [
    {
        "doc_type": "form",
        "subdir": Path("informed_consent_forms") / "text" / "he",
        "prefix": "Form",
        "nums": range(1, 31),      # Form_001–Form_030
    },
    {
        "doc_type": "summary",
        "subdir": Path("summaries"),
        "prefix": "Summary",
        "nums": range(1, 31),      # Summary_001–Summary_030
    },
    {
        "doc_type": "prescript",
        "subdir": Path("prescripts"),
        "prefix": "Prescript",
        "nums": range(35, 65),     # Prescript_035–Prescript_064
    },
    {
        "doc_type": "referral",
        "subdir": Path("referrals"),
        "prefix": "Referral",
        "nums": range(68, 98),     # Referral_068–Referral_097
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


def load_documents(
    data_dir: Path,
    doc_type: str | None = None,
) -> list[DatasetDoc]:
    """Load Hebrew source documents from disk.

    Args:
        data_dir: Root data directory containing all document subdirectories.
        doc_type: Optional filter; one of "form", "summary", "prescript",
            "referral". If None, all four types are loaded.

    Returns:
        List of DatasetDoc, one per document file in the configured range.

    Raises:
        ValueError: If doc_type is provided but does not match any known type.
        FileNotFoundError: If an expected document file is missing from disk.
    """
    if doc_type is not None:
        configs = [cfg for cfg in _DOC_CONFIGS if cfg["doc_type"] == doc_type]
        if not configs:
            valid = [cfg["doc_type"] for cfg in _DOC_CONFIGS]
            raise ValueError(
                f"Unknown doc_type '{doc_type}'. Expected one of: {valid}"
            )
    else:
        configs = _DOC_CONFIGS

    docs: list[DatasetDoc] = []
    for cfg in configs:
        for num in cfg["nums"]:
            doc_id = f"{cfg['prefix']}_{num:03d}"
            # Files follow the <DocId>_HE.txt naming convention
            path = data_dir / cfg["subdir"] / f"{doc_id}_HE.txt"
            docs.append(_read_doc(path, doc_id, cfg["doc_type"]))
    return docs
