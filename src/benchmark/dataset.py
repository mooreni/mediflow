from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetDoc:
    doc_id: str
    hebrew_text: str
    reference_russian: str


TRAIN_IDS = [f"Form_{i:03d}" for i in range(1, 39)]   # 001–038
TEST_IDS  = [f"Form_{i:03d}" for i in range(39, 51)]  # 039–050


def load_doc(doc_id: str, he_dir: Path, ru_dir: Path) -> DatasetDoc:
    he_path = he_dir / f"{doc_id}_HE.txt"
    ru_path = ru_dir / f"{doc_id}_RU.txt"
    hebrew_text = he_path.read_text(encoding="utf-8")
    reference_russian = ru_path.read_text(encoding="utf-8")
    return DatasetDoc(doc_id=doc_id, hebrew_text=hebrew_text, reference_russian=reference_russian)


def load_split(doc_ids: list[str], he_dir: Path, ru_dir: Path) -> list[DatasetDoc]:
    return [load_doc(doc_id, he_dir, ru_dir) for doc_id in doc_ids]
