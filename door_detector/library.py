import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

class Library:
    """Manages the library of uploaded and processed PDFs."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.library_dir = root_dir / "library"
        # Archived items are moved here (keeps labels.json for training).
        self.archive_dir = root_dir / "archive"
        self.index_path = self.library_dir / "index.json"
        self._ensure_dirs()
        self.items = self._load_index()

    def _ensure_dirs(self):
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> Dict[str, Any]:
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self.items, f, indent=2)

    def add_file(self, file_name: str, file_content: bytes) -> str:
        """Add a new file to the library and return its ID."""
        file_id = f"f_{int(time.time() * 1000)}"
        file_dir = self.library_dir / file_id
        file_dir.mkdir(parents=True, exist_ok=True)
        
        source_path = file_dir / "source.pdf"
        with open(source_path, "wb") as f:
            f.write(file_content)
            
        self.items[file_id] = {
            "id": file_id,
            "original_name": file_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "not_processed",
            "path": str(file_dir),
            "error": None
        }
        self._save_index()
        return file_id

    def delete_item(self, file_id: str):
        """Delete an item from the library and its artifacts."""
        if file_id in self.items:
            file_dir = Path(self.items[file_id]["path"])
            if file_dir.exists():
                shutil.rmtree(file_dir)
            del self.items[file_id]
            self._save_index()

    def archive_item(self, file_id: str) -> Optional[Path]:
        """Soft-delete: remove item from library list but preserve artifacts on disk.

        The item folder is moved from `root/library/<file_id>` to `root/archive/<file_id>`.
        Returns the archive path on success, else None.
        """
        if file_id not in self.items:
            return None

        file_dir = Path(self.items[file_id].get("path") or "")
        if not str(file_dir):
            # Nothing sensible to move; behave like a delete from the index.
            del self.items[file_id]
            self._save_index()
            return None

        # Pick a destination that won't clobber an existing archive entry.
        dest = self.archive_dir / str(file_id)
        if dest.exists():
            dest = self.archive_dir / f"{file_id}__archived_{int(time.time() * 1000)}"

        try:
            if file_dir.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_dir), str(dest))
        except Exception:
            # Best-effort: do not remove the index entry if the move failed.
            return None

        # Remove from the active library index so it disappears from the UI list.
        try:
            del self.items[file_id]
        except Exception:
            pass
        self._save_index()
        return dest

    def clear(self):
        """Remove all items from the library (deletes library/* contents)."""
        if self.library_dir.exists():
            shutil.rmtree(self.library_dir)
        self._ensure_dirs()
        self.items = {}
        self._save_index()

    def update_status(self, file_id: str, status: str, error: Optional[str] = None):
        """Update the processing status of a file."""
        if file_id in self.items:
            self.items[file_id]["status"] = status
            self.items[file_id]["error"] = error
            self._save_index()

    def get_items(self) -> List[Dict[str, Any]]:
        """Return all items in the library, sorted by creation date."""
        return sorted(self.items.values(), key=lambda x: x["created_at"], reverse=True)
