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
        self.index_path = self.library_dir / "index.json"
        self._ensure_dirs()
        self.items = self._load_index()

    def _ensure_dirs(self):
        self.library_dir.mkdir(parents=True, exist_ok=True)

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
