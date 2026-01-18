import json
import os
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

    def discover_existing(self):
        """Discover existing artifacts in the root directory and add them to the library if not already present."""
        # This is a bit complex because existing artifacts are in different structures.
        # We'll look for meta.json files.
        for meta_path in self.root_dir.glob("**/meta.json"):
            if "library" in meta_path.parts:
                continue
                
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                
                source_pdf = meta.get("source_pdf")
                if not source_pdf:
                    continue
                
                original_name = Path(source_pdf).name
                # Check if already in library by comparing source_pdf or folder name
                already_exists = False
                for item in self.items.values():
                    if item["original_name"] == original_name:
                        already_exists = True
                        break
                
                if not already_exists:
                    # Create a new entry and copy artifacts
                    file_id = f"m_{int(time.time() * 1000)}_{meta['id'][:8]}"
                    file_dir = self.library_dir / file_id
                    file_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy everything from meta_path.parent to file_dir
                    src_dir = meta_path.parent
                    for item in src_dir.iterdir():
                        if item.is_file():
                            shutil.copy2(item, file_dir)
                        elif item.is_dir():
                            shutil.copytree(item, file_dir / item.name)
                    
                    # Try to find the source PDF and copy it as source.pdf
                    # It might be in the artifacts dir or at the absolute path stored in meta
                    found_pdf = False
                    pdf_path = Path(source_pdf)
                    if pdf_path.exists():
                        shutil.copy2(pdf_path, file_dir / "source.pdf")
                        found_pdf = True
                    else:
                        # Try to find it in the artifacts root (common for _uploads)
                        alt_pdf = self.root_dir / "_uploads" / pdf_path.name
                        if alt_pdf.exists():
                            shutil.copy2(alt_pdf, file_dir / "source.pdf")
                            found_pdf = True
                    
                    if not found_pdf:
                        # Create a dummy or just skip if we really need the PDF for rerun
                        print(f"Warning: Could not find source PDF for {meta_path}")

                    # Set status based on what we found
                    status = "not_processed"
                    if (file_dir / "meta.json").exists() and (file_dir / "page.png").exists():
                        # We have at least Step 1 artifacts. Only mark "done" if doors exist.
                        # IMPORTANT: "processing" is used by the UI to mean "job currently running",
                        # and the app does not run jobs in the background; using it here can
                        # permanently disable the Run button.
                        status = "done" if (file_dir / "doors.json").exists() else "not_processed"

                    self.items[file_id] = {
                        "id": file_id,
                        "original_name": original_name,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "status": status,
                        "path": str(file_dir),
                        "error": None
                    }
            except Exception as e:
                print(f"Error discovering {meta_path}: {e}")
        
        self._save_index()
