"""File operations for the coding agent"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import pathspec


class FileInfo(BaseModel):
    path: str
    size: int
    modified: float
    is_dir: bool
    is_file: bool


class FileOperations:
    """Handle file system operations with safety checks"""
    
    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.gitignore_spec = self._load_gitignore()
    
    def _load_gitignore(self) -> Optional[pathspec.PathSpec]:
        """Load .gitignore patterns for filtering"""
        gitignore_path = self.workspace_root / ".gitignore"
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    return pathspec.PathSpec.from_lines('gitwildmatch', f)
            except Exception:
                pass
        return None
    
    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is within workspace and safe to access"""
        try:
            resolved_path = path.resolve()
            workspace_resolved = self.workspace_root.resolve()
            return str(resolved_path).startswith(str(workspace_resolved))
        except Exception:
            return False
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored based on .gitignore"""
        if self.gitignore_spec:
            relative_path = path.relative_to(self.workspace_root)
            return self.gitignore_spec.match_file(str(relative_path))
        return False
    
    def read_file(self, file_path: str, encoding: str = 'utf-8') -> str:
        """Read file contents safely"""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        
        if not self._is_safe_path(path):
            raise ValueError(f"Path {path} is outside workspace")
        
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")
        
        if not path.is_file():
            raise ValueError(f"Path {path} is not a file")
        
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encodings
            for enc in ['latin-1', 'cp1252', 'utf-16']:
                try:
                    with open(path, 'r', encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file {path}")
    
    def write_file(self, file_path: str, content: str, encoding: str = 'utf-8') -> bool:
        """Write content to file safely"""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        
        if not self._is_safe_path(path):
            raise ValueError(f"Path {path} is outside workspace")
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to write file {path}: {e}")
    
    def list_directory(self, dir_path: str = ".", include_hidden: bool = False) -> List[FileInfo]:
        """List directory contents"""
        path = Path(dir_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        
        if not self._is_safe_path(path):
            raise ValueError(f"Path {path} is outside workspace")
        
        if not path.exists():
            raise FileNotFoundError(f"Directory {path} not found")
        
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")
        
        files = []
        try:
            for item in path.iterdir():
                # Skip hidden files unless requested
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                # Skip ignored files
                if self._should_ignore(item):
                    continue
                
                stat = item.stat()
                files.append(FileInfo(
                    path=str(item.relative_to(self.workspace_root)),
                    size=stat.st_size,
                    modified=stat.st_mtime,
                    is_dir=item.is_dir(),
                    is_file=item.is_file()
                ))
        except PermissionError:
            raise RuntimeError(f"Permission denied accessing {path}")
        
        return sorted(files, key=lambda x: (not x.is_dir, x.path.lower()))
    
    def create_directory(self, dir_path: str) -> bool:
        """Create directory safely"""
        path = Path(dir_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        
        if not self._is_safe_path(path):
            raise ValueError(f"Path {path} is outside workspace")
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to create directory {path}: {e}")
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file safely"""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        
        if not self._is_safe_path(path):
            raise ValueError(f"Path {path} is outside workspace")
        
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")
        
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete {path}: {e}")
    
    def copy_file(self, src_path: str, dst_path: str) -> bool:
        """Copy file safely"""
        src = Path(src_path)
        dst = Path(dst_path)
        
        if not src.is_absolute():
            src = self.workspace_root / src
        if not dst.is_absolute():
            dst = self.workspace_root / dst
        
        if not self._is_safe_path(src) or not self._is_safe_path(dst):
            raise ValueError("Paths must be within workspace")
        
        if not src.exists():
            raise FileNotFoundError(f"Source file {src} not found")
        
        try:
            # Create destination directory if needed
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            if src.is_file():
                shutil.copy2(src, dst)
            elif src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to copy {src} to {dst}: {e}")
    
    def move_file(self, src_path: str, dst_path: str) -> bool:
        """Move file safely"""
        src = Path(src_path)
        dst = Path(dst_path)
        
        if not src.is_absolute():
            src = self.workspace_root / src
        if not dst.is_absolute():
            dst = self.workspace_root / dst
        
        if not self._is_safe_path(src) or not self._is_safe_path(dst):
            raise ValueError("Paths must be within workspace")
        
        if not src.exists():
            raise FileNotFoundError(f"Source file {src} not found")
        
        try:
            # Create destination directory if needed
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to move {src} to {dst}: {e}")
    
    def find_files(self, pattern: str, directory: str = ".", max_results: int = 100) -> List[str]:
        """Find files matching pattern"""
        path = Path(directory)
        if not path.is_absolute():
            path = self.workspace_root / path
        
        if not self._is_safe_path(path):
            raise ValueError(f"Path {path} is outside workspace")
        
        if not path.exists() or not path.is_dir():
            return []
        
        matches = []
        try:
            for file_path in path.rglob(pattern):
                if len(matches) >= max_results:
                    break
                
                if not self._is_safe_path(file_path):
                    continue
                
                if self._should_ignore(file_path):
                    continue
                
                if file_path.is_file():
                    matches.append(str(file_path.relative_to(self.workspace_root)))
        except Exception:
            pass
        
        return matches
    
    def get_file_info(self, file_path: str) -> FileInfo:
        """Get detailed file information"""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        
        if not self._is_safe_path(path):
            raise ValueError(f"Path {path} is outside workspace")
        
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")
        
        stat = path.stat()
        return FileInfo(
            path=str(path.relative_to(self.workspace_root)),
            size=stat.st_size,
            modified=stat.st_mtime,
            is_dir=path.is_dir(),
            is_file=path.is_file()
        )