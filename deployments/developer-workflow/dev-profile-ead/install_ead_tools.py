"""
Install EAD tools into the existing vss_agents package inside the container.

This script copies the updated Python source files (new EAD tools + updated
register.py files that import them) directly into the vss_agents package
directory that is already installed in the container's virtualenv.

No pip install / build backend is needed because:
  - The NAT entry points already exist in the dist-info for vss_agents,
    pointing to vss_agents.tools.register and vss_agents.agents.register.
  - Our updated register.py files import all original tools PLUS the new
    EAD tools, so the entry points automatically pick them up.
  - We just need the .py files to exist at the right location.
"""
import pathlib
import shutil
import sys

src = pathlib.Path("/tmp/vss_agents_ead")
if not src.exists():
    print(f"ERROR: source directory not found: {src}", file=sys.stderr)
    sys.exit(1)

# Locate the installed vss_agents package
try:
    import vss_agents
    dst = pathlib.Path(vss_agents.__file__).parent
except ImportError as e:
    print(f"ERROR: cannot import vss_agents: {e}", file=sys.stderr)
    sys.exit(1)

print(f"Installing EAD tools: {src} -> {dst}")

copied = 0
for src_file in sorted(src.rglob("*.py")):
    rel = src_file.relative_to(src)
    dst_file = dst / rel
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    print(f"  {rel}")
    copied += 1

print(f"Done — {copied} files copied.")
