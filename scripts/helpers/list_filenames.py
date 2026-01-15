import argparse
from pathlib import Path

def iter_paths(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return [p for p in root.rglob("*") if p.is_file()]
    return [p for p in root.iterdir() if p.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Write filenames in a directory to a text file")
    parser.add_argument("--dir", required=True, help="Directory to list files from")
    parser.add_argument("--out", required=True, help="Output text file path")
    parser.add_argument("--recursive", action="store_true", help="Recursively include files in subdirectories")
    parser.add_argument("--absolute", action="store_true", help="Write absolute paths instead of relative-to --dir")
    args = parser.parse_args()

    root = Path(args.dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"--dir does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"--dir is not a directory: {root}")

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = iter_paths(root, recursive=bool(args.recursive))

    lines = [str(p.relative_to(root)) for p in paths]

    lines.sort()

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
