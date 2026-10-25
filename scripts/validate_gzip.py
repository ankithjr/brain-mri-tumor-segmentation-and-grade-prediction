"""Check gzip stream integrity; does not validate NIfTI geometry or anatomy."""
import argparse
import gzip
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=2505)
    args = parser.parse_args()
    files = sorted(args.data_dir.rglob("*.nii.gz"))
    failures = []
    for i, path in enumerate(files, 1):
        try:
            size = 0
            with gzip.open(path, "rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
            if size < 352:
                raise ValueError("Decompressed file is too short for NIfTI")
        except Exception as error:
            failures.append({"file": str(path.relative_to(args.data_dir)), "error": str(error)})
            print(f"FAIL: {path.name}: {error}", flush=True)
        if i % 100 == 0 or i == len(files):
            print(f"Checked {i}/{len(files)}", flush=True)
    report = {"checked": len(files), "passed": len(files) - len(failures),
              "failed": len(failures), "failures": failures}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"RESULT: {report['passed']}/{len(files)} passed; {report['failed']} failed")
    return int(bool(failures) or len(files) != args.expected)

if __name__ == "__main__":
    raise SystemExit(main())
