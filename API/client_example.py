import argparse
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--model-id", required=True, help="One of the bundle filenames from GET /models")
    ap.add_argument("--image", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    img_path = Path(args.image)
    with img_path.open("rb") as f:
        files = {"file": (img_path.name, f, "application/octet-stream")}
        r = requests.post(
            f"{args.api}/predict",
            params={"model_id": args.model_id, "top_k": args.top_k},
            files=files,
            timeout=120,
        )
    r.raise_for_status()
    print(r.json())


if __name__ == "__main__":
    main()

