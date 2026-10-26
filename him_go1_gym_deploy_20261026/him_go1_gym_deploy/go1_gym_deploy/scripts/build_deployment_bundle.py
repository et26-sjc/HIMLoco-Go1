#!/usr/bin/env python3
"""Build or verify a deployment bundle without importing Torch or opening LCM."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEPLOY_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ignored(directory, names):
    ignored_names = {"__pycache__", ".git", "build", "logs", "models", "lcm_position"}
    return [name for name in names if name in ignored_names or
            name.endswith((".pyc", ".pt", ".jit", ".pkl", ".mp4", ".tar", ".tar.gz"))]


def build_bundle(config_path, model_path, output, deploy_root=DEPLOY_ROOT):
    from go1_gym_deploy.himloco.config import load_config

    config_path = Path(config_path).resolve()
    model_path = Path(model_path).resolve()
    output = Path(output).resolve()
    deploy_root = Path(deploy_root).resolve()
    config = load_config(config_path, require_ready=False)
    expected = config["model"]["sha256"]
    if expected is None:
        raise ValueError("model.sha256 must be recorded before packaging")
    actual = sha256_file(model_path)
    if actual != expected.lower():
        raise ValueError("explicit model file does not match the configuration SHA-256")
    if output == deploy_root or deploy_root in output.parents:
        raise ValueError("bundle output must be outside the deployment source directory")
    if output.exists():
        raise ValueError("bundle output already exists; select a new release directory")

    # Preserve all recorded metadata; only the model location changes.
    raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    original_config_sha = sha256_file(config_path)
    raw["model"]["path"] = "models/policy.pt"
    output.mkdir(parents=True)
    shutil.copytree(str(deploy_root), str(output / "go1_gym_deploy"), ignore=_ignored)
    (output / "models").mkdir()
    shutil.copy2(str(model_path), str(output / "models" / "policy.pt"))
    if sha256_file(output / "models" / "policy.pt") != actual:
        raise ValueError("model changed during packaging; incomplete output must not be deployed")
    (output / "himloco.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    # Resolve the portable path and validate the packaged configuration again.
    load_config(output / "himloco.json", require_ready=False)
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            files[path.relative_to(output).as_posix()] = sha256_file(path)
    manifest = {"bundle_schema_version": 1,
                "source_commit": config["source_commit"],
                "original_config_sha256": original_config_sha,
                "model_sha256": actual, "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify_bundle(output)
    return output


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("bundle_schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("unsupported deployment manifest")
    files = manifest["files"]
    if not {"himloco.json", "models/policy.pt"}.issubset(files):
        raise ValueError("manifest must include the portable configuration and model")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("manifest contains a path outside the bundle")
        if sha256_file(path) != expected:
            raise ValueError("bundle integrity failure: " + relative)
    config = json.loads((root / "himloco.json").read_text(encoding="utf-8"))
    if config["model"]["path"] != "models/policy.pt":
        raise ValueError("bundle configuration has a nonportable model path")
    actual = files["models/policy.pt"]
    if actual != manifest.get("model_sha256") or actual != config["model"]["sha256"].lower():
        raise ValueError("model hashes disagree in manifest/configuration")
    return len(files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, help="verify an existing bundle and exit")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path, help="explicit policy.pt to include; SHA-256 must match config")
    parser.add_argument("--output", type=Path, help="new directory, outside the deployment source")
    args = parser.parse_args(argv)
    if args.verify:
        if args.config or args.model or args.output:
            parser.error("--verify cannot be combined with build arguments")
    elif not all((args.config, args.model, args.output)):
        parser.error("building requires --config, --model and --output")
    try:
        if args.verify:
            count = verify_bundle(args.verify)
            print("Verified {} manifest files; no robot connection was opened.".format(count))
        else:
            output = build_bundle(args.config, args.model, args.output)
            print("Bundle: {}".format(output))
            print("Integrity checked; this does not validate the policy or authorize motor control.")
        return 0
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        print("Deployment bundle error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
