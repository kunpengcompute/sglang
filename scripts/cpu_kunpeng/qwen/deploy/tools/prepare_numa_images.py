#!/usr/bin/env python3
"""Copy critical ELF files into independent per-NUMA rank images."""

import argparse
import hashlib
import json
import os
import socket
import re
import shutil
import subprocess
from pathlib import Path


def copy_library(source, destination):
    real = source.resolve(strict=True)
    target = destination / real.name
    if not target.exists():
        shutil.copy2(real, target)
    if source.name != real.name and not (destination / source.name).exists():
        (destination / source.name).symlink_to(real.name)


def native_paths(source):
    """List ELF paths once; pure Python/headers remain in the vendor environment."""
    paths = set()
    for directory, _, files in os.walk(source):
        for name in files:
            if name.endswith(".so") or ".so." in name:
                paths.add(Path(directory, name).relative_to(source))
    return paths


def copy_runtime_package(source, destination, native):
    destination.mkdir()
    for child in source.iterdir():
        relative = Path(child.name)
        selected = {p.relative_to(relative) for p in native if relative in p.parents}
        target = destination / child.name
        if relative in native:
            # Dereference source aliases so every rank owns the actual ELF inode.
            shutil.copy2(child, target)
        elif selected:
            copy_runtime_package(child, target, selected)
        elif child.name != "__pycache__":
            target.symlink_to(child.resolve(), target_is_directory=child.is_dir())


def runtime_dependencies(roots, env):
    """Resolve transitive DT_NEEDED paths using the target runtime loader."""
    found = {}
    for source in sorted({p.resolve(strict=True) for p in roots}):
        output = subprocess.check_output(
            ["ldd", str(source)], env=env, text=True, stderr=subprocess.STDOUT
        )
        if "=> not found" in output:
            raise RuntimeError(f"Unresolved runtime dependency of {source}:\n{output}")
        for line in output.splitlines():
            match = re.search(r"(?:=>\s+|^\s*)(/\S+)", line)
            if match:
                path = Path(match[1])
                # The ELF interpreter retains its platform path. Do not patch it.
                if "ld-linux" not in path.name:
                    found[str(path)] = path
    return list(found.values())


def copy_rank(rank, plan):
    image = Path(plan["output"]) / f"rank{rank}"
    (image / "bin").mkdir(parents=True)
    (image / "lib").mkdir()
    (image / "packages").mkdir()
    shutil.copy2(plan["python"], image / "bin/python")
    for name, package in plan["packages"].items():
        copy_runtime_package(
            Path(package["source"]),
            image / "packages" / name,
            {Path(p) for p in package["native"]},
        )
    for library in plan["libraries"]:
        copy_library(Path(library), image / "lib")
    numa_name = Path(plan["libnuma"]).resolve(strict=True).name
    for name in ("libnuma.so.1", "libnuma.so"):
        if not (image / "lib" / name).exists():
            (image / "lib" / name).symlink_to(numa_name)
    # Some filesystems defer populating the new file's page cache during copy.
    # Hash inside this already NUMA-bound worker, before an unbound parent can
    # touch the ELF contents. The launcher verifies receipts without reading ELF.
    common = list((image / "packages/sgl_kernel").glob("common_ops*.so"))
    if len(common) != 1:
        raise RuntimeError("Expected one common_ops DSO")
    (image / "common_ops_receipt.json").write_text(json.dumps({
        "rank": rank, "path": str(common[0]), "inode": common[0].stat().st_ino,
        "sha256": hashlib.sha256(common[0].read_bytes()).hexdigest(),
        "copy_hash_cpus": sorted(os.sched_getaffinity(0)),
    }, indent=2) + "\n")
    print(f"rank{rank}: {image}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--kutacc-prefix", type=Path, required=True)
    parser.add_argument("--kupl-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    probe = (
        "import sys,json,ctypes,importlib.util;"
        "ctypes.CDLL('libnuma.so.1');"
        "numa=next(line.split()[-1] for line in open('/proc/self/maps') "
        "if '/libnuma.so.' in line);"
        "print(json.dumps(dict(prefix=sys.prefix,libnuma=numa,packages={"
        "n:list(importlib.util.find_spec(n).submodule_search_locations)[0] "
        "for n in ['torch','sgl_kernel']})))"
    )
    info = json.loads(
        subprocess.check_output([str(args.python), "-c", probe], text=True)
    )
    prefix = Path(info["prefix"])
    libraries = []
    for library_dir, patterns in [
        (prefix / "lib", ("libpython*.so*",)),
        (args.kutacc_prefix / "lib", ("libkutacc.so*",)),
        (args.kupl_prefix / "lib", ("libkupl.so*",)),
    ]:
        for pattern in patterns:
            matches = list(library_dir.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No {pattern} in {library_dir}")
            libraries.extend(matches)
    libraries.append(Path(info["libnuma"]))
    packages = {
        name: (Path(source), native_paths(Path(source)))
        for name, source in info["packages"].items()
    }
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    env["LD_LIBRARY_PATH"] = ":".join(
        [
            str(args.kutacc_prefix / "lib"),
            str(args.kupl_prefix / "lib"),
            str(packages["torch"][0] / "lib"),
            str(prefix / "lib"),
            env.get("LD_LIBRARY_PATH", ""),
        ]
    )
    roots = [args.python, *libraries]
    for source, native in packages.values():
        roots.extend(source / relative for relative in native)
    libraries.extend(runtime_dependencies(roots, env))
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {
        "output": str(args.output.resolve()),
        "python": str(args.python.resolve()),
        "libnuma": info["libnuma"],
        "libraries": sorted({str(p) for p in libraries}),
        "packages": {
            name: {"source": str(source), "native": [str(p) for p in native]}
            for name, (source, native) in packages.items()
        },
    }
    # Copy writes can populate the page cache too. Bind the copy process so
    # newly created private ELF pages do not all become resident on one NUMA.
    worker = (
        "import json,runpy,sys;runpy.run_path(sys.argv[1])['copy_rank']"
        "(int(sys.argv[2]),json.load(sys.stdin))"
    )
    for rank in range(16):
        subprocess.run(
            [
                "taskset",
                "-c",
                str(rank * 38),
                "numactl",
                f"--physcpubind={rank * 38}",
                f"--membind={rank}",
                str(args.python),
                "-c",
                worker,
                str(Path(__file__).resolve()),
                str(rank),
            ],
            input=json.dumps(plan),
            text=True,
            check=True,
            env=env,
        )
    (args.output / "images.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "copy_bound_to_rank_numa": True,
                "runtime_lock_sha256": hashlib.sha256(args.runtime_lock.read_bytes()).hexdigest(),
                "hostname": socket.gethostname(),
                "hostname_is_provenance_only": True,
                "topology": {"ranks": 16, "cpus_per_ddr_numa": 38,
                             "ddr_nodes": list(range(16)), "hbm_nodes": list(range(16, 32))},
                "runtime_dependency_closure": sorted({str(p) for p in libraries}),
                "python_prefix": str(prefix),
                "images": [
                    str((args.output / f"rank{rank}").resolve()) for rank in range(16)
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
