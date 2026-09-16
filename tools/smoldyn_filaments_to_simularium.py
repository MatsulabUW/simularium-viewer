"""Convert Smoldyn `printFilaments` output into a .simularium trajectory.

simulariumio's built-in SmoldynConverter reads `listmols` output only (point
molecules -> viz_type 1000). Filaments are polylines, so they need viz_type 1001
agents whose vertices live in the flat `subpoints` array. This script does that.

Input format (one line per filament per snapshot, from `cmd ... printFilaments`):

    FIL <time> <type>:<name> <nseg> <parent-or-"-"> <capped> x0 y0 [z0] x1 y1 [z1] ...

Dimensionality is inferred from the coordinate count: a filament with nseg
segments has nseg+1 nodes, so dim = ncoords / (nseg + 1). 2D runs are embedded
in the z=0 plane.

Note on scaling: TrajectoryConverter does NOT centre or scale on its own -- each
simulator's converter calls the helpers itself. Skipping them leaves sub-micron
data at world-scale ~0.5 while the viewer camera sits at z=120, so nothing is
visible. We therefore call center_fiber_positions() (moves each fiber's origin to
its centroid, subpoints relative) and center_and_scale_agent_data() (centres the
scene and scales it into the viewer's 4-64 unit range), and scale box_size to match.

Usage:
    python smoldyn_filaments_to_simularium.py FRAMES.txt OUT_PREFIX [options]
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, Counter

import numpy as np
from simulariumio import (
    TrajectoryConverter,
    TrajectoryData,
    AgentData,
    MetaData,
    DisplayData,
    UnitData,
    DISPLAY_TYPE,
)

# Sequential ramp: seed filaments dark, later branch generations lighter.
GENERATION_COLORS = ["#8c2d04", "#e6550d", "#fd8d3c", "#fdd0a2"]
CAPPED_COLORS = {"growing": "#e6550d", "capped": "#4a6fa5"}

# simulariumio's VIEWER_DIMENSION_RANGE is 4..64; aim just under the top.
VIEWER_TARGET_SPAN = 60.0


def parse_filament_file(path):
    """-> (frames, dim). frames is an ordered {time: [record, ...]}."""
    frames = OrderedDict()
    dim = None
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line.startswith("FIL"):
                continue
            f = line.split()
            time = float(f[1])
            # f[2] is "<filamenttype>:<filname>"; the parent field (f[4]) is the
            # bare <filname>, so key on filname for parent lookups to resolve.
            ftype, _, name = f[2].partition(":")
            nseg = int(f[3])
            parent = f[4]
            capped = bool(int(f[5]))
            coords = [float(v) for v in f[6:]]
            nnodes = nseg + 1
            if len(coords) % nnodes:
                raise ValueError(
                    f"{path}:{lineno}: {len(coords)} coords not divisible by "
                    f"{nnodes} nodes"
                )
            line_dim = len(coords) // nnodes
            if dim is None:
                dim = line_dim
            elif line_dim != dim:
                raise ValueError(f"{path}:{lineno}: dim {line_dim} != {dim} earlier")
            nodes = np.asarray(coords, dtype=float).reshape(nnodes, dim)
            if dim == 2:  # embed the 2D run in the z=0 plane
                nodes = np.column_stack([nodes, np.zeros(nnodes)])
            frames.setdefault(time, []).append(
                {"name": name, "ftype": ftype, "parent": parent,
                 "capped": capped, "nodes": nodes}
            )
    if dim is None:
        raise ValueError(f"{path}: no FIL records found")
    return frames, dim


def generation_of(name, parents, cache):
    """Walk the parent chain back to a seed. 0 = seed filament."""
    if name in cache:
        return cache[name]
    gen, cur, seen = 0, name, set()
    while True:
        parent = parents.get(cur, "-")
        if parent == "-" or parent not in parents or parent in seen:
            break
        seen.add(cur)
        cur = parent
        gen += 1
    cache[name] = gen
    return gen


PLANE_COLORS = ["#58606c", "#94a7fc", "#bbbb99", "#418463"]


def parse_plane(spec):
    """'NAME:x,y,z,dx,dy[:z1,t0,t1]' -> dict. Geometry order matches Smoldyn's
    `panel rect <normal> x y z dx dy`, so a panel line can be transcribed
    directly. The optional tail describes a linear ramp in z, held flat outside
    [t0, t1] -- the same shape as a piston driven by a `set surface ... panel`
    command."""
    name, _, rest = spec.partition(":")
    if not name or not rest:
        raise SystemExit(f"--plane needs NAME:x,y,z,dx,dy — got {spec!r}")
    geom, _, ramp = rest.partition(":")
    try:
        x, y, z, dx, dy = (float(v) for v in geom.split(","))
    except ValueError:
        raise SystemExit(f"--plane geometry must be 5 numbers x,y,z,dx,dy — got {geom!r}")
    out = {"name": name, "x": x, "y": y, "z": z, "dx": dx, "dy": dy,
           "z1": None, "t0": None, "t1": None}
    if ramp:
        try:
            z1, t0, t1 = (float(v) for v in ramp.split(","))
        except ValueError:
            raise SystemExit(f"--plane ramp must be z1,t0,t1 — got {ramp!r}")
        if t1 <= t0:
            raise SystemExit(f"--plane ramp needs t1 > t0 — got {ramp!r}")
        out.update(z1=z1, t0=t0, t1=t1)
    return out


def plane_nodes(pl, t):
    """Rectangle outline (5 points, closed) at this plane's height at time t."""
    z = pl["z"]
    if pl["z1"] is not None:
        frac = min(max((t - pl["t0"]) / (pl["t1"] - pl["t0"]), 0.0), 1.0)
        z = pl["z"] + frac * (pl["z1"] - pl["z"])
    x0, y0, x1, y1 = pl["x"], pl["y"], pl["x"] + pl["dx"], pl["y"] + pl["dy"]
    return np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z],
                     [x0, y1, z], [x0, y0, z]], dtype=float)


def inject_planes(frames, planes, args):
    """Add one fiber agent per plane per frame, before centring and scaling so
    the panels travel through the same transform as the filaments."""
    for i, pl in enumerate(planes):
        args._static_types.setdefault(
            pl["name"], PLANE_COLORS[i % len(PLANE_COLORS)])
    for t, recs in frames.items():
        for pl in planes:
            recs.append({"ftype": pl["name"], "name": f"__plane_{pl['name']}",
                         "parent": "-", "capped": False,
                         "nodes": plane_nodes(pl, t)})
    return frames


def type_name(rec, parents, args, cache):
    # Static scenery (e.g. a membrane drawn as grid fibers) keeps its own
    # filament-type name and color, outside the generation/capped schemes.
    if rec["ftype"] in args._static_types:
        return rec["ftype"]
    if args.color_by == "capped":
        return "capped" if rec["capped"] else "growing"
    gen = generation_of(rec["name"], parents, cache)
    top = len(GENERATION_COLORS) - 1
    return f"gen{gen}" if gen < top else f"gen{top}+"


def build(frames, dim, args):
    times = np.array(sorted(frames.keys()), dtype=float)
    n_steps = len(times)

    parents, uid_of = {}, {}
    for t in times:
        for rec in frames[t]:
            parents.setdefault(rec["name"], rec["parent"])
            uid_of.setdefault(rec["name"], len(uid_of))

    max_agents = max(len(frames[t]) for t in times)
    max_nodes = max(len(r["nodes"]) for t in times for r in frames[t])

    n_agents = np.zeros(n_steps, dtype=int)
    viz_types = np.zeros((n_steps, max_agents))
    unique_ids = np.zeros((n_steps, max_agents))
    positions = np.zeros((n_steps, max_agents, 3))
    radii = np.full((n_steps, max_agents), args.radius)
    n_subpoints = np.zeros((n_steps, max_agents))
    subpoints = np.zeros((n_steps, max_agents, 3 * max_nodes))
    types, gen_cache, seen_types = [], {}, OrderedDict()

    for ti, t in enumerate(times):
        recs = frames[t]
        n_agents[ti] = len(recs)
        step_types = []
        for ai, rec in enumerate(recs):
            nodes = rec["nodes"]
            viz_types[ti][ai] = 1001.0  # fiber
            unique_ids[ti][ai] = uid_of[rec["name"]]
            n_subpoints[ti][ai] = 3 * len(nodes)
            subpoints[ti][ai][: 3 * len(nodes)] = nodes.flatten()
            tname = type_name(rec, parents, args, gen_cache)
            # Per-type radius, in raw units so center_and_scale_agent_data still
            # scales it. Needed whenever one agent class is much shorter than the
            # others: a crosslink spanning ~1 viewer unit is invisible at the
            # radius that suits a 45-unit fiber, and reads well drawn thicker.
            if tname in args._type_radius:
                radii[ti][ai] = args._type_radius[tname]
            seen_types[tname] = None
            step_types.append(tname)
        # pad so indexing can't run off the end (writers slice by n_agents)
        step_types += [step_types[-1] if step_types else "gen0"] * (
            max_agents - len(step_types))
        types.append(step_types)

    display_data = {}
    for tname in seen_types:
        if tname in args._static_types:
            color = args._static_types[tname]
        elif args.color_by == "capped":
            color = CAPPED_COLORS.get(tname, "#888888")
        else:
            idx = min(int("".join(c for c in tname if c.isdigit()) or 0),
                      len(GENERATION_COLORS) - 1)
            color = GENERATION_COLORS[idx]
        display_data[tname] = DisplayData(
            name=tname, display_type=DISPLAY_TYPE.FIBER, color=color)

    agent_data = AgentData(
        times=times, n_agents=n_agents, viz_types=viz_types,
        unique_ids=unique_ids, types=types, positions=positions, radii=radii,
        n_subpoints=n_subpoints, subpoints=subpoints, display_data=display_data)

    raw_pts = np.vstack([r["nodes"] for t in times for r in frames[t]])
    raw_extent = raw_pts.max(axis=0) - raw_pts.min(axis=0)

    # Give each fiber a real origin (its centroid) with subpoints relative to it,
    # then centre the scene and scale it into the viewer's usable range.
    agent_data = TrajectoryConverter.center_fiber_positions(agent_data)
    # The library's auto scale only guarantees the scene lands somewhere in the
    # viewer's 4-64 unit range, and for sub-micron data it stops at 4 -- which is
    # nearly invisible against the viewer's default camera at z=120. Fill the
    # range instead so the network actually reads on screen.
    sf = args.scale_factor
    if sf is None:
        sf = VIEWER_TARGET_SPAN / max(float(raw_extent.max()), 1e-12)
    agent_data, scale_factor = TrajectoryConverter.center_and_scale_agent_data(
        agent_data, input_scale_factor=sf)

    if args.box_fit:
        # A cubic box around a flat slab wastes most of the viewport and makes the
        # data read as small; follow the per-axis extent instead.
        box = np.maximum(raw_extent, raw_extent.max() * 0.02) * 1.1 * scale_factor
    elif args.box:
        box = np.array([args.box] * 3, float) * scale_factor
    else:
        span = float(max(raw_extent.max(), 1e-12)) * 1.1 * scale_factor
        box = np.array([span, span, span if dim == 3 else span * 0.05])

    args._scale_factor = scale_factor
    args._raw_extent = raw_extent

    return TrajectoryData(
        meta_data=MetaData(box_size=box, scale_factor=scale_factor,
                           trajectory_title=args.title),
        agent_data=agent_data,
        time_units=UnitData(args.time_units),
        spatial_units=UnitData(args.spatial_units))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="printFilaments output file")
    p.add_argument("output", help="output prefix (.simularium is appended)")
    p.add_argument("--title", default="Smoldyn filaments")
    p.add_argument("--radius", type=float, default=None,
                   help="fiber radius in spatial units "
                        "(default: 1/150 of the data extent)")
    p.add_argument("--box", type=float, default=None,
                   help="cubic box edge in spatial units; "
                        "default is the data extent + 10%%")
    p.add_argument("--box-fit", action="store_true",
                   help="size the box to the per-axis data extent instead of a cube")
    p.add_argument("--scale-factor", type=float, default=None,
                   help="override the auto viewer scale factor")
    p.add_argument("--color-by", choices=["generation", "capped"],
                   default="generation")
    p.add_argument("--plane", action="append", default=[], metavar="SPEC",
                   help="draw a Smoldyn rect panel as a wireframe outline. "
                        "SPEC is NAME:x,y,z,dx,dy for a static panel (same "
                        "argument order as Smoldyn's `panel rect`), with an "
                        "optional :z1,t0,t1 suffix for a panel that moves "
                        "linearly from z to z1 over t0..t1 and is held outside "
                        "that window. Repeatable. Example: a floor at z=-0.1 "
                        "and a piston descending 0.15 -> -0.03 over t 2..3 is "
                        "--plane 'floor:-0.5,-0.5,-0.1,1,1' "
                        "--plane 'piston:-0.5,-0.5,0.15,1,1:-0.03,2,3'")
    p.add_argument("--stride", type=int, default=1, metavar="N",
                   help="keep every Nth frame (default 1 = all). Conversion cost "
                        "scales with frames x filaments, so a long branching run "
                        "usually needs this.")
    p.add_argument("--max-time", type=float, default=None, metavar="T",
                   help="drop frames after time T. Useful when filament count "
                        "runs away late in a branching simulation.")
    p.add_argument("--time-units", default="s")
    p.add_argument("--spatial-units", default="um")
    p.add_argument("--type-radius", action="append", default=[],
                   metavar="TYPE=RADIUS",
                   help="override the fiber radius for one agent type, in spatial "
                        "units (repeatable), e.g. crosslink=0.02")
    p.add_argument("--static-type", action="append", default=[],
                   metavar="TYPE=#RRGGBB",
                   help="render records of this filament type as static scenery "
                        "with a fixed color, outside the generation/capped "
                        "schemes (repeatable), e.g. membrane=#9aa0a8")
    args = p.parse_args()
    args._static_types = dict(s.split("=", 1) for s in args.static_type)
    args._type_radius = {k: float(v) for k, v in
                         (s.split("=", 1) for s in args.type_radius)}

    frames, dim = parse_filament_file(args.input)

    # Subsample before anything else touches the data. Conversion cost scales with
    # the total agent-frame count (frames x filaments per frame), and a branching
    # run's filament count grows exponentially, so the last few frames dominate.
    if args.max_time is not None:
        frames = OrderedDict((t, r) for t, r in frames.items() if t <= args.max_time)
    if args.stride > 1:
        frames = OrderedDict((t, r) for i, (t, r) in enumerate(frames.items())
                             if i % args.stride == 0)
    if not frames:
        raise SystemExit("no frames left after --stride/--max-time filtering")
    if args.plane:
        planes = [parse_plane(s) for s in args.plane]
        frames = inject_planes(frames, planes, args)
        print(f"drew {len(planes)} plane(s): "
              + ", ".join(p["name"] + (" (moving)" if p["z1"] is not None else "")
                          for p in planes))

    agent_frames = sum(len(r) for r in frames.values())
    print(f"{len(frames)} frames, {agent_frames} agent-frames "
          f"(t = {min(frames):g}..{max(frames):g})")
    if agent_frames > 50000:
        print(f"  warning: {agent_frames} agent-frames is large; conversion may "
              f"take many minutes. Consider --stride or --max-time.")

    if args.radius is None:  # something visible relative to the data
        pts = np.vstack([r["nodes"] for t in frames for r in frames[t]])
        args.radius = max(float((pts.max(axis=0) - pts.min(axis=0)).max()) / 150.0,
                          1e-12)

    data = build(frames, dim, args)
    TrajectoryConverter(data).save(args.output)

    last = frames[max(frames)]
    names = {r["name"] for t in frames for r in frames[t]}
    n_seed = len({r["name"] for t in frames for r in frames[t]
                  if r["parent"] == "-"})
    ad = data.agent_data
    tally = Counter(ad.types[-1][: int(ad.n_agents[-1])])
    print(f"{args.input}: {dim}D, {len(frames)} frames, "
          f"{len(names)} filaments ({n_seed} seed, {len(names) - n_seed} branched), "
          f"{max(len(frames[t]) for t in frames)} max simultaneous")
    print(f"  final frame by type : {dict(sorted(tally.items()))}")
    print(f"  capped at end       : {sum(1 for r in last if r['capped'])}/{len(last)}")
    print(f"  data extent         : {np.round(args._raw_extent, 4)} {args.spatial_units}")
    print(f"  viewer scale factor : {args._scale_factor:.4g} "
          f"(scene ~{float(args._raw_extent.max()) * args._scale_factor:.1f} units)")
    print(f"wrote {args.output}.simularium")


if __name__ == "__main__":
    main()
