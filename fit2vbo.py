#!/usr/bin/env python3
"""
fit2vbo.py  --  Convert a Garmin FIT activity to a RaceLogic VBOX .vbo file
                for use with VBOX Circuit Tools.

WHY THIS EXISTS
---------------
A Garmin Connect IQ watch app (like HotLap) is sandboxed and can only emit a
FIT activity -- it cannot write a .vbo directly. So the workflow is:

    1. Record + save the session on the watch (HotLap does this).
    2. In Garmin Connect (web): open the activity -> gear/settings ->
       "Export Original" to download the .fit file.
    3. Run this script on your computer to produce a .vbo.
    4. Open the .vbo in Circuit Tools.

WHAT IT MAPS
------------
Native FIT fields  -> VBO channels:
    position_lat/long (semicircles) -> lat / long   (VBO minutes, West +)
    enhanced_speed (m/s)            -> velocity      (km/h)
    (derived) course over ground    -> heading       (deg)
    enhanced_altitude (m)           -> height         (m)
    heart_rate (bpm)                -> heart_rate

Custom developer fields (written by HotLap) -> VBO channels:
    g_lat  (g)      -> LatAcc   (lateral / cornering)
    g_long (g)      -> LongAcc  (longitudinal / brake-accel)
    spo2 (%)        -> SpO2 %          (if present)
    respiration     -> respiration bpm (if present)

If g_lat / g_long are absent (e.g. a plain Garmin FIT), the accelerations are
derived from GPS: longitudinal from the speed change, lateral from the rate of
heading change times speed. This keeps the friction-circle plot usable.

Lap- and session-scope custom fields don't fit VBO's per-sample channel model,
so they're written into the [comments] block (Circuit Tools keeps but ignores
comments). Captured there:
    session : track name; pit count / best / total; best & theoretical-best lap;
              session max G
    per lap : lap time, sector times (1-8), pit-stop time, and in/out-lap flag

USAGE
-----
    python3 fit2vbo.py activity.fit [out.vbo]

Requires:  pip install fitparse
"""

import sys
import math
import datetime as dt

try:
    import fitparse
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install fitparse")

SEMI_TO_DEG = 180.0 / (2 ** 31)   # semicircles -> degrees
G = 9.80665                        # m/s^2 per g


def semicircles_to_deg(v):
    return None if v is None else v * SEMI_TO_DEG


def read_records(path):
    """Return a list of dict rows sorted by time, one per FIT 'record'."""
    fit = fitparse.FitFile(path)
    rows = []
    for msg in fit.get_messages("record"):
        vals = {}
        for d in msg:
            vals[d.name] = d.value
        ts = vals.get("timestamp")
        lat = semicircles_to_deg(vals.get("position_lat"))
        lon = semicircles_to_deg(vals.get("position_long"))
        if ts is None or lat is None or lon is None:
            continue  # VBO needs time + position on every row
        rows.append({
            "ts": ts,
            "lat": lat,
            "lon": lon,
            "speed": vals.get("enhanced_speed", vals.get("speed")),   # m/s
            "alt": vals.get("enhanced_altitude", vals.get("altitude")),
            "hr": vals.get("heart_rate"),
            "temp": vals.get("temperature"),          # deg C (native, if any)
            # HotLap developer fields (may be absent):
            "g_lat": vals.get("g_lat"),
            "g_long": vals.get("g_long"),
            "spo2": vals.get("spo2"),                 # %
            "resp": vals.get("respiration"),          # breaths/min
        })
    rows.sort(key=lambda r: r["ts"])
    _assign_times(rows)
    return rows


def _assign_times(rows):
    """Assign a strictly-increasing time-of-day (seconds) to each row. FIT
    timestamps are 1-second resolution, so a high-rate (e.g. 10 Hz) recording
    has many rows per second; spread duplicates evenly so Circuit Tools sees a
    monotonic time base (duplicate times can crash its analysis)."""
    def sod(ts):
        return ts.hour * 3600 + ts.minute * 60 + ts.second \
            + ts.microsecond / 1e6
    n = len(rows)
    base = [sod(r["ts"]) for r in rows]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and base[j + 1] == base[i]:
            j += 1
        grp = j - i + 1
        if grp > 1:
            for k in range(grp):
                base[i + k] = base[i] + float(k) / grp
        i = j + 1
    for i in range(1, n):
        if base[i] <= base[i - 1]:
            base[i] = base[i - 1] + 0.01   # final monotonic safety
    for i in range(n):
        rows[i]["t"] = base[i]


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, in degrees."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    b = math.degrees(math.atan2(y, x))
    return (b + 360.0) % 360.0


def enrich(rows):
    """Fill heading and (if needed) derived lat/long acceleration in g."""
    for i, r in enumerate(rows):
        # dt to previous sample
        if i > 0:
            dts = (r["ts"] - rows[i - 1]["ts"]).total_seconds()
        else:
            dts = 0.0
        r["dt"] = dts

        # heading from consecutive positions
        if i + 1 < len(rows):
            r["heading"] = bearing_deg(r["lat"], r["lon"],
                                       rows[i + 1]["lat"], rows[i + 1]["lon"])
        elif i > 0:
            r["heading"] = rows[i - 1].get("heading", 0.0)
        else:
            r["heading"] = 0.0

    # accelerations
    for i, r in enumerate(rows):
        # longitudinal: use device field if present, else d(speed)/dt
        if r["g_long"] is not None:
            r["LongAcc"] = float(r["g_long"])
        elif i > 0 and r["dt"] > 0 and r["speed"] is not None \
                and rows[i - 1]["speed"] is not None:
            dv = r["speed"] - rows[i - 1]["speed"]
            r["LongAcc"] = _clamp((dv / r["dt"]) / G)
        else:
            r["LongAcc"] = 0.0

        # lateral: use device field if present, else v * yaw_rate
        if r["g_lat"] is not None:
            r["LatAcc"] = float(r["g_lat"])
        elif i > 0 and r["dt"] > 0 and r["speed"] is not None:
            dh = r["heading"] - rows[i - 1].get("heading", r["heading"])
            # shortest angular difference
            while dh > 180.0:
                dh -= 360.0
            while dh < -180.0:
                dh += 360.0
            yaw = math.radians(dh) / r["dt"]        # rad/s
            r["LatAcc"] = _clamp((r["speed"] * yaw) / G)
        else:
            r["LatAcc"] = 0.0

    # Cumulative ascent / descent (meters), using the same 1 m deadband as the
    # watch app so these channels match the activity's total_ascent/descent.
    ref = None
    asc = 0.0
    desc = 0.0
    for r in rows:
        a = r.get("alt")
        if a is not None:
            if ref is None:
                ref = a
            else:
                d = a - ref
                if d >= 1.0:
                    asc += d
                    ref = a
                elif d <= -1.0:
                    desc += -d
                    ref = a
        r["ascent"] = asc
        r["descent"] = desc
    return rows


def _clamp(g, lo=-3.0, hi=3.0):
    """Clamp a GPS-derived acceleration to a physically sane range."""
    return max(lo, min(hi, g))


def vbo_time(secs):
    """Format seconds-of-day as HHMMSS.SS (VBOX 'time' channel)."""
    hh = int(secs // 3600)
    mm = int((secs % 3600) // 60)
    ss = secs % 60
    return "%02d%02d%05.2f" % (hh, mm, ss)


def vbo_lat(deg):
    """Latitude in minutes, North positive (VBOX convention)."""
    return "%+011.5f" % (deg * 60.0)


def vbo_long(deg):
    """Longitude in minutes, WEST positive (VBOX convention)."""
    return "%+012.5f" % (-deg * 60.0)


def start_finish_gate(path, rows):
    """Derive a start/finish gate (two lat/lon endpoints, degrees) from the
    FIT lap markers -- their start positions cluster on the start/finish
    crossing. Returns ((latA,lonA),(latB,lonB)) or None if not derivable."""
    try:
        fit = fitparse.FitFile(path)
        pts = []
        for m in fit.get_messages("lap"):
            v = {d.name: d.value for d in m}
            la = semicircles_to_deg(v.get("start_position_lat"))
            lo = semicircles_to_deg(v.get("start_position_long"))
            if la is not None and lo is not None:
                pts.append((la, lo))
        # Drop the first lap's start (out-lap origin / pit) when we have several.
        if len(pts) >= 3:
            pts = pts[1:]
        if len(pts) < 1 or rows is None or len(rows) < 2:
            return None
        sf_lat = sum(p[0] for p in pts) / len(pts)
        sf_lon = sum(p[1] for p in pts) / len(pts)
    except Exception:
        return None

    # Heading at the S/F: nearest record vs one a couple of samples later.
    def d2(r):
        return (r["lat"] - sf_lat) ** 2 + \
               ((r["lon"] - sf_lon) * math.cos(math.radians(sf_lat))) ** 2
    ni = min(range(len(rows)), key=lambda i: d2(rows[i]))
    nj = min(ni + 2, len(rows) - 1)
    hdg = bearing_deg(rows[ni]["lat"], rows[ni]["lon"],
                      rows[nj]["lat"], rows[nj]["lon"])

    # ~30 m line perpendicular to travel, centered on the S/F point.
    half = 15.0
    perp = math.radians(hdg + 90.0)
    dlat = (half * math.cos(perp)) / 111320.0
    dlon = (half * math.sin(perp)) / (111320.0 * math.cos(math.radians(sf_lat)))
    return ((sf_lat + dlat, sf_lon + dlon), (sf_lat - dlat, sf_lon - dlon))


def read_laps(path):
    """Per-lap detail from the FIT 'lap' messages: lap time (native
    total_elapsed_time), HotLap sectors 1-8, pit-stop time, and in/out flag."""
    laps = []
    try:
        fit = fitparse.FitFile(path)
        for m in fit.get_messages("lap"):
            v = {d.name: d.value for d in m}
            laps.append({
                "time": v.get("total_elapsed_time"),          # s
                "start_time": v.get("start_time"),            # datetime, lap start
                "sectors": [v.get("sector%d" % i) for i in range(1, 9)],
                "pit": v.get("pit_time"),                     # s
                "flag": v.get("lap_flag"),                    # 0/1/2
                "max_lat_g": v.get("lap_max_lat_g"),          # g
                "max_brake_g": v.get("lap_max_brake_g"),      # g
            })
    except Exception:
        pass
    return laps


def read_session(path):
    """Session-scope HotLap fields (track name, pit aggregates, best laps)."""
    keys = ("track", "pit_count", "pit_best", "pit_total",
            "best_lap", "theoretical_best", "max_g",
            "max_speed", "avg_speed", "max_lat_g", "max_brake_g",
            "max_accel_g", "max_temp")
    out = {}
    try:
        fit = fitparse.FitFile(path)
        for m in fit.get_messages("session"):
            for d in m:
                if d.name in keys and d.value is not None:
                    out[d.name] = d.value
                    # keep the recorded unit for speed/temp (km/h vs mph, C vs F)
                    if d.name in ("max_speed", "avg_speed", "max_temp"):
                        u = getattr(d, "units", None)
                        if u:
                            out[d.name + "_u"] = u
            break        # first session message only
    except Exception:
        pass
    return out


def fmt_lap_time(s):
    """Seconds -> M:SS.hh (e.g. 72.34 -> '1:12.34')."""
    if s is None:
        return "--"
    s = float(s)
    m = int(s // 60)
    return "%d:%05.2f" % (m, s - 60 * m)


def fmt_secs(s):
    return "--" if s is None else "%.1fs" % float(s)


def fmt_speed(val, unit):
    """Speed with its recorded unit; add the km/h equivalent when it's mph, so
    it lines up with the VBO velocity channel (always km/h)."""
    if val is None:
        return None
    v = float(val); u = (unit or "").strip().lower()
    if u == "mph":
        return "%.1f mph (%.1f km/h)" % (v, v * 1.609344)
    if u in ("km/h", "kph", "kmh"):
        return "%.1f km/h" % v
    return "%.1f%s" % (v, (" " + unit) if unit else "")


def fmt_temp(val, unit):
    if val is None:
        return None
    v = float(val); u = (unit or "").strip().upper()
    if u in ("C", "F"):
        return "%.0f°%s" % (v, u)
    return "%.0f%s" % (v, (" " + unit) if unit else "")


def meta_comment_lines(laps, session):
    """Build the [comments] lines that carry the lap/session custom fields."""
    out = []
    if session:
        tn = session.get("track")
        if tn:
            out.append("Track: %s" % tn)
        pc = session.get("pit_count")
        if pc:
            out.append("Pit stops: %s  best %s  total %s"
                       % (pc, fmt_secs(session.get("pit_best")),
                          fmt_secs(session.get("pit_total"))))
        extra = []
        if session.get("best_lap"):
            extra.append("best lap %s" % fmt_lap_time(session.get("best_lap")))
        if session.get("theoretical_best"):
            extra.append("theo best %s"
                         % fmt_lap_time(session.get("theoretical_best")))
        if session.get("max_g") is not None:
            extra.append("max G %.2f" % float(session.get("max_g")))
        if session.get("max_speed") is not None:
            extra.append("max spd " + fmt_speed(session.get("max_speed"),
                                                 session.get("max_speed_u")))
        if session.get("avg_speed") is not None:
            extra.append("avg spd " + fmt_speed(session.get("avg_speed"),
                                                 session.get("avg_speed_u")))
        if session.get("max_lat_g") is not None:
            extra.append("max lat %.2fg" % float(session.get("max_lat_g")))
        if session.get("max_brake_g") is not None:
            extra.append("max brake %.2fg" % float(session.get("max_brake_g")))
        if session.get("max_accel_g") is not None:
            extra.append("max accel %.2fg" % float(session.get("max_accel_g")))
        if session.get("max_temp") is not None:
            extra.append("max temp " + fmt_temp(session.get("max_temp"),
                                                 session.get("max_temp_u")))
        if extra:
            out.append("Session: " + " | ".join(extra))
    if laps:
        out.append("Lap detail (time | sectors s | pit | max G):")
        flag_lbl = {1: " (in)", 2: " (out)"}
        for i, l in enumerate(laps):
            secs = [x for x in l["sectors"] if x is not None]
            spart = ("S " + "/".join("%.1f" % float(x) for x in secs)) \
                if secs else ""
            pit = l.get("pit")
            ppart = ("  pit %.1f" % float(pit)) if pit else ""
            mlg = l.get("max_lat_g"); mbg = l.get("max_brake_g")
            gs = []
            if mlg is not None:
                gs.append("lat %.2f" % float(mlg))
            if mbg is not None:
                gs.append("brk %.2f" % float(mbg))
            gpart = ("  G(" + "/".join(gs) + ")") if gs else ""
            lbl = flag_lbl.get(l.get("flag") or 0, "")
            out.append("Lap %d  %s  %s%s%s%s"
                       % (i + 1, fmt_lap_time(l.get("time")), spart, ppart, gpart, lbl))
    return out


def tag_laps(rows, laps):
    """Tag each row with a Lap/LapNumber and SectorNumber, matching the columns
    Circuit Tools exports (which is what your analysis tool reads to show laps):
      LapNumber    0 before the first S/F crossing (out-lap), then 1, 2, 3, ...
      SectorNumber 1..M within a lap (by cumulative sector time), 0 when the lap
                   has no recorded sectors (out-lap or the partial final lap).
    HotLap records the leading out-lap as its own FIT lap, so lap numbering keys
    off whether the first FIT lap has sectors (out-lap) or not. VERIFY on your
    data: if lap 1 doesn't line up with your first flying lap, the out-lap
    handling here needs adjusting."""
    for r in rows:
        r["lap_no"] = 0
        r["sec_no"] = 0
    if not laps:
        return
    # If the first FIT lap already has sectors it's a real timed lap (no leading
    # out-lap recorded) -> number from 1. Otherwise the first lap IS the out-lap.
    first_secs = [x for x in laps[0]["sectors"] if x is not None]
    offset = 1 if first_secs else 0

    bounds = []
    for l in laps:
        st = l.get("start_time")
        dur = l.get("time")
        if st is not None and dur is not None:
            bounds.append((st, st + dt.timedelta(seconds=float(dur))))
        else:
            bounds.append(None)

    last_end = None
    last_num = offset - 1
    for i, b in enumerate(bounds):
        if b is not None:
            last_end = b[1]
            last_num = i + offset

    for r in rows:
        ts = r["ts"]
        placed = False
        for i, b in enumerate(bounds):
            if b is None:
                continue
            (s0, s1) = b
            if s0 <= ts and ts < s1:
                r["lap_no"] = i + offset
                secs = [x for x in laps[i]["sectors"] if x is not None]
                if secs:
                    acc = s0
                    sn = len(secs)              # default: final sector
                    for k, sd in enumerate(secs):
                        nxt = acc + dt.timedelta(seconds=float(sd))
                        if ts < nxt:
                            sn = k + 1
                            break
                        acc = nxt
                    r["sec_no"] = sn
                placed = True
                break
        # Rows past the last completed lap = the partial in-progress lap.
        if not placed and last_end is not None and ts >= last_end:
            r["lap_no"] = last_num + 1
            r["sec_no"] = 0


def write_vbo(rows, out_path, gate=None, laps=None, session=None):
    tag_laps(rows, laps or [])
    created = dt.datetime.now()
    # Channel list (header) and matching short column tokens (data order).
    header_channels = [
        "satellites", "time", "latitude", "longitude",
        "velocity kmh", "heading", "height",
        "LongAcc", "LatAcc",
        "Lap", "Lap Number", "Sector Number",
        "heart_rate",
    ]
    columns = [
        "sats", "time", "lat", "long",
        "velocity", "heading", "height",
        "LongAcc", "LatAcc",
        "Lap", "LapNumber", "SectorNumber",
        "heart_rate",
    ]

    # Optional extra channels, included only if the FIT actually has data.
    has_alt = any(r.get("alt") is not None for r in rows)
    has_temp = any(r.get("temp") is not None for r in rows)
    has_spo2 = any(r.get("spo2") is not None for r in rows)
    has_resp = any(r.get("resp") is not None for r in rows)
    if has_alt:
        header_channels.append("ascent m"); columns.append("ascent")
        header_channels.append("descent m"); columns.append("descent")
    if has_temp:
        header_channels.append("temperature C"); columns.append("temp")
    if has_spo2:
        header_channels.append("SpO2 %"); columns.append("spo2")
    if has_resp:
        header_channels.append("respiration bpm"); columns.append("resp")

    lines = []
    lines.append("File created on %s @ %s"
                 % (created.strftime("%d/%m/%Y"), created.strftime("%H:%M:%S")))
    lines.append("")
    lines.append("[header]")
    lines.extend(header_channels)
    lines.append("")
    lines.append("[comments]")
    lines.append("Converted from Garmin FIT by HotLap fit2vbo.py")
    lines.append("Longitude is WEST-positive and lat/long are in minutes "
                 "(VBOX convention).")
    lines.extend(meta_comment_lines(laps, session))
    lines.append("")

    # Start/finish gate so Circuit Tools auto-splits laps. Endpoints use the
    # same coordinate encoding as the data rows, so the crossing detection
    # matches regardless of absolute convention.
    if gate is not None:
        (a, b) = gate
        lines.append("[laptiming]")
        lines.append("Start " + vbo_lat(a[0]) + " " + vbo_long(a[1]) + " "
                     + vbo_lat(b[0]) + " " + vbo_long(b[1]))
        lines.append("")

    lines.append("[column names]")
    lines.append(" ".join(columns))
    lines.append("")
    lines.append("[data]")

    for r in rows:
        sats = 10  # FIT has no sat count; VBOX just needs a valid fix
        speed_kmh = (r["speed"] or 0.0) * 3.6
        height = r["alt"] if r["alt"] is not None else 0.0
        hr = int(r["hr"]) if r["hr"] is not None else 0
        vals = [
            "%02d" % sats,
            vbo_time(r["t"]),
            vbo_lat(r["lat"]),
            vbo_long(r["lon"]),
            "%.3f" % speed_kmh,
            "%.2f" % r["heading"],
            "%.2f" % height,
            "%+.4f" % r["LongAcc"],
            "%+.4f" % r["LatAcc"],
            "%d" % r.get("lap_no", 0),      # Lap (== LapNumber, per the export)
            "%d" % r.get("lap_no", 0),      # LapNumber
            "%d" % r.get("sec_no", 0),      # SectorNumber
            "%d" % hr,
        ]
        if has_alt:
            vals.append("%.1f" % r.get("ascent", 0.0))
            vals.append("%.1f" % r.get("descent", 0.0))
        if has_temp:
            vals.append("%.1f" % (r["temp"] if r["temp"] is not None else 0))
        if has_spo2:
            vals.append("%d" % (r["spo2"] if r["spo2"] is not None else 0))
        if has_resp:
            vals.append("%.1f" % (r["resp"] if r["resp"] is not None else 0))
        lines.append(" ".join(vals))

    with open(out_path, "w", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 fit2vbo.py activity.fit [out.vbo]")
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else \
        (src.rsplit(".", 1)[0] + ".vbo")

    rows = read_records(src)
    if not rows:
        sys.exit("No GPS records with position+time found in %s" % src)
    rows = enrich(rows)
    gate = start_finish_gate(src, rows)
    laps = read_laps(src)
    session = read_session(src)
    write_vbo(rows, out, gate, laps, session)

    used_dev = any(r["g_lat"] is not None for r in rows)
    span = (rows[-1]["ts"] - rows[0]["ts"]).total_seconds()
    print("wrote %s" % out)
    print("  samples : %d over %.0fs" % (len(rows), span))
    print("  accel   : %s"
          % ("from device g_lat/g_long fields"
             if used_dev else "derived from GPS (no g fields in FIT)"))
    print("  laps    : %s"
          % ("start/finish gate embedded (Circuit Tools auto-splits)"
             if gate is not None
             else "no lap markers found -- set start/finish in Circuit Tools"))
    if session.get("track"):
        print("  track   : %s" % session["track"])
    if session.get("pit_count"):
        print("  pit     : %s stop(s), best %s, total %s"
              % (session["pit_count"], fmt_secs(session.get("pit_best")),
                 fmt_secs(session.get("pit_total"))))
    if laps:
        nsec = max((len([x for x in l["sectors"] if x is not None])
                    for l in laps), default=0)
        print("  lap meta: %d lap record(s), up to %d sector(s) each, "
              "in [comments]" % (len(laps), nsec))


if __name__ == "__main__":
    main()
