#!/bin/bash
# The watch-only page (spotter_web.py --view) on another computer — a Windows box with Docker Desktop, reached by
# SSH with a key (its SSH shell PowerShell) — serving a copy of the analysed data: pilots watch their races,
# spotters approve missed gates with the spotter code. Nothing is analysed there.
#
#   ./deploy-optiplex.sh               code + data (the first time: everything)
#   ./deploy-optiplex.sh code          the committed code (git HEAD): rebuild the image, restart the container
#   ./deploy-optiplex.sh data          the analysed data that changed (deploy/view_files.py), no restart needed
#   ./deploy-optiplex.sh approvals     the spotters' approvals made there → this computer's tracks (backed up first)
#   ./deploy-optiplex.sh spotter-code     show the spotter code
#   DRY_RUN=1 ./deploy-optiplex.sh data    list what would be sent
#
# Where: deploy.env (not in the repository; see deploy.env.example). On the box, in $DIR:
#   build\  the image's source    data\  the copy, mounted at /data    .env  SPOTTER_KEY=…    docker-spotter.bat
# A track's spotter_approved.json is the box's once it's there: the data sync never replaces it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/deploy.env" ] || { echo "deploy.env is missing: copy deploy.env.example and fill it in"; exit 1; }
# shellcheck disable=SC1091
source "$HERE/deploy.env"
: "${OPTIPLEX_HOST:?}" "${OPTIPLEX_USER:?}" "${VIEW_TRACKS:?}"
DIR="${OPTIPLEX_DIR:-C:\\services\\fpv-spotter}"
PORT="${VIEW_PORT:-5190}"
CHUNK_MB="${CHUNK_MB:-300}"                      # small parts: a stalled one is sent again (below)
T="$OPTIPLEX_USER@$OPTIPLEX_HOST"
OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30)
PY="$HERE/venv/bin/python"
DIRF="${DIR//\\//}"                                  # C:/services/fpv-spotter, for scp

# a PowerShell script on the box (sent encoded: no quoting through ssh)
# (its progress records off: over SSH they come out as CLIXML in the output)
# (each one 15 min at most: an SSH session to the box has hung after its script ended, 2026-11-16)
ps() { perl -e 'alarm shift; exec @ARGV' 900 ssh "${OPTS[@]}" "$T" "powershell -NoProfile -NonInteractive -EncodedCommand $(printf '%s' "\$ProgressPreference = 'SilentlyContinue'; $1" | iconv -t UTF-16LE | base64 | tr -d '\n')" 2>&1 | tr -d '\r'; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

deploy_code() {
  echo "[code] staging $(git -C "$HERE" rev-parse --short HEAD)"
  mkdir -p "$WORK/code/build/app"
  # the committed code: the top-level modules, race_day/ and the web app (no models, data or tests)
  git -C "$HERE" archive HEAD -- ':(glob)*.py' race_day web | tar -x -C "$WORK/code/build/app"
  cp "$HERE/deploy/Dockerfile.view" "$HERE/deploy/requirements-view.txt" "$WORK/code/build/"
  sed 's/$/\r/' "$HERE/deploy/docker-spotter.bat" > "$WORK/code/docker-spotter.bat"
  COPYFILE_DISABLE=1 tar -c --no-xattrs --no-mac-metadata -f "$WORK/code.tar" -C "$WORK/code" .
  ps "New-Item -ItemType Directory -Force '$DIR\\_incoming','$DIR\\data' | Out-Null" >/dev/null
  scp -q "${OPTS[@]}" "$WORK/code.tar" "$T:$DIRF/_incoming/code.tar"
  KEY="$(openssl rand -base64 12 | tr -d '/+=' | cut -c1-12)"
  echo "[code] unpacking (the previous build kept as build.prev)"
  ps "\$ErrorActionPreference = 'Stop'; Set-Location '$DIR'
    Remove-Item -Recurse -Force _incoming\\code, build.prev -ErrorAction SilentlyContinue
    New-Item -ItemType Directory _incoming\\code | Out-Null
    & C:\\Windows\\System32\\tar.exe -xf _incoming\\code.tar -C _incoming\\code; if (\$LASTEXITCODE) { throw 'tar failed' }
    Move-Item -Force _incoming\\code\\docker-spotter.bat docker-spotter.bat
    if (Test-Path build) { Rename-Item build build.prev }
    Move-Item _incoming\\code\\build build
    Remove-Item -Recurse -Force _incoming\\code, _incoming\\code.tar
    if (-not (Test-Path .env) -or -not (Select-String -Quiet -Path .env -Pattern '^SPOTTER_KEY=.+')) {
      Set-Content -Encoding ascii .env 'SPOTTER_KEY=$KEY'; 'new spotter code: $KEY' }"
  echo "[code] building and starting (port $PORT)"
  # over SSH Docker Desktop's credential helper has no logon session: a copy of the config without it, and the
  # classic builder, pull the public base image without it (as fpvsim's deploy does)
  ps "\$src = \"\$env:USERPROFILE\\.docker\"; \$dst = '$DIR\\_dockercfg'
    Remove-Item -Recurse -Force \$dst -ErrorAction SilentlyContinue; New-Item -ItemType Directory \$dst | Out-Null
    foreach (\$d in 'contexts','buildx','cli-plugins') { if (Test-Path \"\$src\\\$d\") { Copy-Item -Recurse \"\$src\\\$d\" \$dst } }
    \$j = Get-Content \"\$src\\config.json\" -Raw | ConvertFrom-Json
    \$j.PSObject.Properties.Remove('credsStore'); \$j.PSObject.Properties.Remove('credHelpers')
    (\$j | ConvertTo-Json -Depth 10) | Set-Content \"\$dst\\config.json\" -Encoding ascii
    \$env:DOCKER_CONFIG = \$dst; \$env:DOCKER_BUILDKIT = '0'; \$env:PORT = '$PORT'
    cmd /c '$DIR\\docker-spotter.bat' 2>&1 | Select-Object -Last 12; exit \$LASTEXITCODE"
}

deploy_data() {
  echo "[data] what's here (${VIEW_TRACKS})"
  # shellcheck disable=SC2086
  "$PY" "$HERE/deploy/view_files.py" --sizes ${VIEW_TRACKS//,/ } > "$WORK/local.tsv"
  echo "[data] what's there"
  ps "\$r = '$DIR\\data'; if (Test-Path \$r) { Get-ChildItem \$r -Recurse -File | ForEach-Object {
      \"{0}\`t{1}\`t{2}\" -f \$_.Length, [DateTimeOffset]::new(\$_.LastWriteTimeUtc).ToUnixTimeSeconds(),
      \$_.FullName.Substring(\$r.Length + 1).Replace('\\', '/') } }" > "$WORK/remote.tsv"
  ROOT="$("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); from dataset_paths import data_root; print(data_root())' "$HERE")"
  "$PY" - "$WORK" "$CHUNK_MB" <<'EOF'
import sys
from pathlib import Path
work, chunk = Path(sys.argv[1]), int(sys.argv[2]) << 20
def read(f):
    out = {}
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].isdigit():
            out[parts[2]] = (int(parts[0]), int(parts[1]))
    return out
here, there = read(work / "local.tsv"), read(work / "remote.tsv")
send = [p for p, (n, t) in here.items()
        if p not in there or (not p.endswith("/spotter_approved.json") and
                              (there[p][0] != n or abs(there[p][1] - t) > 2))]
size, k, lists = 0, 0, []
for p in send:
    if not lists or size + here[p][0] > chunk:
        lists.append([]); size = 0
    lists[-1].append(p); size += here[p][0]
for i, l in enumerate(lists):
    (work / f"chunk{i:03d}.txt").write_text("\n".join(l) + "\n", encoding="utf-8")
total = sum(here[p][0] for p in send)
print(f"[data] {len(here)} files here, {len(there)} there: sending {len(send)} ({total / 2**30:.2f} GB) in {len(lists)} part(s)")
EOF
  shopt -s nullglob
  for L in "$WORK"/chunk*.txt; do
    if [ -n "${DRY_RUN:-}" ]; then sed 's/^/      /' "$L" | head -40; continue; fi
    n="$(basename "$L" .txt)"
    COPYFILE_DISABLE=1 tar -c --no-xattrs --no-mac-metadata -f "$WORK/$n.tar" -C "$ROOT" -T "$L"
    echo "      $n: $(wc -l < "$L" | tr -d ' ') files, $(du -h "$WORK/$n.tar" | cut -f1)"
    ps "New-Item -ItemType Directory -Force '$DIR\\_incoming','$DIR\\data' | Out-Null" >/dev/null
    # a copy to the box can stall for good (seen 2026-11-16: 0 bytes in 16 min): each part gets 15 min, 3 tries
    for try in 1 2 3; do
      perl -e 'alarm shift; exec @ARGV' 900 scp -q "${OPTS[@]}" "$WORK/$n.tar" "$T:$DIRF/_incoming/$n.tar" && break
      [ "$try" = 3 ] && { echo "      $n: not sent after 3 tries"; exit 1; }
      echo "      $n: stalled or failed, again ($try)"
    done
    ps "& C:\\Windows\\System32\\tar.exe -xf '$DIR\\_incoming\\$n.tar' -C '$DIR\\data'; \$e = \$LASTEXITCODE
        Remove-Item '$DIR\\_incoming\\$n.tar'; if (\$e) { 'tar failed'; exit 1 }"
    rm -f "$WORK/$n.tar"
  done
  [ -n "${DRY_RUN:-}" ] && echo "[dry run] nothing sent"
  return 0
}

pull_approvals() {
  ROOT="$("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); from dataset_paths import data_root; print(data_root())' "$HERE")"
  for t in ${VIEW_TRACKS//,/ }; do
    if ! scp -q "${OPTS[@]}" "$T:$DIRF/data/$t/spotter_approved.json" "$WORK/$t.json" 2>/dev/null; then
      echo "      $t: no approvals there"; continue
    fi
    f="$ROOT/$t/spotter_approved.json"
    if [ -f "$f" ] && cmp -s "$f" "$WORK/$t.json"; then echo "      $t: the same here"; continue; fi
    [ -f "$f" ] && cp -p "$f" "$f.$(date +%Y%m%d-%H%M%S).bak"
    cp "$WORK/$t.json" "$f"
    echo "      $t: approvals updated here (the old ones kept as .bak)"
  done
}

case "${1:-all}" in
  code) deploy_code ;;
  data) deploy_data ;;
  all) deploy_data; deploy_code ;;
  approvals) pull_approvals ;;
  spotter-code) ps "Get-Content '$DIR\\.env' | Select-String '^SPOTTER_KEY=' | ForEach-Object { \$_.Line.Substring(12) }" ;;
  *) sed -n '6,12p' "$0"; exit 1 ;;
esac
[ "${1:-all}" = approvals ] || [ "${1:-all}" = spotter-code ] || [ -n "${DRY_RUN:-}" ] || \
  echo "      page: http://$OPTIPLEX_HOST:$PORT/"
