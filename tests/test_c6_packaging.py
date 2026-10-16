"""C6: Packaging, desktop integration, systemd, Docker & CI."""
import configparser
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def test_desktop_file_exists():
    """Desktop file exists at standard path."""
    assert (ROOT / "packaging/deb/photocure.desktop").exists()


def test_desktop_file_canonical_app_id():
    """Desktop file uses canonical app ID io.github.safiyu.photocure."""
    content = (ROOT / "packaging/deb/photocure.desktop").read_text()

    assert "Icon=io.github.safiyu.photocure" in content
    assert "Name=Photocure" in content
    assert "Exec=/usr/bin/photocure start" in content
    assert "Terminal=false" in content
    assert "Type=Application" in content
    assert "Categories=Graphics;Photography;" in content


def test_metainfo_file_exists():
    """AppStream metainfo file exists."""
    assert (ROOT / "packaging/deb/photocure.metainfo.xml").exists()


def test_metainfo_structure():
    """AppStream metainfo parses and carries the required fields."""
    tree = ET.parse(ROOT / "packaging/deb/photocure.metainfo.xml").getroot()

    assert tree.findtext("id") == "io.github.safiyu.photocure"
    assert tree.findtext("metadata_license") == "CC0-1.0"
    assert tree.findtext("project_license") == "MIT"
    assert tree.findtext("name") == "Photocure"
    assert tree.find("developer_name") is None, "deprecated tag"
    assert tree.findtext("developer/name") == "Safiyu"
    assert tree.find("developer").get("id")
    assert tree.find("icon").get("type") == "stock"
    assert tree.findtext("launchable") == "io.github.safiyu.photocure.desktop"
    cats = [c.text for c in tree.findall("categories/category")]
    assert "Graphics" in cats
    assert "Photography" in cats


def test_systemd_service_exists():
    """Systemd user service file exists."""
    assert (ROOT / "packaging/deb/photocure.service").exists()


def test_systemd_service_structure():
    """Service uses a type the app supports and keeps start limits in [Unit]."""
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read(ROOT / "packaging/deb/photocure.service")

    assert cp["Service"]["Type"] in {"simple", "exec"}, "app never calls sd_notify"
    assert cp["Service"]["ExecStart"] == "/usr/bin/photocure serve --no-open"
    assert cp["Service"]["Restart"] == "on-failure"
    assert cp["Unit"]["StartLimitBurst"] == "5"
    assert cp["Unit"]["StartLimitIntervalSec"] == "60s"
    assert "StartLimitBurst" not in cp["Service"]
    assert "StartLimitIntervalSec" not in cp["Service"]
    assert cp["Install"]["WantedBy"] == "default.target"


def test_no_sd_notify_when_type_notify():
    """Type=notify is only allowed if the code can send READY=1."""
    svc = (ROOT / "packaging/deb/photocure.service").read_text()
    if "Type=notify" in svc:
        src = "".join(p.read_text() for p in (ROOT / "photocure").rglob("*.py"))
        assert "NOTIFY_SOCKET" in src or "sd_notify" in src


def test_docker_entrypoint_exists():
    """Docker entrypoint script exists and is executable."""
    ep = ROOT / "packaging/docker/entrypoint.sh"
    assert ep.exists()
    assert ep.stat().st_mode & 0o111, "Entrypoint should be executable"


def _run_entrypoint(tmp_path, group_exists, user_exists, owner="0:0"):
    """Run the entrypoint with fake system commands that log their calls."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    stubs = {
        "getent": f'if [ "$1" = group ]; then {"true" if group_exists else "false"}; else {"true" if user_exists else "false"}; fi',
        "groupadd": 'echo "groupadd $*" >> "$LOG"',
        "useradd": 'echo "useradd $*" >> "$LOG"',
        "chown": 'echo "chown $*" >> "$LOG"',
        "stat": f'echo {owner}',
        "gosu": 'echo "gosu $* HOME=$HOME" >> "$LOG"',
    }
    for name, body in stubs.items():
        f = bindir / name
        f.write_text("#!/bin/bash\n" + body + "\n")
        f.chmod(0o755)
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "LOG": str(log),
        "PUID": "5000",
        "PGID": "1000",
        "PCUR_DATA_DIR": "/data",
        "HOME": "/root",
    }
    subprocess.run(
        ["/bin/bash", str(ROOT / "packaging/docker/entrypoint.sh")],
        env=env, check=True,
    )
    return log.read_text().splitlines()


def test_docker_entrypoint_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "packaging/docker/entrypoint.sh")], check=True)


def test_docker_entrypoint_creates_missing_group_and_user(tmp_path):
    calls = _run_entrypoint(tmp_path, group_exists=False, user_exists=False)

    assert "groupadd -g 1000 photocure" in calls
    assert any(c.startswith("useradd -u 5000 -g 1000") for c in calls)
    assert "chown -R 5000:1000 /data" in calls


def test_docker_entrypoint_skips_group_when_it_exists(tmp_path):
    calls = _run_entrypoint(tmp_path, group_exists=True, user_exists=True)

    assert not any(c.startswith(("groupadd", "useradd")) for c in calls)


def test_docker_entrypoint_drops_privileges_with_writable_home(tmp_path):
    calls = _run_entrypoint(tmp_path, group_exists=True, user_exists=True)

    gosu = [c for c in calls if c.startswith("gosu")]
    assert len(gosu) == 1
    assert gosu[0].startswith("gosu 5000:1000 /usr/bin/photocure serve --host 0.0.0.0")
    assert gosu[0].endswith("HOME=/data")


def test_docker_entrypoint_skips_chown_when_already_owned(tmp_path):
    calls = _run_entrypoint(tmp_path, True, True, owner="5000:1000")

    assert not any(c.startswith("chown") for c in calls)


def test_dockerfile_builds_wheel_and_probes_health():
    """Wheel build may not be swallowed and the probe must hit the server."""
    content = (ROOT / "packaging/docker/Dockerfile").read_text()

    assert "FROM ubuntu:24.04" in content
    assert "|| true" not in content
    assert "/dev/null 2>&1" not in content.split("pip wheel", 1)[1].splitlines()[0]
    health = content.split("HEALTHCHECK", 1)[1]
    assert "/healthz" in health
    assert "--version" not in health
    assert "--start-period=1800s" in health


def test_docker_compose_runtime_user_and_healthcheck():
    """Compose runs as PUID:PGID and its healthcheck works without curl."""
    yaml = pytest.importorskip("yaml")
    svc = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["photocure"]

    assert svc["user"] == "${PUID:-1000}:${PGID:-1000}"
    assert "PUID" not in svc["environment"], "root image never reads PUID"
    hc = svc["healthcheck"]
    assert "curl" not in hc["test"]
    assert any("/healthz" in part for part in hc["test"])
    assert hc["start_period"] == "1800s"


def test_compose_image_has_no_curl_dependency():
    """Compose builds the root Dockerfile, which must not rely on curl."""
    df = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()

    assert "curl" not in df
    assert '"curl"' not in compose


def test_build_deb_script_exists():
    """build_deb.sh exists and is executable."""
    script = ROOT / "packaging/deb/build_deb.sh"
    assert script.exists()
    assert script.stat().st_mode & 0o111, "build_deb.sh should be executable"


def test_build_deb_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "packaging/deb/build_deb.sh")], check=True)


def test_build_deb_creates_icons():
    """build_deb.sh creates multi-resolution icons."""
    content = (ROOT / "packaging/deb/build_deb.sh").read_text()

    for size in ("16x16", "256x256", "512x512", "io.github.safiyu.photocure.png"):
        assert size in content


def test_build_deb_includes_metainfo():
    """build_deb.sh copies metainfo and systemd service."""
    content = (ROOT / "packaging/deb/build_deb.sh").read_text()

    for needle in ("photocure.metainfo.xml", "photocure.service",
                   "usr/share/metainfo", "usr/lib/systemd/user"):
        assert needle in content


def _ci_steps(job):
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    return {s.get("name", ""): s for s in wf["jobs"][job]["steps"]}


def test_ci_packaging_job_installs_validators_it_calls():
    steps = _ci_steps("packaging")
    install = steps["Install build tools"]["run"]

    assert "appstreamcli validate" in steps["Validate AppStream metainfo"]["run"]
    assert "appstream" in install.replace("appstream-glib", "")
    assert "desktop-file-utils" in install
    assert "desktop-file-validate" in steps["Validate desktop file"]["run"]


def test_ci_smoke_test_can_fail_and_resolves_dependencies():
    steps = _ci_steps("packaging")
    smoke = next(s for n, s in steps.items() if n.startswith("Smoke test"))["run"]
    lines = [ln.strip() for ln in smoke.splitlines() if ln.strip()]

    assert "set -e" in " ".join(lines)
    assert any("apt-get install" in ln and ".deb" in ln for ln in lines)
    assert not any("dpkg -i" in ln for ln in lines)
    version = next(ln for ln in lines if "--version" in ln)
    assert "||" not in version
    assert all("stop" in ln for ln in lines if "|| true" in ln)


def test_notice_and_license_exist():
    """NOTICE and LICENSE files exist."""
    assert (ROOT / "NOTICE").exists()
    assert (ROOT / "LICENSE").exists()
