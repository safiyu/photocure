"""C6: Packaging, desktop integration, systemd, Docker & CI."""
from pathlib import Path
import subprocess


def test_desktop_file_exists():
    """Desktop file exists at standard path."""
    root = Path(__file__).parent.parent
    assert (root / "packaging/deb/photocure.desktop").exists()


def test_desktop_file_canonical_app_id():
    """Desktop file uses canonical app ID io.github.safiyu.photocure."""
    root = Path(__file__).parent.parent
    content = (root / "packaging/deb/photocure.desktop").read_text()

    assert "Icon=io.github.safiyu.photocure" in content
    assert "Name=Photocure" in content
    assert "Exec=/usr/bin/photocure start" in content
    assert "Terminal=false" in content
    assert "Type=Application" in content
    assert "Categories=Graphics;Photography;" in content


def test_metainfo_file_exists():
    """AppStream metainfo file exists."""
    root = Path(__file__).parent.parent
    assert (root / "packaging/deb/photocure.metainfo.xml").exists()


def test_metainfo_structure():
    """AppStream metainfo has correct structure."""
    root = Path(__file__).parent.parent
    content = (root / "packaging/deb/photocure.metainfo.xml").read_text()

    assert '<id>io.github.safiyu.photocure</id>' in content
    assert '<metadata_license>CC0-1.0</metadata_license>' in content
    assert '<project_license>MIT</project_license>' in content
    assert '<name>Photocure</name>' in content
    assert '<developer_name>Safiyu</developer_name>' in content
    assert 'type="stock">io.github.safiyu.photocure' in content
    assert 'type="desktop-id">io.github.safiyu.photocure.desktop' in content


def test_systemd_service_exists():
    """Systemd user service file exists."""
    root = Path(__file__).parent.parent
    assert (root / "packaging/deb/photocure.service").exists()


def test_systemd_service_structure():
    """Systemd service has correct structure for user unit."""
    root = Path(__file__).parent.parent
    content = (root / "packaging/deb/photocure.service").read_text()

    assert "[Unit]" in content
    assert "[Service]" in content
    assert "[Install]" in content
    assert "ExecStart=/usr/bin/photocure serve --no-open" in content
    assert "Restart=on-failure" in content
    assert "RestartSec=5s" in content
    assert "StartLimitBurst=5" in content
    assert "StartLimitIntervalSec=60s" in content
    assert "Environment=PYTHONUNBUFFERED=1" in content


def test_docker_entrypoint_exists():
    """Docker entrypoint script exists and is executable."""
    root = Path(__file__).parent.parent
    ep = root / "packaging/docker/entrypoint.sh"
    assert ep.exists()
    assert ep.stat().st_mode & 0o111, "Entrypoint should be executable"


def test_docker_entrypoint_structure():
    """Docker entrypoint handles PUID/PGID correctly."""
    root = Path(__file__).parent.parent
    content = (root / "packaging/docker/entrypoint.sh").read_text()

    assert "PUID=" in content
    assert "PGID=" in content
    assert "groupadd" in content
    assert "useradd" in content
    assert "chown -R" in content
    assert "gosu" in content
    assert "photocure serve --host 0.0.0.0" in content


def test_dockerfile_exists():
    """Dockerfile exists and builds from Ubuntu."""
    root = Path(__file__).parent.parent
    df = root / "packaging/docker/Dockerfile"
    assert df.exists()
    content = df.read_text()

    assert "FROM ubuntu:24.04" in content
    assert "gosu" in content
    assert "python3 python3-venv python3-pip" in content
    assert "HEALTHCHECK" in content


def test_docker_compose_puid_pgid():
    """docker-compose.yml has PUID/PGID environment variables."""
    root = Path(__file__).parent.parent
    content = (root / "docker-compose.yml").read_text()

    assert "PUID:" in content
    assert "PGID:" in content
    assert "healthcheck:" in content


def test_build_deb_script_exists():
    """build_deb.sh exists and is executable."""
    root = Path(__file__).parent.parent
    script = root / "packaging/deb/build_deb.sh"
    assert script.exists()
    assert script.stat().st_mode & 0o111, "build_deb.sh should be executable"


def test_build_deb_creates_icons():
    """build_deb.sh creates multi-resolution icons."""
    root = Path(__file__).parent.parent
    content = (root / "packaging/deb/build_deb.sh").read_text()

    # Check that it creates multiple icon sizes
    assert "16x16" in content
    assert "256x256" in content
    assert "512x512" in content
    assert "io.github.safiyu.photocure.png" in content


def test_build_deb_includes_metainfo():
    """build_deb.sh copies metainfo and systemd service."""
    root = Path(__file__).parent.parent
    content = (root / "packaging/deb/build_deb.sh").read_text()

    assert "photocure.metainfo.xml" in content
    assert "photocure.service" in content
    assert "usr/share/metainfo" in content
    assert "usr/lib/systemd/user" in content


def test_ci_workflow_includes_packaging_job():
    """CI workflow includes packaging validation steps."""
    root = Path(__file__).parent.parent
    content = (root / ".github/workflows/ci.yml").read_text()

    assert "appstreamcli validate" in content
    assert "desktop-file-validate" in content
    assert "dpkg" in content


def test_notice_and_license_exist():
    """NOTICE and LICENSE files exist."""
    root = Path(__file__).parent.parent
    assert (root / "NOTICE").exists()
    assert (root / "LICENSE").exists()


if __name__ == "__main__":
    test_desktop_file_exists()
    test_desktop_file_canonical_app_id()
    test_metainfo_file_exists()
    test_metainfo_structure()
    test_systemd_service_exists()
    test_systemd_service_structure()
    test_docker_entrypoint_exists()
    test_docker_entrypoint_structure()
    test_dockerfile_exists()
    test_docker_compose_puid_pgid()
    test_build_deb_script_exists()
    test_build_deb_creates_icons()
    test_build_deb_includes_metainfo()
    test_ci_workflow_includes_packaging_job()
    test_notice_and_license_exist()
    print("✓ All C6 tests passed")
