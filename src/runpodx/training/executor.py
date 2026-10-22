from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("runpodx.executor")


@dataclass
class SSHConnection:
    """SSH connection parameters for a RunPod pod."""

    host: str
    user: str = "root"
    port: int = 22
    key_path: str | None = None
    key_passphrase: str | None = None
    connect_timeout: int = 30
    _unlocked_key: str | None = field(default=None, repr=False, init=False)

    def _ensure_unlocked_key(self) -> str:
        """Return path to a passphrase-free private key.

        If the key has a passphrase, creates a temporary copy with the
        passphrase stripped using ``ssh-keygen -p``.  This avoids the
        SSH_ASKPASS dance which doesn't work reliably on Windows.
        """
        key_path = str(Path(self.key_path or "~/.ssh/id_ed25519").expanduser())

        if not self.key_passphrase:
            return key_path

        if self._unlocked_key and Path(self._unlocked_key).exists():
            return self._unlocked_key

        fd, tmp_path = tempfile.mkstemp(prefix="runpodx_key_", suffix=".tmp")
        os.close(fd)
        shutil.copy2(key_path, tmp_path)

        result = subprocess.run(
            ["ssh-keygen", "-p", "-P", self.key_passphrase, "-N", "", "-f", tmp_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            Path(tmp_path).unlink(missing_ok=True)
            raise RuntimeError(f"Failed to unlock SSH key: {result.stderr}")

        self._unlocked_key = tmp_path
        logger.debug("Created temporary unlocked key at %s", tmp_path)
        return tmp_path

    def cleanup(self) -> None:
        """Remove the temporary unlocked key if one was created."""
        if self._unlocked_key:
            Path(self._unlocked_key).unlink(missing_ok=True)
            # ssh-keygen may also create a .pub file next to it
            Path(self._unlocked_key + ".pub").unlink(missing_ok=True)
            logger.debug("Cleaned up temporary key %s", self._unlocked_key)
            self._unlocked_key = None

    def _ssh_base_args(self) -> list[str]:
        """Common SSH flags (used by ssh)."""
        key = self._ensure_unlocked_key()
        return [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes",
            "-o", "PreferredAuthentications=publickey",
            "-o", "LogLevel=ERROR",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-p", str(self.port),
            "-i", key,
        ]

    def _scp_base_args(self) -> list[str]:
        """Common SCP flags (uses -P for port, not -p)."""
        key = self._ensure_unlocked_key()
        return [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes",
            "-o", "PreferredAuthentications=publickey",
            "-o", "LogLevel=ERROR",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-P", str(self.port),
            "-i", key,
        ]


class SSHExecutor:
    """Execute commands and transfer files to a remote RunPod pod over SSH."""

    def __init__(self, connection: SSHConnection) -> None:
        self._conn = connection

    def test_connection(self) -> bool:
        """Verify SSH connectivity."""
        return self.check_connection().returncode == 0

    def check_connection(self) -> subprocess.CompletedProcess:
        """Run a cheap SSH probe and keep stderr available to callers.

        Bounded by an overall timeout: ``ConnectTimeout`` only covers the TCP
        connect, so a probe whose handshake stalls after connecting would
        otherwise hang forever while the pod keeps billing (observed twice,
        ~13 h on 2026-09-01). ``subprocess.run`` kills the child on expiry and
        the ``TimeoutExpired`` is folded into a non-zero result so the caller's
        retry loop simply tries again.
        """
        probe_timeout = max(60, 2 * self._conn.connect_timeout)
        try:
            result = self.run("echo ok", stream=False, timeout=probe_timeout)
            return result
        except Exception as exc:
            return subprocess.CompletedProcess(
                args=["ssh"],
                returncode=255,
                stdout="",
                stderr=str(exc),
            )

    def run(
        self,
        command: str,
        *,
        stream: bool = True,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess | subprocess.Popen:
        """Execute a command on the remote pod via SSH.

        stream=True: returns Popen, invokes callbacks for each line.
        stream=False: blocks until completion, returns CompletedProcess.
        """
        ssh_cmd = [
            "ssh",
            *self._conn._ssh_base_args(),
            f"{self._conn.user}@{self._conn.host}",
            command,
        ]
        logger.debug("SSH exec: %s", " ".join(ssh_cmd))

        if not stream:
            return subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=timeout,
                stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
            )

        proc = subprocess.Popen(
            ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
        )

        def _read_stream(pipe, callback):
            for line in iter(pipe.readline, ""):
                if callback:
                    callback(line.rstrip("\n"))
            pipe.close()

        stdout_thread = threading.Thread(
            target=_read_stream, args=(proc.stdout, on_stdout), daemon=True
        )
        stderr_thread = threading.Thread(
            target=_read_stream, args=(proc.stderr, on_stderr), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()

        return proc

    def upload(
        self,
        local_path: str | Path,
        remote_path: str,
        *,
        exclude: list[str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Upload a file to the remote pod via scp."""
        local_path = Path(local_path)
        scp_cmd = [
            "scp",
            *self._conn._scp_base_args(),
            str(local_path),
            f"{self._conn.user}@{self._conn.host}:{remote_path}",
        ]
        logger.debug("Upload: %s", " ".join(scp_cmd))
        return subprocess.run(scp_cmd, capture_output=True, text=True)

    def download(
        self,
        remote_path: str,
        local_path: str | Path,
    ) -> subprocess.CompletedProcess:
        """Download a file from the remote pod via scp."""
        scp_cmd = [
            "scp",
            *self._conn._scp_base_args(),
            f"{self._conn.user}@{self._conn.host}:{remote_path}",
            str(local_path),
        ]
        logger.debug("Download: %s", " ".join(scp_cmd))
        return subprocess.run(scp_cmd, capture_output=True, text=True)
