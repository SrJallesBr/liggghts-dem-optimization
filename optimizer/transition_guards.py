"""Fail-closed transition checks. No code here signals or terminates solvers."""
import hashlib
from pathlib import Path

# Reviewed candidate bytes, embedded independently of the mutable sidecar.
CANDIDATE_SHA256 = {
    'Ang Repouso/in.repouso': '5cdd4801811e461c696a45d6de8d0c13d4785fed12c05297bfa3a947ccc19039',
    'Ang Deslizamento/in.deslizamento': 'bd3b5d5c5021f95c0e7378095bca2973e76fc49c76b82ff5200259254c3612c6',
    'Ang Rolamento/in.fill_1s': 'b558588d1e4358e4bfe0b8ba81a27f9f9a0333bd98313ff1cb9079b56b6729b2',
    'Ang Rolamento/in.rolamento': '3428802206947943e03235849915b5815dcb776b8eb54fc1f54dd2503927cda6',
}

def verify_candidate_pins(root):
    root = Path(root)
    for rel, expected in CANDIDATE_SHA256.items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != expected:
            raise ValueError('candidate pin mismatch: ' + rel)
    return True

def drain_requested(root):
    # Existence, not contents, is the signal. Leave it in place for restart safety.
    return (Path(root) / '.drain-adaptive').exists()
