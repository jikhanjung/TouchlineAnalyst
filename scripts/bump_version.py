#!/usr/bin/env python3
"""Bump the version, roll the changelog, and tag the release.

`version.py` is the single source of truth (pyproject.toml,
docs/conf.py and config/constants.py all derive from it — see
tests/test_version_consistency.py). This script automates the release-tag step
so the version, the CHANGELOG entry, the commit, and the `v<version>` tag can
never drift out of sync.

Flow (mirrors Modan2's process — see docs/RELEASE_PROCESS.md):

    1. Choose the new version (bump a part, or set it explicitly).
    2. version.py is rewritten.
    3. CHANGELOG.md's "## [Unreleased]" section is renamed to
       "## [<version>] - <date>" and a fresh empty Unreleased section is added.
    4. version.py + CHANGELOG.md are committed as "chore: release v<version>".
    5. An annotated tag "v<version>" is created on that commit.
    6. With --push, the commit and tag are pushed, which triggers release.yml
       (build all platforms + GitHub release with notes from the CHANGELOG).

Commands (the same vocabulary Modan2's VERSION_MANAGEMENT.md describes):

    major / minor / patch          1.2.3 -> 2.0.0 / 1.3.0 / 1.2.4
    premajor / preminor / prepatch start a pre-release cycle; the optional
      [token]                      token is alpha (default), beta or rc
                                   1.2.3 -> preminor beta -> 1.3.0-beta.1
    prerelease                     bump the pre-release number
                                   1.3.0-beta.1 -> 1.3.0-beta.2
    stage <alpha|beta|rc>          move stage, resetting the number to 1
                                   1.3.0-alpha.4 -> stage beta -> 1.3.0-beta.1
    release                        drop the pre-release suffix
                                   1.3.0-rc.2 -> 1.3.0
    --set X.Y.Z                    an explicit version

Examples:
    python scripts/bump_version.py patch            # 0.2.3 -> 0.2.4
    python scripts/bump_version.py preminor beta    # 0.2.3 -> 0.3.0-beta.1
    python scripts/bump_version.py prerelease       # 0.2.4-beta.1 -> 0.2.4-beta.2
    python scripts/bump_version.py stage rc         # 0.2.4-beta.3 -> 0.2.4-rc.1
    python scripts/bump_version.py release          # 0.2.4-rc.1 -> 0.2.4
    python scripts/bump_version.py patch --push     # also push commit + tag

Dry run:
    python scripts/bump_version.py patch --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

try:
    import semver
except ImportError:
    sys.exit("error: the 'semver' package is required (pip install semver)")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = PROJECT_ROOT / "version.py"
CHANGELOG_FILE = PROJECT_ROOT / "CHANGELOG.md"
VERSION_RE = re.compile(r'^(__version__\s*=\s*")([^"]+)(")', re.MULTILINE)

VALID_STAGES = ("alpha", "beta", "rc")


# --------------------------------------------------------------------------- #
# version.py
# --------------------------------------------------------------------------- #
def read_current_version() -> str:
    text = VERSION_FILE.read_text(encoding="utf-8")
    m = VERSION_RE.search(text)
    if not m:
        sys.exit(f"error: could not find __version__ in {VERSION_FILE}")
    return m.group(2)


def write_version(new_version: str) -> None:
    text = VERSION_FILE.read_text(encoding="utf-8")
    text = VERSION_RE.sub(rf"\g<1>{new_version}\g<3>", text, count=1)
    VERSION_FILE.write_text(text, encoding="utf-8")


def _validate_stage(token: str) -> None:
    if token not in VALID_STAGES:
        sys.exit(f"error: stage must be one of {', '.join(VALID_STAGES)} (got {token!r})")


def _start_prerelease_cycle(ver: semver.VersionInfo, part: str, token: str | None) -> str:
    """premajor/preminor/prepatch: bump the part, then open stage .1."""
    stage = token or "alpha"
    _validate_stage(stage)
    bump = {"premajor": ver.bump_major, "preminor": ver.bump_minor, "prepatch": ver.bump_patch}
    return str(bump[part]().bump_prerelease(token=stage))


def _move_stage(ver: semver.VersionInfo, current: str, token: str | None) -> str:
    """stage <alpha|beta|rc>: change stage, restarting the number at 1."""
    if not ver.prerelease:
        sys.exit(
            "error: 'stage' moves between pre-release stages, and "
            f"{current} is already stable. Use premajor/preminor/prepatch "
            "to start a pre-release cycle."
        )
    if not token:
        sys.exit(f"error: 'stage' needs a token: {', '.join(VALID_STAGES)}")
    _validate_stage(token)
    if ver.prerelease.split(".")[0] == token:
        sys.exit(f"error: already in the {token!r} stage ({current})")
    return str(ver.replace(prerelease=f"{token}.1"))


def compute_new_version(current: str, args: argparse.Namespace) -> str:
    if args.set:
        # Validate but keep the literal string the user asked for.
        semver.VersionInfo.parse(args.set)
        return args.set

    ver = semver.VersionInfo.parse(current)
    part = args.part
    token = args.token

    if part in ("major", "minor", "patch") and token:
        sys.exit(f"error: '{part}' takes no token (got {token!r})")

    if part == "major":
        return str(ver.bump_major())
    if part == "minor":
        return str(ver.bump_minor())
    if part == "patch":
        return str(ver.bump_patch())

    if part in ("premajor", "preminor", "prepatch"):
        return _start_prerelease_cycle(ver, part, token)
    if part == "prerelease":
        # 0.2.4-beta.1 -> 0.2.4-beta.2; 0.2.4 -> 0.2.4-rc.1
        return str(ver.bump_prerelease())
    if part == "stage":
        return _move_stage(ver, current, token)
    if part == "release":
        if not ver.prerelease:
            sys.exit(f"error: {current} is already a stable version")
        return str(ver.finalize_version())

    raise AssertionError(part)  # argparse restricts choices


# --------------------------------------------------------------------------- #
# CHANGELOG.md
# --------------------------------------------------------------------------- #
def roll_changelog(new_version: str, today: str) -> None:
    """Rename [Unreleased] to the new version and add a fresh [Unreleased].

    If there is no [Unreleased] section, require a "## [<new_version>]" section
    to already exist — refuse to tag an undocumented release.
    """
    text = CHANGELOG_FILE.read_text(encoding="utf-8")

    unreleased_re = re.compile(r"^## \[Unreleased\].*$", re.MULTILINE)
    if unreleased_re.search(text):
        fresh = f"## [Unreleased]\n\n## [{new_version}] - {today}"
        new_text = unreleased_re.sub(fresh, text, count=1)
        CHANGELOG_FILE.write_text(new_text, encoding="utf-8")
        return

    if re.search(rf"^## \[{re.escape(new_version)}\]", text, re.MULTILINE):
        # User pre-wrote the section; nothing to roll.
        return

    sys.exit(
        f"error: CHANGELOG.md has no '## [Unreleased]' section and no "
        f"'## [{new_version}]' section.\n"
        f"       Add release notes under an '## [Unreleased]' heading first."
    )


def changelog_section(version: str) -> str:
    """Return the CHANGELOG body for a version (for the dry-run preview)."""
    text = CHANGELOG_FILE.read_text(encoding="utf-8")
    m = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    return (m.group(1).strip() if m else "").strip()


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #
def run(cmd: list[str], dry_run: bool) -> None:
    print(f"  $ {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def git_output(cmd: list[str]) -> str:
    return subprocess.run(
        cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def ensure_clean_tree() -> None:
    if git_output(["git", "status", "--porcelain"]):
        sys.exit(
            "error: working tree is not clean. Commit or stash changes before "
            "releasing (the release commit must contain only the version bump)."
        )


def ensure_tag_absent(tag: str) -> None:
    existing = git_output(["git", "tag", "--list", tag])
    if existing:
        sys.exit(f"error: tag {tag} already exists.")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "part",
        nargs="?",
        choices=[
            "major",
            "minor",
            "patch",
            "premajor",
            "preminor",
            "prepatch",
            "prerelease",
            "stage",
            "release",
        ],
        help="which version part to bump (see the examples above)",
    )
    g.add_argument("--set", metavar="X.Y.Z", help="set an explicit version")
    p.add_argument(
        "token",
        nargs="?",
        help="pre-release stage (alpha, beta or rc) for premajor/preminor/prepatch/stage",
    )
    p.add_argument(
        "--push", action="store_true", help="push the commit and tag (triggers release.yml)"
    )
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--dry-run", action="store_true", help="show what would happen, change nothing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # astimezone() rather than date.today(): the changelog entry should carry
    # the release date in the releaser's own timezone, stated explicitly
    # rather than inherited from whatever the process locale happens to be.
    today = datetime.datetime.now().astimezone().date().isoformat()

    current = read_current_version()
    new_version = compute_new_version(current, args)
    tag = f"v{new_version}"
    is_prerelease = bool(semver.VersionInfo.parse(new_version).prerelease)

    if not args.dry_run:
        ensure_clean_tree()
        ensure_tag_absent(tag)

    print(
        f"Release: {current} -> {new_version}  (tag {tag}, "
        f"{'pre-release' if is_prerelease else 'stable'})"
    )
    print()

    # Apply file changes (skipped on dry-run).
    if not args.dry_run:
        write_version(new_version)
        roll_changelog(new_version, today)

    preview = (
        changelog_section(new_version) if not args.dry_run else "(dry-run: changelog not rolled)"
    )
    print("CHANGELOG section that will be the release body:")
    print("-" * 60)
    print(preview or "(empty — did you forget to write release notes?)")
    print("-" * 60)
    print()

    if not preview and not args.dry_run:
        sys.exit(
            "error: the changelog section is empty; aborting. "
            "Write notes under [Unreleased] and re-run."
        )

    # Commit + tag.
    print("Git steps:")
    run(
        [
            "git",
            "add",
            str(VERSION_FILE.relative_to(PROJECT_ROOT)),
            str(CHANGELOG_FILE.relative_to(PROJECT_ROOT)),
        ],
        args.dry_run,
    )
    run(["git", "commit", "-m", f"chore: release {tag}"], args.dry_run)
    run(["git", "tag", "-a", tag, "-m", f"Release {tag}"], args.dry_run)

    if args.push:
        if not args.yes and not args.dry_run:
            reply = input(f"\nPush commit and tag {tag} to origin? This triggers a release. [y/N] ")
            if reply.strip().lower() not in ("y", "yes"):
                print("Not pushed. The commit and tag exist locally; push manually when ready:")
                print("  git push --follow-tags origin HEAD")
                return 0
        run(["git", "push", "--follow-tags", "origin", "HEAD"], args.dry_run)
        print(f"\nPushed. release.yml will build and publish {tag}.")
    else:
        print("\nLocal commit and tag created. To trigger the release, push:")
        print("  git push --follow-tags origin HEAD")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
