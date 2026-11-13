# Publishing the plugin

Published: [livekit-plugins-realtimevad 0.1.0](https://pypi.org/project/livekit-plugins-realtimevad/0.1.0/)
on 2026-11-13, built from commit 4e1fafc, after `realtimevad` 0.1.0 ([#5](https://github.com/KoljaB/RealtimeVAD/issues/5)).
This is how a release of the plugin is built, checked and uploaded.

## Build and check

From this folder, in any environment with `build` and `twine`:

```bash
python -m build                 # dist/livekit_plugins_realtimevad-<version>.tar.gz and -py3-none-any.whl
python -m twine check dist/*
```

Then in a fresh virtual environment, against `realtimevad` from PyPI:

```bash
pip install dist/livekit_plugins_realtimevad-<version>-py3-none-any.whl pytest pytest-asyncio
pytest -o asyncio_mode=auto tests   # from a folder that holds only tests/ and no livekit/ directory
```

`-o asyncio_mode=auto` stands in for the setting in `pyproject.toml`, which that folder does not
have; without it every test fails with "async def functions are not natively supported".
`twine check` needs twine 7 or newer: hatchling writes metadata version 2.5, which twine 6 rejects.
For 0.1.0 this gave 29 passed against `realtimevad` 0.1.0 from PyPI.

The wheel holds `livekit/plugins/realtimevad/` (six files) and the license. The sdist adds
`tests/`, `README.md` and `pyproject.toml`. The benchmarks are not packaged: they need the
repository's test audio and a checkout of livekit/agents.

The version comes from `livekit/plugins/realtimevad/version.py`. 0.1.0 was the first upload, the
same number as the RealtimeVAD release it needs (`realtimevad>=0.1.0`). The two are versioned
separately after that. When a plugin release needs a new core, upload the core first: the
fresh-environment check above installs it from PyPI.

Before an upload:

- `dist/` in this folder may hold builds of an older version (it is git-ignored): delete it first,
  or build into a fresh folder (`python -m build --outdir <folder>`).
- The README's links to `benchmarks/`, `examples/` and `PAUSE.md` are absolute links to the `main`
  branch on GitHub, because PyPI does not resolve relative ones. Push `main` first.

Then `twine upload <folder>/*`, and compare the SHA-256 values PyPI shows with the local files.

## The name

The distribution is called `livekit-plugins-realtimevad`, decided by the author before the first
upload. It follows LiveKit's own naming (`livekit-plugins-silero`, ...), so people find it where
they look, and it can be taken for an official plugin: the package description and the README
say it is a community plugin, not made by LiveKit. The alternative was `realtimevad-livekit`.
The import path is `livekit.plugins.realtimevad`, because that is where LiveKit plugins live and
what `from livekit.plugins import realtimevad` needs.

A name on PyPI cannot be reused for something else, and a renamed project needs a stub release
under the old name to redirect people.
