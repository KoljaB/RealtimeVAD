# Publishing the plugin

Not published yet ([#5](https://github.com/KoljaB/RealtimeVAD/issues/5)). This is what was checked and what is left to decide.

## Build and check

From this folder, in any environment with `build` and `twine`:

```bash
python -m build                 # dist/livekit_plugins_realtimevad-<version>.tar.gz and -py3-none-any.whl
python -m twine check dist/*
```

Then in a fresh virtual environment, against `realtimevad` from PyPI:

```bash
pip install dist/livekit_plugins_realtimevad-<version>-py3-none-any.whl pytest pytest-asyncio
pytest tests                    # from a folder that holds only tests/ and no livekit/ directory
```

The wheel holds `livekit/plugins/realtimevad/` (five files) and the license. The sdist adds
`tests/`, `README.md` and `pyproject.toml`. The benchmarks are not packaged: they need the
repository's test audio and a checkout of livekit/agents.

The version comes from `livekit/plugins/realtimevad/version.py`. It still says 0.0.5, the version of
RealtimeVAD the plugin first shipped beside; `update_options()` and `set_agent_speaking()` came
after the 0.0.5 tag, so pick the version for the first upload before building it.

Upload (`twine upload dist/*`) is the one step not done.

## The name

`livekit-plugins-realtimevad` follows LiveKit's own naming (`livekit-plugins-silero`, ...) and can
be taken for an official plugin. `realtimevad-livekit` cannot. Both names were free on PyPI on
2026-11-13. Only the distribution name is in question; the import path stays
`livekit.plugins.realtimevad` either way, because that is where LiveKit plugins live and what
`from livekit.plugins import realtimevad` needs.

A rename to `realtimevad-livekit` changes:

| What | Now | After |
|---|---|---|
| `pyproject.toml`, `[project] name` | `livekit-plugins-realtimevad` | `realtimevad-livekit` |
| Install command (this README, later the main README) | `pip install livekit-plugins-realtimevad` | `pip install realtimevad-livekit` |
| File names in `dist/` | `livekit_plugins_realtimevad-0.0.5*` | `realtimevad_livekit-0.0.5*` |
| What `pip show`, `pip list` and `importlib.metadata` call it | `livekit-plugins-realtimevad` | `realtimevad-livekit` |
| The text of issue #5 | names the first | names the second |

It does not change:

- the import: `from livekit.plugins import realtimevad`, and `[tool.hatch.build.targets.wheel] packages = ["livekit"]`
- the plugin registration. `Plugin(__name__, __version__, __package__, logger)` passes the import
  path `livekit.plugins.realtimevad`; LiveKit uses it to preload the module in its worker
  processes and in log lines, never to look up the distribution (livekit-agents 1.8.11,
  `worker.py`, `__main__.py`)
- the logger name, the tests, the benchmarks
- the folder `livekit-plugins-realtimevad/` in this repository. It can keep its name; if it is
  renamed too, the links in the main README (two) and `Documentation` in `pyproject.toml` follow.

Things a different distribution name brings with it:

- People who know LiveKit's plugins will try `pip install livekit-plugins-realtimevad` first. The
  README's first lines have to carry the install command.
- `pip install "livekit-agents[silero]"`-style extras exist only for LiveKit's own plugins, under
  either name.
- Once a name is on PyPI it cannot be reused for something else, and a renamed project needs a
  stub release under the old name to redirect people. Deciding before the first upload is free.
