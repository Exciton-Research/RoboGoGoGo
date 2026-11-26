# Contributing

Thanks for contributing to robogogogo. Start with the [README](README.md),
[Studio walkthrough](docs/getting-started.md), and [development guide](docs/development.md).

## Scope

Product changes target RG Mobile Duo and its versioned robot contract. Keep
simulation, measured hardware telemetry, and the command test backend distinct.
Training, learned inference, physical execution, and hardware qualification are
still integration work; do not present a numerical regression as a hardware
certificate. Use small fixtures for isolated mathematical tests.

## Issues and pull requests

For a bug, include the command or UI action that triggers it, expected and actual
behavior, OS/architecture, Python/Node versions, and a minimal reproduction. Remove
credentials, private paths, personal data, and measured deployment details before
attaching logs or recordings. Use the [security policy](SECURITY.md) for suspected
vulnerabilities.

For a pull request:

1. Install with `bash scripts/setup.sh --dev`.
2. Keep the change focused and describe the concrete resulting behavior.
3. Add regression coverage for meaningful behavior changes. Run the relevant
   Python tests, Ruff, frontend tests, formatting, type checking, and production
   build as described in the development guide. Full numerical tests can take
   several minutes due to JAX compilation and simulation.
4. Update docs when commands, controls, limitations, or data contracts change.
5. If npm dependencies change, commit the lockfile and regenerate
   `THIRD_PARTY_NOTICES.md` with `npm run notices --prefix studio`.

Do not commit local recordings, credentials, generated UI assets, dependency
directories, or downloaded vendor CAD. Preserve attribution and licensing for any
third-party material you add.

## License

The project's original code, documentation, screenshots, and robot design assets
are licensed under [Apache-2.0](LICENSE). Contributions intentionally submitted
for inclusion are provided under that license as described in its contribution
terms. Only submit material you have the right to contribute. Third-party
components retain their own licenses; see [the notices](THIRD_PARTY_NOTICES.md).
