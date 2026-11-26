# Security policy

robogogogo is a local research workbench. The server binds to loopback and uses
one shared workspace. It is not an authenticated multi-user or internet-facing
service. Hardware telemetry is read-only; the opt-in command backend is a test
fixture and cannot actuate hardware. Software checks do not replace a physical
robot's stop system or commissioning.

## Reporting a vulnerability

If the repository's **Security** tab offers **Report a vulnerability**, use GitHub's
private vulnerability reporting feature. Its availability depends on the
repository settings; this file does not enable that feature.

If private reporting is unavailable, open a minimal issue requesting a private
security contact, without exploit details, credentials, personal data, or sensitive
recordings. Maintainers can arrange a private channel before you share details.
Do not post an unredacted deployment configuration or an active secret publicly.

Include affected versions/commits, a minimal reproduction, expected impact, and
any relevant environment details in the private report. No response-time or
hardware-safety guarantee is made. Reproduce reports against the current main
branch where practical; there are no separately maintained security release lines.

## Maintainer release practices

Before publishing a release, enable a usable private reporting channel in GitHub
or document a verified maintainer contact. Review the staged changes for private
data, run the relevant checks, and verify the built artifacts as described in
[the release instructions](docs/development.md#prepare-a-release).
