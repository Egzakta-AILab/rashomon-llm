# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's **private
vulnerability reporting**: open the repository's **Security** tab and choose
**Report a vulnerability**. Do not open a public issue, pull request or
discussion for a suspected vulnerability.

Include what you can: the affected version, the steps or a minimal run file
that reproduces the issue, and the impact you expect. We aim to acknowledge a
report within five working days and will keep you informed while we work on a
fix. We credit reporters in the release notes unless you prefer otherwise.

## Scope

In scope: the `rashomon-llm` package and the scripts in this repository —
for example, a run file that makes `rashomon report` execute code, a report or
card that leaks data the run file was not meant to share, or a secret or
private address that the repository checks fail to catch.

Out of scope: the behaviour of the LLM servers you measure.

## Supported versions

Only the latest released version receives security fixes.
