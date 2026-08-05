# Security Policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose credentials, private infrastructure, personal information, or user systems.

Contact the repository maintainer privately through the contact method listed on the maintainer's GitHub profile. Include:

- A concise description of the issue
- The affected file, function, or commit
- Reproduction steps
- Potential impact
- A suggested remediation, when available

Do not include active credentials, private keys, access tokens, or sensitive captured data in the report.

## Sensitive data policy

Contributions must not contain:

- API keys, passwords, bearer tokens, cookies, or private certificates
- Public IP addresses, private hostnames, internal DNS names, or LAN endpoints
- SSH usernames, key locations, or remote-access instructions tied to a real system
- Personal file-system paths or personally identifying information
- Gameplay captures, datasets, or model weights without redistribution rights
- `.env` files, logs, caches, or generated artifacts containing local information

Use environment variables or an untracked local configuration file for endpoint and credential settings.

## Scope

Security reports are accepted for the source code and documented workflows in this repository. Third-party software, game services, model providers, and local infrastructure are outside the repository's control.

This project is provided under the Apache License 2.0 without warranties or guarantees.
