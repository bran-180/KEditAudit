# Name, license, and citation decision

Decision date: 2026-08-16

## Project and package name

The selected project name is **KEditAudit** and the Python distribution name is
`kedit-audit`. The import package remains `kedit_audit`.

The name communicates the project boundary: auditing the evidence produced by a
knowledge edit, rather than providing another catalogue of editing algorithms.

## Availability evidence

The following public checks were completed on the decision date:

| Registry | Query | Result |
|---|---|---|
| GitHub repository search API | `KEditAudit in:name` | `total_count: 0` |
| PyPI JSON API | `pypi/kedit-audit/json` | HTTP 404 |
| Crossref works API | title query `KEditAudit` | `total-results: 0` |
| General web search | exact `KEditAudit` with knowledge-editing terms | no relevant software or paper match found |

These checks reduce the risk of an obvious naming collision. They do not reserve
a GitHub or PyPI namespace and are not a trademark opinion. Repeat the checks
immediately before creating public accounts or publishing a package.

## License

KEditAudit is licensed under the Apache License 2.0. The repository contains the
complete unmodified license text in `LICENSE`, sourced from the
[Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0.txt).

Apache-2.0 was selected because it is permissive and includes an explicit patent
license for contributions. No `NOTICE` file is currently required because the
project has no additional attribution notices.

## Citation format

`CITATION.cff` is the authoritative machine-readable citation metadata. Until a
DOI-backed release exists, cite the exact repository revision used. After the
first archived release:

1. replace `0.0.0` with the released version;
2. add the release date;
3. add the repository URL;
4. add the DOI returned by the archive;
5. replace the contributor entity with named authors only after they consent to
   publishing their identities.

Do not infer personal names or email addresses from browser sessions, Git
configuration, or private conversations.
