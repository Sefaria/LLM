#!/bin/bash

cat << EOF > chart/.releaserc
tagFormat: chart-\${version}
plugins:
  - - "@semantic-release/commit-analyzer"
    - preset: "conventionalcommits"
      releaseRules:
        - {"type": "feat", "release": "minor"}
        - {"type": "fix", "release": "patch"}
        - {"type": "chore", "release": "patch"}
        - {"type": "docs", "release": "patch"}
        - {"type": "style", "release": "patch"}
        - {"type": "refactor", "release": "patch"}
        - {"type": "perf", "release": "patch"}
        - {"type": "test", "release": "patch"}
        - {"type": "static", "release": "patch"}
      parserOpts:
        noteKeywords:
          - MAJOR RELEASE
  - - "@semantic-release/release-notes-generator"
    - preset: "conventionalcommits"
      presetConfig:
        "types":
          - {"type": "feat", "hidden": true}
          - {"type": "fix", "hidden": true}
          - {"type": "chore", "hidden": true}
          - {"type": "docs", "hidden": true}
          - {"type": "style", "hidden": true}
          - {"type": "refactor", "hidden": true}
          - {"type": "perf", "hidden": true}
          - {"type": "test", "hidden": true}
          - {"type": "static", "hidden": true}
EOF
export branch=$(git branch --show-current)
export channel=$(echo $branch | awk '{print tolower($0)}' | sed 's|.*/\([^/]*\)/.*|\1|; t; s|.*|\0|' | sed 's/[^a-z0-9\.\-]//g')
if [[ $branch != "main" ]]; then
cat << EOF >> chart/.releaserc
branches: [
    {"name": "main"},
    {"name": "${branch}", "prerelease": "$channel"}
  ]
EOF
else
# Only create github release and update workflows for full release
cat << EOF >> chart/.releaserc
  - - "@semantic-release/github"
    - "successComment": false
  - - "@semantic-release/exec"
    - "prepareCmd": "../build/chart-prerelease.sh \${nextRelease.gitTag}"
  - - "@semantic-release/git"
    - assets:
        - Chart.yaml
        - ../.github/workflows/deploy.yaml
branches: [
    {"name": "main"}
  ]
EOF
fi
