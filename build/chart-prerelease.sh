#!/bin/bash

export appVersion=$(git describe --match 'v*' --abbrev=0 HEAD --tags || echo "0.0.0")
export chartVersion=$(echo $1 | sed 's/chart-\(.*\)/\1/')

yq -i e '.version = strenv(chartVersion)' Chart.yaml
yq -i e '.appVersion = strenv(appVersion)"' Chart.yaml
yq -i e '.on.workflow_dispatch.inputs.chart_version.default = strenv(chartVersion)' ../.github/workflows/deploy.yaml
