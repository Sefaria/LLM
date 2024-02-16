#!/bin/bash

export appVersion=$(git describe --match 'v*' --abbrev=0 HEAD --tags)
export chartVersion=$1

./yq -i e '.version = strenv(chartVersion)' chart/Chart.yaml
./yq -i e '.appVersion = strenv(appVersion)"' chart/Chart.yaml
./yq -i e '.on.workflow_dispatch.inputs.chart_version.default = strenv(chartVersion)' .github/workflows/deploy.yaml
