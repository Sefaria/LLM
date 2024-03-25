#!/bin/bash

export appVersion=$1

yq -i e '.on.workflow_dispatch.inputs.version.default = strenv(appVersion)' ../.github/workflows/deploy.yaml
