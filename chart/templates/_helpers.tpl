{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "llm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "llm.labels" -}}
helm.sh/chart: {{ include "llm.chart" . }}
{{ include "llm.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "llm.selectorLabels" -}}
app.kubernetes.io/name: llm
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

