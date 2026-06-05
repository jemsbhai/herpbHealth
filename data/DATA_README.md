# Data Provenance

## Datasets

*Datasets will be documented here as they are acquired.*

### Expected datasets:
- **PubMedQA:** Biomedical question answering from PubMed abstracts
- **MedQA:** USMLE-style medical exam questions
- **MedMCQA:** Medical multiple-choice questions from Indian medical exams
- **HealthSearchQA:** Health information retrieval benchmark

## Data Integrity

All raw data files are checksummed in `checksums.sha256`. Verify with:

```powershell
Get-Content checksums.sha256 | ForEach-Object {
    $parts = $_ -split "  "
    $expected = $parts[0]
    $file = $parts[1]
    $actual = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
    if ($actual -eq $expected) { Write-Host "OK: $file" }
    else { Write-Host "MISMATCH: $file" -ForegroundColor Red }
}
```

## Collection Protocol

All datasets are publicly available research datasets downloaded from HuggingFace Hub or their official sources. No patient data is used directly. The experiments demonstrate the protocol on medical text embeddings derived from these public corpora.
