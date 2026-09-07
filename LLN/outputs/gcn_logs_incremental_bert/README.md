# LLN LightGCN Outputs

This directory stores the trained LLN/LightGCN graph outputs used by MMOE.

The original output directory was compressed into `gcn_logs_incremental_bert.zip` and split into multiple parts to keep every GitHub file below the 100 MB single-file limit.

## Contents

- `gcn_logs_incremental_bert.zip.part001` ... `gcn_logs_incremental_bert.zip.part009`: split archive parts for the trained graph outputs.

After reconstruction, the archive contains scene-level LightGCN outputs, including user/item graph embeddings and trained model checkpoints. These outputs can be used directly as graph embedding inputs for MMOE training.

## Reconstruct on Windows PowerShell

Run this command inside this directory:

```powershell
Get-Content -Encoding Byte -Path .\gcn_logs_incremental_bert.zip.part* | Set-Content -Encoding Byte .\gcn_logs_incremental_bert.zip
Expand-Archive .\gcn_logs_incremental_bert.zip -DestinationPath .\gcn_logs_incremental_bert
```

## Reconstruct on Linux/macOS

Run this command inside this directory:

```bash
cat gcn_logs_incremental_bert.zip.part* > gcn_logs_incremental_bert.zip
unzip gcn_logs_incremental_bert.zip -d gcn_logs_incremental_bert
```
