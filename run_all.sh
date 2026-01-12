#!/bin/bash

# master script to chain download and finetune jobs
# the finetune job will only start after download completes successfully

echo "submitting download job..."
JOB1=$(sbatch --parsable download_bridge_dataset.sh)
echo "download job submitted with ID: ${JOB1}"

echo "submitting finetune job (will wait for download to complete)..."
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} test_finetune.sh)
echo "finetune job submitted with ID: ${JOB2}"

echo ""
echo "jobs submitted successfully!"
echo "download job: ${JOB1}"
echo "finetune job: ${JOB2} (depends on ${JOB1})"
echo ""
echo "monitor with: jview"
echo "or: squeue -u $USER"
