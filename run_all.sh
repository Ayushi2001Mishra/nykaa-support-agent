#!/usr/bin/env bash
# Produces the graded transcripts. Zero API keys, zero network (after the one-time model download).
export MOCK_LLM=1 CREWAI_DISABLE_TELEMETRY=true OTEL_SDK_DISABLED=true
mkdir -p transcripts
python dataset.py        | tee transcripts/task1_dataset.txt
python orders.py         | tee transcripts/task6_orders.txt
python demo_part1.py     | tee transcripts/part1.txt
python demo_part2.py     | tee transcripts/part2.txt
python demo_part3.py     | tee transcripts/part3.txt
python demo_part4.py     | tee transcripts/part4.txt
