FROM whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3

RUN export NVTE_FRAMEWORK=pytorch && pip3 install boto3