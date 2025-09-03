model="models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
outputf="qwen2.5-7b"
vllm_port="8000"


python -m vllm.entrypoints.openai.api_server --model models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 &

until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$vllm_port/health)" -eq 200 ]; do
    echo "Waiting for vLLM server to be ready on $vllm_port..."
    sleep 1
done

api_key="YOUR OPENAI KEY"

python DischargeSim.py \
    --openai_api_key $api_key \
    --doctor_llm qwen \
    --patient_llm gpt-4o-mini \
    --inf_type llm \
    --agent_dataset MIMICIV \
    --num_scenarios 10 \
    --output_file $outputf \
    --model_file $model \
    --vllm_port $vllm_port
