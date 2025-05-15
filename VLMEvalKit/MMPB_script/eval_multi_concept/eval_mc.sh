for dir in /workspace/VLMEvalKit/eval_concept_conflict /workspace/VLMEvalKit/eval_concept_random; do
  for script in "$dir"/*.sh; do
    echo "Running $script"
    bash "$script"
  done
done