#!/bin/bash
# Hook: Block Claude from auto-merging PRs.
# PRs must be merged by the user manually after review.

input=$(cat)
command=$(echo "$input" | python -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)

if echo "$command" | grep -qE 'gh\s+pr\s+merge'; then
  echo '{"error": "Auto-merge is blocked. PRs must be merged by the user manually after review."}'
  exit 2
fi

exit 0
