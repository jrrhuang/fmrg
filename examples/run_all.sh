#!/bin/bash
set -e
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )"

bash "$SCRIPT_DIR/inverse_problems.sh"
bash "$SCRIPT_DIR/aesthetic.sh"
bash "$SCRIPT_DIR/geneval.sh"
bash "$SCRIPT_DIR/best_of_n.sh"
