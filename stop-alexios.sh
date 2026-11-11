#!/usr/bin/env bash

pattern='(^|/)python(3(\.[0-9]+)?)?[[:space:]]+.*continuous[.]py([[:space:]]|$)'

mapfile -t pids < <(pgrep -u "$(id -u)" -f "$pattern" || true)

if ((${#pids[@]} == 0)); then
    echo "Nenhum Alexios em execução."
    exit 0
fi

# Encerramento controlado
kill -TERM "${pids[@]}"

# Aguarda a liberação dos recursos
sleep 8

# Força somente processos que permanecerem ativos
for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
    fi
done

echo "Encerramento concluído."