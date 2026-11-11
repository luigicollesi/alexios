# Arquitetura do Alexios Python

## Objetivo

Manter apenas o cliente de áudio local por turnos, com a inteligência conversacional e a memória no Alexios Web. O arquivo `main.py` é o único executável principal.

## Fluxo

1. `main.py` carrega ambiente e argumentos, valida microfone, carrega os modelos e sincroniza a sequência com o servidor.
2. `AgentRuntime` recebe comandos remotos por long-polling e liga/desliga a captura.
3. `TurnAwareCapture` captura blocos de áudio; `SileroVad` identifica voz e `TurnEngine` decide quando terminar o turno com `SmartTurnDetector`.
4. `EventDispatcher` transcreve o turno com faster-whisper, identifica o locutor via CAM++ e envia os eventos em sequência. `speech_started` utiliza canal prioritário.
5. `floor_open` indica que o servidor pode considerar uma resposta após silêncio válido. `turn_end` sozinho **não autoriza fala**.
6. O Alexios Web decide a resposta, controla o avatar e produz a voz.

## Limites de responsabilidade

- Python: microfone, VAD, turnos, ASR, identificação de locutores, fila e comunicação.
- Web: personalidade, memória, decisão de resposta, avatar, TTS e persistência.

## Contrato HTTP vigente

- `/v2/agent/poll` e `/v2/agent/state`: endpoints compartilhados para presença e controle, preservados por compatibilidade.
- `/v3/agent/sync`, `/v3/agent/turn`, `/v3/agent/chunk`, `/v3/agent/floor` e `/v3/agent/activity`: protocolo de conversa.

## Propriedades importantes

- Não reter áudio de turnos antigos; preservar a correção de buffers cumulativos.
- Não deduplicar frases apenas por igualdade textual.
- Preservar `eventId` e payload durante tentativas de reenvio.
- Não permitir que `floor_open` ultrapasse transcrições pendentes.
- Invalidar oportunidades antigas quando uma nova fala começar.
- Manter filas limitadas e interromper a captura em caso de saturação.
- Os embeddings de locutor não são persistidos.

## Desenvolvimento

A suíte `unittest` é mantida para cobertura dos contratos existentes. A refatoração do cliente não deve modificar o protocolo de rede nem os thresholds dos algoritmos sem avaliação independente.
