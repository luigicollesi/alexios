# Alexios Python

Cliente local do assistente de voz Alexios. **Um único modo de execução**: captura o microfone, identifica turnos e locutores, transcreve em português e envia eventos ao [Alexios Web](https://github.com/luigicollesi/alexios-web), onde ficam a IA, a memória e a síntese de voz.

## Requisitos

Python 3.12, microfone compatível com PortAudio e acesso ao servidor Alexios Web.

```bash
git clone https://github.com/luigicollesi/alexios.git
cd alexios
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_v3_models.py
cp .env.example .env
```

Configure `ALEXIOS_BRIDGE_URL`, `ALEXIOS_AGENT_TOKEN` e `ALEXIOS_INSTANCE_ID` no arquivo `.env`. **Nunca publique credenciais**.

## Execução

```bash
python main.py --list-devices
python main.py --mic-test 5
python main.py --model base --model-threads 1
# Alternativa para testar qualidade de transcrição:
python main.py --model small --model-threads 1
```

Use `--device N` para selecionar um microfone e `--debug-audio` para diagnósticos. Consulte `python main.py --help` para os demais parâmetros.

## Arquitetura

- `main.py`: ponto único de entrada.
- `agent/config.py`: configuração e validação de conexão.
- `agent/runtime.py`: long-polling de presença e controle do microfone.
- `agent/bridge.py`: transporte HTTP e comandos de presença.
- `agent/v3/capture.py`: captura de áudio e processamento fora do callback.
- `agent/v3/turns.py`: turnos e oportunidades de resposta.
- `agent/v3/models.py`: Silero VAD, Smart Turn e CAM++.
- `agent/v3/transcription.py`: faster-whisper e contexto de vocabulário.
- `agent/v3/dispatch.py`: fila serial, transcrição, identificação e envio de eventos.
- `agent/v3/bridge.py`: API de eventos do servidor.

Os caminhos HTTP `/v2/agent/poll` e `/v2/agent/state` permanecem por compatibilidade do servidor; **não representam uma segunda versão do cliente**. Os eventos da conversa usam `/v3/agent/*`.

Detalhes em [docs/architecture.md](docs/architecture.md).

## Testes

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py agent scripts tests
```

O áudio bruto e os embeddings de locutores permanecem locais. O texto transcrito e os identificadores de locutor são enviados ao servidor; informe as pessoas presentes antes da escuta contínua.
