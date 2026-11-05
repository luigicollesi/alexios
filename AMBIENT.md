# Alexios V2 — escuta contínua e IA no Alexios Web

A branch `feature/web-v2-client` conecta o Python diretamente à produção de
**https://alexios-web.vercel.app**. Usa somente Vercel e Neon;
nenhum servidor adicional, Cloudflare ou wake word.

## Arquitetura

| Python local | Alexios Web (Vercel + Neon) |
|---|---|
| Microfone 16 kHz e Whisper local | Validação de identidade e sessão por instância |
| Janelas 4s, deslocamento 3s, sobreposição 1s | Deduplicação das janelas |
| POST /v2/transcript (somente palavras com timestamps) | Vercel Queues -> OpenRouter -> contexto, fala e postura |
| POST /v2/agent/poll (presença, ligar/desligar) | Painel mostra texto da IA e altera a personagem |
| POST /v2/agent/state (confirmação de microfone) | Histórico e estado no Neon |

O long polling **não envia nem recebe falas da IA**. Serve apenas para presença
e controle do microfone. Quando o painel envia FIM, o microfone fecha, mas a
conexão do agente continua ativa. INICIAR reabre a captura em sequência nova.

## Instalação no Ubuntu (Python 3.12)

```bash
git switch feature/web-v2-client
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edite `.env`:

```dotenv
ALEXIOS_BRIDGE_URL=https://alexios-web.vercel.app
ALEXIOS_AGENT_TOKEN=TOKEN_QUE_JA_EXISTE_NA_VERCEL
ALEXIOS_INSTANCE_ID=alx_SEU_IDENTIFICADOR_ALEATORIO_FIXO
```

Gere um identificador **uma única vez** com `printf 'alx_%s\n' "$(openssl rand -hex 24)"`.
Não compartilhe o ID: ele concede acesso ao painel.

```bash
python continuous.py --list-devices
python continuous.py --model base
python continuous.py --device 10 --model base
```

O parâmetro antigo `--no-tts` ainda é aceito por compatibilidade, mas **não existe
TTS local** na V2. A chave `OPENROUTER_API_KEY` não é usada por `continuous.py`.

Após iniciar o Python, abra https://alexios-web.vercel.app, clique
**AUTORIZAR ACESSO** e informe o mesmo `ALEXIOS_INSTANCE_ID`.

## Comportamentos e diagnósticos

- **Online/escutando:** long polling ativo, microfone aberto, transcrições enviadas.
- **Online/microfone desligado:** o polling continua ativo e aguarda INICIAR.
- **Offline:** ao parar o polling, o servidor expira a presença em aproximadamente 45s.
- **Sem rede por 35s:** o Python suspende a captação e tenta reconectar (até 10s de backoff).
- **Conflito HTTP 409 no poll:** outro processo possui a mesma instância; o cliente
  espera a expiração do lease anterior. Não deve criar automaticamente outro ID.
- **HTTP 401/403:** credencial inválida; execução para com mensagem de erro.
- **Envio HTTP falhou:** mesma janela (`captureId` e `windowIndex`) é repetida
  sem criar outra sequência lógica.
- **FIM:** a fila local é descartada e o polling permanece ativo.
- **Ctrl+C/SIGTERM:** encerra o microfone, os trabalhadores e o long polling.
- **Resposta da IA:** aparece **somente no painel web**, não no terminal Python.

Ao testar pela primeira vez, confirme no Neon a existência da instância e
do processamento de inferências. O status READY da Vercel não prova que o
OpenRouter ou a fila estão funcionando.

## Cuidados de segurança

O áudio cru permanece no computador; as **transcrições** são encaminhadas para
Vercel, Neon e OpenRouter. Informe as pessoas próximas sobre a escuta contínua
e evite capturar conversas sem autorização.

O `main.py` legado continua separado e ainda pode usar OpenRouter localmente
quando executado manualmente. O modo V2 usa apenas `continuous.py`.
