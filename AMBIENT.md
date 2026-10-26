# Alexios — escuta contínua sem wake word

Este modo usa **Vercel + Neon PostgreSQL**, sem Cloudflare e sem servidores adicionais. O microfone e o Whisper rodam na máquina local, mas a sessão, os comandos, o contexto e a análise são tratados pelo servidor Next.js público em **https://alexios-web.vercel.app**.

## Como funciona

```text
Alexios Python                        Next.js na Vercel                  Neon
──────────────                        ─────────────────                  ────
microfone sempre ativo (quando ON)
janelas [0–4], [3–7], [6–10]...
Whisper local → timestamps ──────────▶ POST /v1/transcript ────────────▶ palavras pendentes
                      1s overlap      deduplicação por tempo
       a cada ~8 segundos ───────────▶ POST /v1/agent/analyze ────────▶ claim atômico
                                      OpenRouter com mensagens:
                                        1. identidade Alexios
                                        2. posturas e postura atual
                                        3. contexto anterior
                                        4. nova transcrição
                                      JSON { speech, pose, context } ──▶ atualiza contexto
long polling contínuo ◀────────────── GET /v1/agent/poll  ◀──────────── comandos + fala
TTS local / desligar mic
painel ↔ /api/alexios/{status,power} ─────────────────────────────────▶ estado
```

A cada resposta válida, o `context` substitui o anterior, a `pose` troca a imagem no painel e o `speech` é exibido e falado localmente, se não estiver vazio. Quando desligado, o Core mantém o long polling ativo, mas não capta áudio. As palavras não são enviadas em forma de arquivo de áudio: somente transcrições com timestamps são transmitidas à API.

## Como executar

1. No repositório **alexios-web**, faça o deploy da branch [feature/ambient-listening-bridge](https://github.com/luigicollesi/alexios-web/tree/feature/ambient-listening-bridge) na Vercel. No projeto Vercel configure `DATABASE_URL` do Neon `long-bar-99133807`, `OPENROUTER_API_KEY`, `ALEXIOS_DASHBOARD_TOKEN` e `ALEXIOS_AGENT_TOKEN`. A migração de banco `db/migrations/001_alexios_runtime.sql` já foi aplicada à branch Neon `production` fornecida.
2. Use Python **3.12**, crie um venv e instale `pip install -r requirements.txt`.
3. Em `alexios/.env`, configure:

```dotenv
ALEXIOS_BRIDGE_URL=https://alexios-web.vercel.app
ALEXIOS_AGENT_TOKEN=mesma-chave-secreta-do-alexios-web
```

4. Inicie `python continuous.py --model base`. Depois abra https://alexios-web.vercel.app, autorize o painel com a chave de dashboard e use **FIM / INICIAR**.

**Microfone:** `python continuous.py --list-devices` para descobrir IDs; `python continuous.py --device 2` para selecionar. Use `--no-tts` se quiser somente texto na plataforma.

No desenvolvimento local, substitua o endereço por `ALEXIOS_BRIDGE_URL=http://localhost:3000`, configure `.env.local` no Next.js e rode `npm install && npm run dev`.

## Limites e segurança

- Não é necessário wake word e não existe ativação por voz. Quando ligado, informe às pessoas no ambiente que há transcrição contínua.
- O áudio cru permanece no computador. Transcrições e contexto seguem pela Vercel ao Neon e ao OpenRouter, portanto não use conversas sensíveis sem autorização.
- O TTS usa voz do sistema: SAPI no Windows, `say` no macOS e `espeak-ng`/`espeak` no Linux. A escuta pausa brevemente durante a própria fala para reduzir eco.
- O Core interrompe a gravação após aproximadamente 35 segundos sem contato com a API. A conexão continua tentando restabelecimento.
- O long polling aguarda até 20 segundos por requisição, mantendo sua lease por até 45 segundos; não precisa de um servidor permanentemente executando na nuvem.
- A IA só é chamada quando há novas palavras. As decisões ocorrem em ciclos aproximados de oito segundos, não com latência garantida, devido a rede e limites do modelo.
- O Neon mantém histórico de decisões, contexto, e estado; a política de retenção desses dados deve ser decidida antes de uso permanente.
- A plataforma ainda precisa dos quatro PNGs reais nas posturas `idle`, `listening`, `thinking`, `speaking` para substituir o SVG provisório.
