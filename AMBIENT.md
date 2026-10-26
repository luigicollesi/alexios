# Alexios: modo ambiente com escuta contínua

Este modo não possui wake word. O **Alexios Python** capta o microfone continuamente (enquanto ligado), transcreve localmente com Faster-Whisper, envia somente palavras reconhecidas e seus horários ao **Alexios Agent Hub** hospedado em um Cloudflare Durable Object. O Hub consolida a janela e consulta a IA via OpenRouter para escolher um texto opcional, uma postura e um contexto substituto. O **Alexios Web** apresenta a última fala, a expressão e o estado do microfone.

## Conexão HTTPS via Vercel (produção)

O endereço oficial usado pelo Core local é **`https://alexios-web.vercel.app`**. Configure no arquivo `alexios/.env`:

```dotenv
ALEXIOS_BRIDGE_URL=https://alexios-web.vercel.app
ALEXIOS_AGENT_TOKEN=mesmo-token-seguro-configurado-na-vercel-e-no-worker
```

A branch de integração do `alexios-web` adiciona um proxy autenticado para os caminhos `https://alexios-web.vercel.app/v1/agent/poll`, `/v1/agent/state` e `/v1/transcript`. O Agent Python acessa **somente a URL pública da Vercel**. O dashboard mantém as APIs `/api/alexios/status` e `/api/alexios/power`.

O proxy da Vercel **não substitui** o backend persistente: configure `ALEXIOS_API_BASE_URL=https://SEU-WORKER.workers.dev` na Vercel com a URL HTTPS do Agent Hub publicado no Cloudflare. Configure `ALEXIOS_AGENT_TOKEN` e `ALEXIOS_DASHBOARD_TOKEN` como Environment Variables no projeto Vercel; `ALEXIOS_AGENT_TOKEN` precisa ser idêntico ao do Worker e Python. `OPENROUTER_API_KEY` fica exclusivamente no Worker. Publique a branch `feature/ambient-listening-bridge` na Vercel e o Worker na Cloudflare antes de utilizar o endereço de produção. A existência do site atual **não significa** que as novas rotas já estão publicadas.

## Instalação integrada — desenvolvimento local

1. Baixe/abra `luigicollesi/alexios-web` na branch `feature/ambient-listening-bridge`. Em `worker/`, instale com `npm install`, copie `.dev.vars.example` para `.dev.vars` e configure a **mesma** chave `ALEXIOS_AGENT_TOKEN` usada nos outros componentes; configure também a chave `OPENROUTER_API_KEY`. Inicie `npm run dev` na pasta `worker/` (porta 8787).
2. Na raiz `alexios-web/`, copie `.env.example` para `.env.local`. Configure `ALEXIOS_DASHBOARD_TOKEN` (chave diferente da do Agent, 24+ caracteres), `ALEXIOS_API_BASE_URL=http://127.0.0.1:8787` e `ALEXIOS_AGENT_TOKEN` igual à do Worker. Inicie `npm install && npm run dev` (porta 3000).
3. Abra `alexios` na branch `feature/ambient-listening`, use **Python 3.12** e instale `pip install -r requirements.txt`. Em `.env`, configure `ALEXIOS_BRIDGE_URL=http://127.0.0.1:8787` e a **mesma** chave `ALEXIOS_AGENT_TOKEN`; `OPENROUTER_API_KEY` só é necessária para o antigo `main.py`.
4. Execute `python continuous.py --model base` e autorize o painel com `ALEXIOS_DASHBOARD_TOKEN` em http://localhost:3000. O Agent inicia a escuta automaticamente na primeira conexão. O botão vermelho **FIM** interrompe a captura; o botão verde **INICIAR** retoma, sem interromper o long polling.

Para escolher microfone use `python continuous.py --list-devices` e `python continuous.py --device 2`. Use `--no-tts` para manter respostas só em texto. Em Linux, o TTS local utiliza `espeak-ng` ou `espeak` se instalados; em Windows utiliza a voz nativa SAPI via PowerShell; macOS utiliza `say`. Enquanto o Alexios fala, o reconhecimento é brevemente silenciado para evitar captar a própria voz.

## Fluxo temporal

```text
Microfone (Python)  0──────4──────7──────10──────13 s ...
                    [  janela 0  ]
                         [  janela 1  ]
                              [  janela 2  ]
                    └ 1 s de sobreposição ┘
Whisper local   →    {captureId, windowIndex, words:[{text,startMs,endMs}]}
Hub DO          →    valida sequência e elimina palavras duplicadas por timestamp
Contexto        →    análise ~8 s, com filas de texto até 420 palavras
OpenRouter      →    {speech: "" | "fala", pose:"idle|listening|thinking|speaking", context:"..."}
Painel          →    última fala + imagem associada à pose + botão de energia
Core            ←    comandos e eventos de fala no mesmo long polling (até 20 s)
```

O prompt do modelo transmite **em mensagens separadas** a descrição do Alexios, as posturas disponíveis e postura atual, o contexto anterior e a nova transcrição. Cada sucesso substitui o contexto anterior. Saídas inválidas não atualizam o contexto; a próxima janela tenta novamente. Decisões silenciosas não adicionam falas.

## Privacidade, estado e limitações

- O microfone fica permanentemente habilitado **enquanto o botão indicar execução**, exceto breves interrupções durante a reprodução local da própria fala. Todas as pessoas próximas devem saber que há uma escuta ativa.
- Nenhum arquivo de áudio cru é enviado à rede. Texto reconhecido é enviado ao Worker e em seguida a um provedor externo de IA, OpenRouter; não use em conversas confidenciais sem consentimento.
- O Durable Object armazena contexto condensado, última fala e palavras pendentes. Não implementa limpeza automática do contexto salvo após desligamento. Defina política de expiração/remoção se necessário.
- Sem acesso ao Worker por mais de 35 s, o Agent interrompe o microfone por segurança e mantém tentativas de reconexão.
- O objetivo é um ciclo de aproximadamente 8 s. Latência de Whisper, rede, limites do modelo gratuito e backpressure podem aumentar esse intervalo; **não é real time** nem garante frases perfeitas nas bordas.
- O painel usa `/characters/<pose>.png`; enquanto os quatro PNGs reais não forem adicionados ao repositório web, continuará mostrando o SVG de fallback.
- Este projeto é experimental. Testes automáticos cobrem o alinhamento das janelas; valide também o hardware e as rotas HTTP antes de utilizar continuamente.
