# Alexios

## V3 experimental — fala por turnos e vários locutores

Na branch `feature/turn-aware-multispeaker-v3`, o executável
`python continuous_v3.py` adiciona Silero VAD, Smart Turn e CAM++.
Ele envia **um evento por turno**, além de sinais de abertura de fala e
atividade, sem compartilhar o histórico em memórias separadas por locutor.

A V3 é experimental e depende do protocolo `/v3` no `alexios-web`,
**ainda não publicado**. `python continuous.py` continua usando o modo V2.

Para instruções de instalação no Raspberry Pi 4, pesos ONNX, endpoints
necessários e limitações conhecidas, leia [V3.md](V3.md).

---


Assistente local com captura de áudio e transcrição em português. O **modo contínuo V2**
usa somente a API de produção do [Alexios Web](https://alexios-web.vercel.app);
a IA, o contexto e a personagem ficam no servidor (Vercel + Neon).

## Modo contínuo — integração V2

- Microfone local a 16 kHz; janelas de 4 segundos a cada 3 segundos.
- Whisper local com timestamps; envia somente palavras ao servidor.
- Long polling exclusivo para **presença** e **comandos de captura**.
- **FIM** desliga o microfone, mas mantém o long polling.
- **INICIAR** retoma a captura com novo `captureId`.
- IA, respostas e troca de postura aparecem **somente na web**.
- **Ctrl+C** ou SIGTERM encerra também o polling e libera os recursos locais.
- Reconexão progressiva, upload com retries da mesma janela e memória limitada.

### Instalação (Python 3.12)

```bash
git clone https://github.com/luigicollesi/alexios.git
cd alexios
git switch feature/web-v2-client
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edite o `.env`:

```dotenv
ALEXIOS_BRIDGE_URL=https://alexios-web.vercel.app
ALEXIOS_AGENT_TOKEN=TOKEN_PRIVADO_IGUAL_AO_DA_VERCEL
ALEXIOS_INSTANCE_ID=alx_IDENTIFICADOR_FIXO_GERADO_ALEATORIAMENTE
```

Para gerar o identificador **uma vez**:
```bash
printf 'alx_%s\n' "$(openssl rand -hex 24)"
```

Use o mesmo ID no botão **AUTORIZAR ACESSO** do painel. Trate-o como senha:
quem o conhece consegue autorizar o acesso àquela instância.

### Execução

```bash
python continuous.py --list-devices
python continuous.py --device 10 --model base
```

Para usar o microfone padrão, omita `--device`. O `--no-tts` antigo continua
aceito, mas não executa nenhuma ação, pois o modo V2 não possui TTS local.

Confira [AMBIENT.md](AMBIENT.md) para os contratos HTTP, fluxos de
reconexão, estados e diagnósticos.

## Modo interativo legado

O `main.py` continua disponível e **não faz parte do modo V2**. Ele capta
manualmente entre dois Enter e consulta o OpenRouter pelo próprio Python.

Para esse modo legado, configure `OPENROUTER_API_KEY` e execute:

```bash
python main.py --list-devices
python main.py --device 2
```

## Testes automatizados

```bash
python -m unittest discover -s tests -v
python -m compileall -q continuous.py audio_windows.py agent tests
```

Nunca suba o arquivo `.env` para o GitHub. O áudio cru permanece no computador,
mas as transcrições seguem para a Vercel, Neon e OpenRouter; informe as pessoas
presentes antes de ativar a captura contínua.
