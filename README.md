# Alexios

Assistente de voz em Python: capta áudio pelo microfone, transcreve localmente com Faster-Whisper e envia **somente o texto transcrito** ao NVIDIA Nemotron 3 Ultra (free), via OpenRouter. A resposta textual aparece no terminal.

## Requisitos

- Windows 64 bits e Python **3.12** (Python 3.15 não é compatível com a dependência CTranslate2 utilizada nesta versão).
- Microfone e acesso à internet para a API OpenRouter.
- Chave de API criada em https://openrouter.ai/settings/keys

## Instalação (CMD do Windows)

```cmd
git clone https://github.com/luigicollesi/alexios.git
cd alexios
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Se já clonou o repositório, use `git pull origin main` em vez de `git clone`.

Crie um arquivo `.env` na raiz do projeto, copiando o exemplo `.env.example`:

```env
OPENROUTER_API_KEY=sk-or-v1-sua-chave-aqui
```

Nunca envie a chave ao GitHub: `.env` está incluído no `.gitignore`.

## Execução

```cmd
python main.py
```

Pressione Enter para começar a gravação, fale e pressione Enter novamente para parar. A transcrição será enviada ao modelo e a resposta será impressa no terminal. Pressione Enter para gravar outra pergunta ou digite `n` para sair.

Para listar ou escolher microfones:

```cmd
python main.py --list-devices
python main.py --device 2
```

Para trocar o modelo de transcrição local:

```cmd
python main.py --model tiny
```

## Organização

- `main.py`: todo o fluxo (microfone, transcrição, solicitação HTTP e histórico temporário).
- `context.py`: contexto base do assistente enviado como mensagem `system`.
- `.env`: chave privada da API; não versionado.
- `.env.example`: modelo sem credenciais reais.

O histórico das últimas seis interações é mantido apenas na memória do processo para contextualizar perguntas subsequentes. Não há banco de dados nem síntese de voz nesta etapa. Erros de API não são adicionados ao histórico.

O endpoint gratuito pode ter limites de requisições e políticas de tratamento de dados. Evite enviar informações pessoais ou confidenciais pela versão gratuita.
