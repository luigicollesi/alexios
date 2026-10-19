# Alexios

Assistente de voz em Python: capta áudio pelo microfone, transcreve localmente com Faster-Whisper e envia **somente o texto transcrito** ao NVIDIA Nemotron 3 Ultra (free), via OpenRouter. A resposta textual aparece no terminal.

## Requisitos

- Windows ou Linux x86_64 e Python **3.12** (Python 3.15 não é compatível com a dependência CTranslate2 utilizada nesta versão).
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

## Modelo de ativação "Aléxios" (openWakeWord)

O projeto agora usa [openWakeWord](https://github.com/dscripka/openWakeWord), **sem AccessKey**. Seu modelo ONNX pode ser utilizado tanto no Windows quanto no Linux.

**Importante:** a biblioteca não fornece um modelo pré-treinado para o nome "Aléxios". Você precisa **treinar ou obter** um classificador compatível e colocá-lo em:

```text
models/
  alexios.onnx
```

Consulte as instruções de [treinamento de modelos personalizados](https://github.com/dscripka/openWakeWord#training-new-models). Não basta renomear um modelo de outra palavra: o arquivo deve ter sido treinado para reconhecer "Aléxios". Sem esse arquivo, o programa mostra um erro claro e encerra, em vez de disparar por palavras incorretas.

O detector recebe áudio PCM 16-bit/16 kHz em blocos de 80 ms, calcula a probabilidade de ativação e dispara quando a pontuação for pelo menos 0,5. O reconhecimento local funciona offline depois da instalação dos modelos.

## Execução

```cmd
python main.py
```

Diga **“Aléxios”**, em seguida sua pergunta. O microfone fica ativo enquanto aguarda a palavra-chave. Após a ativação, encerra a gravação quando detectar aproximadamente 1,1 segundo de silêncio e imprime a resposta do OpenRouter. Para sair, pressione `Ctrl+C`.

Para listar ou escolher microfones:

```cmd
python main.py --list-devices
python main.py --device 2
```

Se o ambiente tiver muito ruído ou fala baixa, ajuste o limiar de volume usado para encerrar comandos com `python main.py --threshold 0.012`. Um valor menor aumenta a sensibilidade e pode dificultar detectar o silêncio. Este detector simples usa RMS, não um VAD neural.

Para trocar o modelo de transcrição local:

```cmd
python main.py --model tiny
```

## Organização

- `main.py`: todo o fluxo (wake word, captura, detecção de silêncio, transcrição, solicitação HTTP e histórico temporário).
- `context.py`: contexto base do assistente enviado como mensagem `system`.
- `.env`: chave privada da API; não versionado.
- `.env.example`: modelo sem credenciais reais.

O histórico das últimas seis interações é mantido apenas na memória do processo para contextualizar perguntas subsequentes. Não há banco de dados nem síntese de voz nesta etapa. Erros de API não são adicionados ao histórico.

O endpoint gratuito pode ter limites de requisições e políticas de tratamento de dados. Evite enviar informações pessoais ou confidenciais pela versão gratuita.

Em Linux, instale PortAudio se necessário (`sudo apt install libportaudio2`) e ative o ambiente com `source .venv/bin/activate`. O dispositivo de entrada precisa aceitar áudio mono int16 a 16 kHz. A aplicação pausa a escuta enquanto processa a pergunta; só retoma após imprimir a resposta.
