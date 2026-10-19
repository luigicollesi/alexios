# Alexios

Assistente de voz em Python. **Etapa 1:** reconhecer a fala pelo microfone e transcrevê-la localmente em português, sem chave de API.

## Requisitos
- Python 3.10 ou superior
- Microfone configurado
- Internet apenas no primeiro download do modelo Whisper
- No Linux, pode ser necessário instalar a biblioteca PortAudio do sistema

## Instalação

```bash
git clone https://github.com/luigicollesi/alexios.git
cd alexios
python -m venv .venv
```

Ative o ambiente virtual:

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
python main.py
```

Pressione **Enter** para começar a gravação, fale e pressione **Enter** novamente para transcrever. Repita ou digite `n` para sair.

## Selecionar microfone

```bash
python main.py --list-devices
python main.py --device 2
```

O dispositivo precisa suportar captura mono a 16 kHz. Se não suportar, selecione outro dispositivo.

## Modelo de transcrição

Por padrão é usado `base` em CPU/int8. Para dispositivos mais lentos:

```bash
python main.py --model tiny
```

Para maior qualidade de transcrição (com maior consumo de CPU):

```bash
python main.py --model small
```

## Escopo

Nesta etapa não há envio de áudio à IA, síntese de voz ou persistência em banco de dados. Próximas etapas: detecção automática de início/fim de fala, integração à API de IA, TTS e armazenamento assíncrono de perguntas e respostas.
