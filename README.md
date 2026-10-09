# Local Voice Assistant

Interface web com FastAPI, Nemotron ASR (gRPC), Qwen3 no Ollama (HTTP) e Qwen3-TTS (HTTP).

## Executar

```bash
cp .env.example .env
docker compose up -d --build
```

Acesse `http://localhost:8080` no Linux com os serviços ou `http://IP_DA_GPU:8080` em outro computador. **Para acesso ao microfone em um navegador remoto (MacBook), é necessário HTTPS** ou um acesso via `localhost` (por exemplo, tunelamento SSH):

```bash
# Executar no MacBook
ssh -L 8080:localhost:8080 usuario@ai-gpu-lab
# Navegar para http://localhost:8080
```

O Compose só sobe a interface FastAPI. Nemotron, Ollama e Qwen3-TTS continuam como serviços separados, já instalados no host de GPU. Ajuste `.env` conforme os endereços reais. Se o Ollama escutar apenas 127.0.0.1, configure um bind apropriado (`OLLAMA_HOST=0.0.0.0:11434`) **com firewall restringindo acesso à rede confiável**, ou use `network_mode: host` no Compose (Linux) e altere os endpoints para `localhost`. Se o ASR ou TTS tiverem portas publicadas apenas em 127.0.0.1, use também `network_mode: host` e remova `ports` e `extra_hosts` do Compose; com isso, `.env` deve apontar `localhost`.

## Fluxo

1. Navegador captura o microfone (AudioWorklet, PCM 16 kHz) e envia por WebSocket (`/ws/listen`) enquanto você fala; a transcrição aparece ao vivo e o envio é automático quando você para de falar (~0,8 s de silêncio).
2. FastAPI repassa o áudio ao Nemotron via Riva `StreamingRecognize`. O servidor ASR não detecta fim de fala, então o backend faz isso pela energia do áudio (com reserva pela transcrição parada).
3. A transcrição vai por WebSocket (`/ws/chat`) ao backend, que chama o Ollama em streaming com histórico das últimas 12 mensagens; o texto aparece na tela conforme é gerado.
4. A cada frase completa, o backend já chama o endpoint `/synthesize` do Qwen3-TTS (uma frase por vez, em ordem) enquanto o LLM continua gerando, e envia o WAV pelo mesmo WebSocket.
5. O navegador enfileira os áudios com Web Audio API e toca as frases em sequência, sem esperar a resposta inteira.

## Observações

- A conversa é **por turnos**: LLM e TTS são encadeados por frase, mas não há barge-in (falar por cima do assistente).
- O modelo Nemotron precisa aceitar chamadas de reconhecimento não-streaming (`Recognize`). Caso o servidor exponha somente `StreamingRecognize`, adapte o cliente ao gRPC streaming.
- É necessária permissão do microfone no navegador.
- A API não inclui autenticação: **não exponha a porta 8080 diretamente à internet**.
- O frontend foi criado sem dependências Node.js; CSS e JavaScript estão em `app/static/index.html`.
- Teste isolado: `curl http://localhost:8080/health`.
