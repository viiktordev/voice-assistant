# Local Voice Assistant

Web voice assistant built with FastAPI, Nemotron ASR (Riva gRPC), Qwen3 on Ollama (HTTP) and Qwen3-TTS (HTTP). Speech recognition, the LLM response and speech synthesis all stream, so the assistant starts talking before the full answer is ready.

## Running

```bash
cp .env.example .env
docker compose up -d --build
```

Open `http://localhost:8080` on the Linux host running the services, or `http://GPU_HOST_IP:8080` from another machine. **Browsers only allow microphone access over HTTPS or `localhost`**, so from a remote machine (e.g. a MacBook) use an SSH tunnel:

```bash
# On the MacBook
ssh -L 8080:localhost:8080 user@ai-gpu-lab
# Then browse to http://localhost:8080
```

Compose only starts the FastAPI app. Nemotron, Ollama and Qwen3-TTS run as separate services on the GPU host; set their addresses in `.env`. If Ollama only listens on 127.0.0.1, bind it to the network (`OLLAMA_HOST=0.0.0.0:11434`) **behind a firewall that restricts access to your trusted network**, or use `network_mode: host` in Compose (Linux) and point the endpoints to `localhost`. If the ASR or TTS ports are only published on 127.0.0.1, also use `network_mode: host`, remove `ports` and `extra_hosts` from Compose and point `.env` to `localhost`.

The TTS server must expose `POST /synthesize/stream`, returning raw 16-bit mono PCM in chunks as it is generated, with the sample rate in the `X-Sample-Rate` header (in addition to the regular `/synthesize`, which returns a full WAV).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ASR_SERVER` | `host.docker.internal:50051` | Riva/Nemotron gRPC address |
| `ASR_LANGUAGE` | `pt-BR` | Recognition language |
| `ASR_END_SILENCE_MS` | `800` | Silence that ends an utterance |
| `ASR_VAD_MIN_RMS` | `50` | Minimum energy (16-bit PCM) counted as voice; above it, 3x the measured noise floor is used |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama address |
| `OLLAMA_MODEL` | `qwen3:8b` | Model used for answers (thinking disabled for latency) |
| `TTS_URL` | `http://host.docker.internal:8000/synthesize` | Full-WAV synthesis endpoint |
| `TTS_STREAM_URL` | `TTS_URL` + `/stream` | Streaming synthesis endpoint |
| `TTS_LANGUAGE` / `TTS_SPEAKER` | `Portuguese` / `Ryan` | Qwen3-TTS voice |
| `SYSTEM_PROMPT` | short Portuguese prompt | LLM system prompt |

## How it works

1. The browser captures the microphone (AudioWorklet, 16 kHz PCM) and sends it over a WebSocket (`/ws/listen`) while you speak. The transcript appears live, and the message is sent automatically once you stop talking.
2. The backend forwards the audio to Nemotron through Riva `StreamingRecognize`. The ASR server does not detect the end of speech, so the backend does it using audio energy (with a fallback on the transcript going unchanged).
3. The transcript goes over a WebSocket (`/ws/chat`) to the backend, which streams from Ollama with the last 12 messages of history; the text shows up as it is generated.
4. As soon as each sentence is complete, the backend sends it to `/synthesize/stream` (one sentence at a time, in order) while the LLM keeps generating, and relays the PCM chunks over the same WebSocket as they arrive.
5. The browser schedules the chunks back to back with the Web Audio API, without waiting for the full answer.
6. In **continuous conversation** mode (on by default), the microphone reopens once the answer finishes playing. The conversation pauses after ~8 s of silence; click **Falar** again to resume.

## Notes

- Conversation is **turn-based**: there is no barge-in (talking over the assistant), since the microphone only reopens after playback ends.
- Microphone permission is required in the browser.
- The API has no authentication: **do not expose port 8080 directly to the internet**.
- The frontend has no Node.js dependencies; CSS and JavaScript live in `app/static/index.html`.
- `scripts/check_asr.py` checks which recognition modes the ASR server supports: `docker exec -i voice-assistant python - < scripts/check_asr.py`.
- Quick check: `curl http://localhost:8080/health`.
