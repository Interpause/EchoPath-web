# EchoPath Web + Backend (Local Run)

[![demo video thumbnail](https://i3.ytimg.com/vi/SjxQXo5ecP4/hqdefault.jpg)](https://youtube.com/shorts/SjxQXo5ecP4)

Demo Video: <https://youtube.com/shorts/SjxQXo5ecP4>

## Prerequisites

- Install [Node.js](https://nodejs.org/)
- Install [Poetry](https://python-poetry.org/docs/#installation)

## Setup and run

1. If you did not clone with submodules, initialize the backend submodule:

   ```bash
   git submodule update --init --recursive
   ```

2. In the repository root, install frontend dependencies and build:

   ```bash
   npm i
   npm run build
   ```

3. Copy the generated `dist/` folder into `backend/dist/`.

   ```bash
   cp -r dist backend/dist
   ```

4. Go into the backend folder and install backend dependencies:

   ```bash
   cd backend
   poetry install
   ```

5. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

6. Start the backend development server:

   ```bash
   poe dev
   ```

The webapp will be hosted at <http://localhost:3000/index.html>.

## Important note

The environment description feature requires an OpenAI-compatible backend server running a multimodal model.

## Inspiration

EchoPath was built to address a gap in accessibility tools: many systems assume visual interaction first. This project treats audio as the primary interface for blind and low-vision users.

## What it does

EchoPath is a real-time voice-and-vision navigation assistant that:

- Streams live camera frames to the backend for perception.
- Listens for the wake phrase “hey john.”
- Captures a spoken command after wake-word detection.
- Sends command + current frame to the backend (`query_llm`).
- Speaks concise, non-visual responses from the backend (`query_llm_response`).
- Plays spatial audio cues for nearby obstacles using 3D position data.

## How it is built

- Frontend: React + TypeScript + Capacitor camera integration.
- Transport: WebSocket for continuous frame and message streaming.
- Voice loop: Browser speech recognition + speech synthesis.
- Spatial audio: Custom Web Audio directional cue engine.
- Backend: FastAPI orchestration, Hugging Face Transformers (depth), Ultralytics YOLO (detection), and llama.cpp exposed via an OpenAI-compatible API.
- Message contracts: `image`, `query_llm`, `query_llm_response`.

## Challenges

- Stable wake-word behavior in continuous recognition.
- Avoiding stale transcript and repeated-command state bugs.
- Recovery and retries after speech/WebSocket failures.
- Keeping responses actionable without visual-only language.
- Meeting real-time timing constraints across capture, network, inference, and TTS.

## Accomplishments

- End-to-end wake-word → command → backend → spoken-response loop.
- Live camera streaming integrated with backend vision + LLM querying.
- Spatial audio cues for obstacle direction and proximity.
- Improved robustness with timeout and retry safeguards.
- Hands-free, accessibility-first interaction flow.

## What we learned

- Accessibility-first design changes system architecture, not only UI text.
- Reliability and state handling are as important as model quality.
- Clear protocol contracts accelerate iteration in real-time systems.
- Helpful guidance for blind users should be concise, actionable, and sensory-aware.

## What’s next

- Stronger on-device/offline fallback for voice commands.
- Better personalization (voice style, verbosity, route preferences).
- Expanded route safety signals (surface, curb, and crossing cues).
- Confidence-aware responses when model certainty is low.
- Broader testing with blind and low-vision participants.
