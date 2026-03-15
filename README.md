# EchoPath Web + Backend (Local Run)

<div style="max-width: 56vh;"><div style="left: 0; width: 100%; height: 0; position: relative; padding-bottom: 177.7778%;"><iframe src="https://www.youtube.com/embed/SjxQXo5ecP4?rel=0&cc_load_policy=1" style="top: 0; left: 0; width: 100%; height: 100%; position: absolute; border: 0;" allowfullscreen scrolling="no" allow="accelerometer *; clipboard-write *; encrypted-media *; gyroscope *; picture-in-picture *; web-share *;" referrerpolicy="strict-origin"></iframe></div></div>

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

## Important note

The environment description feature requires an OpenAI-compatible backend server running a multimodal model.
