# EchoPath Web + Backend (Local Run)

![Watch our demo video here](https://youtube.com/shorts/SjxQXo5ecP4)

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
