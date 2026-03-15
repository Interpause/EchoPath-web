# EchoPath Web + Backend (Local Run)

<iframe
  width="360"
  height="640"
  src="https://www.youtube.com/embed/SjxQXo5ecP4"
  title="YouTube video player"
  frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  referrerpolicy="strict-origin-when-cross-origin"
  allowfullscreen>
</iframe>

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
