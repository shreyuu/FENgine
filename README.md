# FENgine

A Chess OCR pipeline that converts a board image into FEN and PGN notation.

---

## 1. System Architecture Overview

FENgine follows a modular, client-server design:

### 1.1 Frontend (React + TailwindCSS)

The frontend provides a seamless user experience for image upload, board review, and result display.

- **Upload UI** (`/upload`):
  - Drag-and-drop area or file picker for board photographs.
  - Instant preview of the selected image.
- **Board Review** (`/review`):
  - Canvas overlay showing the detected 8×8 grid.
  - Ability to click any square to correct misclassified pieces via a dropdown.
  - “Confirm” button to send corrections before finalization.
- **Result Display** (`/result`):
  - Shows the generated FEN string in a monospace code block.
  - Displays PGN text in a scrollable `<pre>` section.
- **Styling**
  - TailwindCSS for responsive utility classes.
  - Flex and grid layouts to align the 8×8 board neatly on all screen sizes.

---

### 1.2 Backend (Python + FastAPI)

The backend orchestrates image processing, piece classification, and notation generation.

- **Image Preprocessing & Board Detection**
  - Load image with OpenCV, convert to grayscale, and apply Gaussian blur.
  - Use adaptive thresholding and contour detection to find the board’s largest quadrilateral.
  - Compute a perspective transform to warp the board to a square image (e.g., 800×800 px).
- **Piece Classification**
  - Divide the warped board into 64 equal cells.
  - Resize each cell (e.g., 64×64 px) and run through a pretrained CNN (ResNet/VGG transfer learning).
  - Output one of 13 classes: {white pawn, white knight, …, black king, empty}.
- **Coordinate Mapping**
  - Map the 64 labels to algebraic notation (`a8` through `h1`) by row/column indices.
  - Build an 8×8 board matrix representing piece placement.
- **FEN & PGN Generation**
  - Instantiate an empty `chess.Board()` object.
  - Set each piece at its corresponding square.
  - Extract the FEN via `board.fen()`.
  - Create a `chess.pgn.Game()`, assign the setup position, and serialize to PGN.
- **REST API**
  - Single endpoint: `POST /api/ocr`
    - Accepts multipart/form-data (`file: UploadFile`)
    - Returns JSON:
      ```json
      {
        "fen": "<generated FEN>",
        "pgn": "<generated PGN>"
      }
      ```
  - Implemented with FastAPI (or Flask), served by Uvicorn (or Gunicorn).

---

### 1.3 Data & Model Training

- **Synthetic Dataset Generation**
  - Programmatically render 100K+ board positions using SVG or PIL.
  - Overlay official chess-set sprites (MERIDA, Alpha).
- **Augmentation**
  - Random rotations (±15°), brightness/contrast jitter, and occlusion patches.
- **Training Pipeline**
  - Choose a CNN backbone (ResNet-18 or MobileNetV2).
  - Train with cross-entropy loss and early stopping.
  - Export final model as `model.h5` (TensorFlow) or TorchScript.

---

### 1.4 Development Workflow & Deployment

- **Local Development**
  - Frontend: `npm run dev` (Vite or CRA).
  - Backend: `uvicorn main:app --reload`.
- **Dockerization**
  - Separate Dockerfiles for frontend (Node + Nginx) and backend (Python).
- **CI/CD**
  - GitHub Actions pipeline for linting, unit tests, and Docker image builds.
- **Hosting**
  - AWS ECS/EKS or Heroku for service containers.
  - S3 for model artifacts, served via CloudFront CDN.
