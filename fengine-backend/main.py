from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from processing import classify_cells, generate_notation
import numpy as np
import cv2
import traceback
import time

app = FastAPI()

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the detect_and_warp function here if it's not in processing.py
def detect_and_warp(image):
    # Implementation as shown in Option 1
    pass


@app.post("/api/ocr")
async def process_image(file: UploadFile = File(...), corrections: str = None):
    try:
        # Read the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Process the image
        try:
            start_time = time.time()
            warped = detect_and_warp(img)
            warp_time = time.time() - start_time

            # Ensure warped is not None
            if warped is None:
                print("Warning: detect_and_warp returned None. Using original image.")
                warped = cv2.resize(img, (800, 800))

            # Save the warped image for debugging (optional)
            # cv2.imwrite("debug_warped.jpg", warped)

            start_time = time.time()
            board_labels = classify_cells(warped)
            classify_time = time.time() - start_time

            print(f"Time to warp: {warp_time:.2f} seconds")
            print(f"Time to classify: {classify_time:.2f} seconds")

            # Debug print the board
            print("Detected board before corrections:")
            for row in board_labels:
                print(row)

            # Apply any corrections if provided
            if corrections:
                import json

                corrections_dict = json.loads(corrections)
                for position, piece in corrections_dict.items():
                    # Expecting positions like "a1", "h8", etc.
                    col = ord(position[0]) - ord("a")
                    row = 8 - int(position[1])
                    if 0 <= row < 8 and 0 <= col < 8:
                        board_labels[row][col] = piece

                # Debug print the board after corrections
                print("Board after corrections:")
                for row in board_labels:
                    print(row)

            # Generate FEN and PGN
            fen, pgn = generate_notation(board_labels)

            return {"fen": fen, "pgn": pgn}
        except Exception as e:
            print(f"Error during image processing: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Image processing error: {str(e)}"
            )
    except Exception as e:
        # Log the full traceback
        error_details = traceback.format_exc()
        print(f"Error processing image: {error_details}")

        # Return a more helpful error message
        raise HTTPException(
            status_code=500, detail=f"Failed to process image: {str(e)}"
        )
