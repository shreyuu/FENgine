import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import chess

# Load the model
model = load_model("model/model.h5")

# Updated piece mapping based on your model's output
PIECE_MAP = {
    0: None,  # Empty square
    5: None,  # Empty square (your model seems to use 5 for empty)
    1: "P",  # White pawn
    2: "N",  # White knight
    3: "B",  # White bishop
    4: "R",  # White rook
    6: "Q",  # White queen (if your model uses 6 for white queen)
    7: "K",  # White king (based on your output)
    8: "p",  # Black pawn
    9: "n",  # Black knight
    10: "b",  # Black bishop
    11: "r",  # Black rook
    12: "k",  # Black king (based on your output)
    13: "q",  # Black queen (if your model uses 13 for black queen)
}


def process_image(image_data, corrections=None):
    # Process the image to detect chess pieces
    # Returns an 8x8 array of piece labels (integers)

    # Convert image data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize and preprocess the image
    # Add your preprocessing steps here

    # For now, return a placeholder 8x8 array of zeros
    # Replace this with your actual piece detection logic
    board_labels = np.zeros((8, 8), dtype=int)

    return board_labels


def classify_cells(warped_board):
    """
    Divides a warped chessboard image into 64 cells and classifies each cell
    Returns an 8x8 array of piece labels (integers)
    """
    # Get dimensions of the warped board
    height, width = warped_board.shape[:2]

    # Calculate cell dimensions
    cell_height = height // 8
    cell_width = width // 8

    # Initialize empty 8x8 board for labels
    board_labels = np.zeros((8, 8), dtype=int)

    # For each cell on the board
    for row in range(8):
        for col in range(8):
            # Extract the cell image
            y_start = row * cell_height
            y_end = (row + 1) * cell_height
            x_start = col * cell_width
            x_end = (col + 1) * cell_width

            cell = warped_board[y_start:y_end, x_start:x_end]

            # Preprocess cell for the model
            cell = cv2.resize(cell, (64, 64))  # Resize to model input size

            # Convert to grayscale (this is the key fix)
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # Normalize
            cell_gray = cell_gray / 255.0

            # Reshape to match model's expected input shape (add channel dimension)
            cell_input = np.expand_dims(cell_gray, axis=-1)  # Shape becomes (64, 64, 1)

            # Add batch dimension
            cell_input = np.expand_dims(
                cell_input, axis=0
            )  # Shape becomes (1, 64, 64, 1)

            # Classify the cell
            prediction = model.predict(cell_input, verbose=0)
            piece_label = np.argmax(prediction)

            # Store the label
            board_labels[row][col] = piece_label

    return board_labels


def classify_cells_batch(warped_board):
    # Calculate cell dimensions
    board_height, board_width = warped_board.shape[:2]
    cell_height = board_height // 8
    cell_width = board_width // 8

    # Extract all cells first
    cells = []
    for row in range(8):
        for col in range(8):
            # Extract the cell image
            y_start = row * cell_height
            y_end = (row + 1) * cell_height
            x_start = col * cell_width
            x_end = (col + 1) * cell_width

            cell = warped_board[y_start:y_end, x_start:x_end]

            # Preprocess cell for the model
            cell = cv2.resize(cell, (64, 64))  # Resize to model input size

            # Convert to grayscale (this is the key fix)
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # Normalize
            cell_gray = cell_gray / 255.0

            # Reshape to match model's expected input shape (add channel dimension)
            cell_input = np.expand_dims(cell_gray, axis=-1)  # Shape becomes (64, 64, 1)

            cells.append(cell_input)

    # Batch all cells
    batch_input = np.vstack(cells)

    # Single prediction call
    predictions = model.predict(batch_input, verbose=0)

    # Process results
    board_labels = np.zeros((8, 8), dtype=int)
    for i, pred in enumerate(predictions):
        row, col = i // 8, i % 8
        board_labels[row][col] = np.argmax(pred)

    return board_labels


def detect_and_warp(image):
    # Find the chessboard in the image and warp it to a square
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to create a binary image
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (which should be the chessboard)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # If we have a quadrilateral (4 corners), we can warp
        if len(approx) == 4:
            # Order the points [top-left, top-right, bottom-right, bottom-left]
            ordered_points = order_points(approx.reshape(4, 2))

            # Get width and height of the warped image
            width = 800  # Desired width of the warped image
            height = 800  # Desired height of the warped image

            # Define destination points for the perspective transform
            dst = np.array(
                [
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1],
                ],
                dtype=np.float32,
            )

            # Compute the perspective transform matrix
            M = cv2.getPerspectiveTransform(ordered_points.astype(np.float32), dst)

            # Warp the image
            warped = cv2.warpPerspective(image, M, (width, height))

            return warped

    # If we reach here, we couldn't find a valid contour or it wasn't a quadrilateral
    # Resize the original image to 800x800 instead of returning None
    print("Warning: No valid chessboard contour found. Using the original image.")
    return cv2.resize(image, (800, 800))


def order_points(pts):
    """
    Orders points in [top-left, top-right, bottom-right, bottom-left] order
    """
    # Initialize an array of ordered points
    rect = np.zeros((4, 2), dtype=np.float32)

    # The top-left point will have the smallest sum of coordinates
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference
    # The bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def generate_notation(board_labels, corrections=None):
    # Convert the 8x8 array of piece labels to FEN and PGN notation

    # Apply any corrections if provided
    if corrections:
        for position, piece in corrections.items():
            file, rank = position[0], int(position[1])
            file_idx = ord(file) - ord("a")
            rank_idx = 8 - rank
            # Update the board with the corrected piece
            board_labels[rank_idx][file_idx] = piece

    # Print the board for debugging
    print("Board labels detected:")
    for row in board_labels:
        print(row)

    # Initialize empty board
    board = chess.Board(fen="8/8/8/8/8/8/8/8 w - - 0 1")

    # Place pieces on the board according to the labels
    for rank_idx, row in enumerate(board_labels):
        for file_idx, piece_label in enumerate(row):
            # Skip if empty square (0 or 5 are empty)
            if piece_label == 0 or piece_label == 5:
                continue

            # Get the chess position (e.g., 'a1', 'h8')
            file = chr(ord("a") + file_idx)
            rank = 8 - rank_idx
            position = f"{file}{rank}"

            # Get the piece from the mapping
            piece = PIECE_MAP.get(piece_label)

            if piece:
                # Set the piece on the board
                board.set_piece_at(
                    chess.parse_square(position), chess.Piece.from_symbol(piece)
                )
            else:
                # Log warning for debugging but try to interpret based on output pattern
                print(
                    f"Warning: Unknown piece label '{piece_label}' at position {position}"
                )

                # Common patterns from your terminal output:
                if piece_label == 12:  # Might be black king
                    board.set_piece_at(
                        chess.parse_square(position), chess.Piece.from_symbol("k")
                    )
                elif piece_label == 2:  # Might be white knight
                    board.set_piece_at(
                        chess.parse_square(position), chess.Piece.from_symbol("N")
                    )
                elif piece_label == 7:  # Might be white king
                    board.set_piece_at(
                        chess.parse_square(position), chess.Piece.from_symbol("K")
                    )

    # Generate FEN and PGN
    fen = board.fen()
    pgn = generate_pgn_from_fen(fen)

    print(f"Generated FEN: {fen}")
    return fen, pgn


def generate_pgn_from_fen(fen):
    # Generate a basic PGN from a FEN string
    return f"""[Event "Chess Position"]
[Site "FENgine"]
[Date "????.??.??"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "*"]
[FEN "{fen}"]
[SetUp "1"]

*"""


def check_model_input_shape():
    """
    Print the expected input shape for the model to help with debugging
    """
    # Get the input shape from the model
    input_shape = model.layers[0].input_shape
    print(f"Model expects input shape: {input_shape}")

    # Call this function when your app starts up
    return input_shape


# Add this line near the top of the file after model loading
input_shape = check_model_input_shape()
