import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import chess

# Load the model
model = load_model("model/model.h5")

# Updated piece mapping - you may need to adjust these based on your model
PIECE_MAP = {
    0: None,  # Empty square
    1: "P",  # White pawn
    2: "N",  # White knight
    3: "B",  # White bishop
    4: "R",  # White rook
    5: None,  # Empty square (alternative)
    6: "Q",  # White queen
    7: "K",  # White king
    8: "p",  # Black pawn
    9: "n",  # Black knight
    10: "b",  # Black bishop
    11: "r",  # Black rook
    12: "k",  # Black king
    13: "q",  # Black queen (if your model uses 13)
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
    # Calculate cell dimensions
    board_height, board_width = warped_board.shape[:2]
    cell_height = board_height // 8
    cell_width = board_width // 8

    board_labels = np.zeros((8, 8), dtype=int)

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

            # Convert to grayscale
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # Apply histogram equalization to improve contrast
            cell_gray = cv2.equalizeHist(cell_gray)

            # Normalize to [0, 1]
            cell_gray = cell_gray.astype(np.float32) / 255.0

            # Reshape to match model's expected input shape
            cell_input = np.expand_dims(cell_gray, axis=-1)  # Shape: (64, 64, 1)
            cell_input = np.expand_dims(cell_input, axis=0)  # Shape: (1, 64, 64, 1)

            # Classify the cell
            prediction = model.predict(cell_input, verbose=0)
            piece_label = np.argmax(prediction)
            confidence = np.max(prediction)

            # Only accept predictions with high confidence
            if confidence < 0.6:  # Adjust threshold as needed
                piece_label = 0  # Treat as empty square

            # Debug: Print predictions for troubleshooting
            if piece_label != 0 and piece_label != 5:
                print(
                    f"Cell ({row}, {col}): predicted {piece_label} with confidence {confidence:.3f}"
                )

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

    # Try multiple thresholding approaches
    # Approach 1: Adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Approach 2: Otsu's thresholding
    _, thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Try both thresholds
    for thresh in [thresh1, thresh2]:
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Sort contours by area and try the largest ones
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours[:5]:  # Try top 5 largest contours
                area = cv2.contourArea(contour)

                # Filter by minimum area (adjust based on your image size)
                if area < 10000:  # Adjust this threshold
                    continue

                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:
                    # Found a quadrilateral, proceed with perspective transform
                    pts = approx.reshape(4, 2).astype(np.float32)
                    ordered_pts = order_points(pts)

                    # Define destination points for 800x800 square
                    dst = np.array(
                        [[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32
                    )

                    # Get perspective transform matrix and warp
                    matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
                    warped = cv2.warpPerspective(image, matrix, (800, 800))

                    print(f"Successfully warped board using contour with area: {area}")
                    return warped

    # If no valid contour found, try edge detection approach
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > 5000:  # Lower threshold for edge detection
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) >= 4:
                # Take first 4 points if more than 4
                pts = approx[:4].reshape(4, 2).astype(np.float32)
                ordered_pts = order_points(pts)

                dst = np.array(
                    [[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32
                )
                matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
                warped = cv2.warpPerspective(image, matrix, (800, 800))

                print(f"Successfully warped board using edges with area: {area}")
                return warped

    # If we reach here, resize the original image
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
    # Apply corrections if provided
    if corrections:
        for position, piece in corrections.items():
            file, rank = position[0], int(position[1])
            file_idx = ord(file) - ord("a")
            rank_idx = 8 - rank
            board_labels[rank_idx][file_idx] = piece

    # Initialize empty board
    board = chess.Board(fen="8/8/8/8/8/8/8/8 w - - 0 1")

    has_white_king = False
    has_black_king = False

    # Place pieces on the board
    for rank_idx, row in enumerate(board_labels):
        for file_idx, piece_label in enumerate(row):
            if piece_label == 0 or piece_label == 5:
                continue

            file = chr(ord("a") + file_idx)
            rank = 8 - rank_idx
            position = f"{file}{rank}"

            piece = PIECE_MAP.get(piece_label)

            if piece:
                board.set_piece_at(
                    chess.parse_square(position), chess.Piece.from_symbol(piece)
                )

                # Track kings
                if piece == "K":
                    has_white_king = True
                elif piece == "k":
                    has_black_king = True

    # If missing kings, add them to safe squares
    if not has_white_king:
        # Add white king to e1 if empty
        if not board.piece_at(chess.E1):
            board.set_piece_at(chess.E1, chess.Piece.from_symbol("K"))
            print("Added missing white king to e1")

    if not has_black_king:
        # Add black king to e8 if empty
        if not board.piece_at(chess.E8):
            board.set_piece_at(chess.E8, chess.Piece.from_symbol("k"))
            print("Added missing black king to e8")

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
    # Get the input shape from the model's input layer
    input_shape = model.input_shape
    print(f"Model expects input shape: {input_shape}")

    # Call this function when your app starts up
    return input_shape


# Add this line near the top of the file after model loading
input_shape = check_model_input_shape()
