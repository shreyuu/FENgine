import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import chess
import time

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
    board_height, board_width = warped_board.shape[:2]
    cell_height = board_height // 8
    cell_width = board_width // 8

    board_labels = np.zeros((8, 8), dtype=int)

    print(f"Board shape: {warped_board.shape}")
    print(f"Cell dimensions: {cell_width}x{cell_height}")

    for row in range(8):
        for col in range(8):
            y_start = row * cell_height
            y_end = (row + 1) * cell_height
            x_start = col * cell_width
            x_end = (col + 1) * cell_width

            cell = warped_board[y_start:y_end, x_start:x_end]

            # Save a few sample cells for debugging
            if row < 2 and col < 2:
                cv2.imwrite(f"debug_cell_{row}_{col}_original.jpg", cell)

            # Preprocess cell
            cell_resized = cv2.resize(cell, (64, 64))

            # Try different preprocessing approaches
            # Approach 1: RGB preprocessing
            cell_rgb = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2RGB)
            cell_rgb_normalized = cell_rgb.astype(np.float32) / 255.0
            cell_rgb_input = np.expand_dims(cell_rgb_normalized, axis=0)

            # Approach 2: Grayscale preprocessing
            cell_gray = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2GRAY)
            cell_gray_normalized = cell_gray.astype(np.float32) / 255.0
            cell_gray_input = np.expand_dims(
                np.expand_dims(cell_gray_normalized, axis=-1), axis=0
            )

            # Save preprocessed samples for debugging
            if row < 2 and col < 2:
                cv2.imwrite(
                    f"debug_cell_{row}_{col}_processed_gray.jpg",
                    (cell_gray_normalized * 255).astype(np.uint8),
                )
                cv2.imwrite(
                    f"debug_cell_{row}_{col}_processed_rgb.jpg",
                    (cell_rgb_normalized * 255).astype(np.uint8),
                )

            # Try both preprocessing approaches
            for approach, cell_input in [
                ("RGB", cell_rgb_input),
                ("Grayscale", cell_gray_input),
            ]:
                try:
                    prediction = model.predict(cell_input, verbose=0)
                    piece_label = np.argmax(prediction)
                    confidence = np.max(prediction)

                    # Debug output for first few cells
                    if row < 2 and col < 2:
                        print(f"Cell ({row}, {col}) - {approach}:")
                        print(f"  Input shape: {cell_input.shape}")
                        print(f"  Prediction: {prediction}")
                        print(f"  Label: {piece_label}, Confidence: {confidence:.3f}")
                        print(
                            f"  Top 3 predictions: {np.argsort(prediction[0])[-3:][::-1]}"
                        )

                    # Use the approach that gives higher confidence
                    if approach == "RGB" or (
                        approach == "Grayscale" and confidence > 0.5
                    ):
                        board_labels[row][col] = piece_label if confidence > 0.3 else 0
                        break

                except Exception as e:
                    print(f"Error predicting cell ({row}, {col}) with {approach}: {e}")
                    board_labels[row][col] = 0

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


def debug_model_info():
    """Debug function to check model input/output specifications"""
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)

    # Test with a dummy input
    if len(model.input_shape) == 4:  # (batch, height, width, channels)
        dummy_shape = (
            1,
            model.input_shape[1],
            model.input_shape[2],
            model.input_shape[3],
        )
    else:
        dummy_shape = model.input_shape

    dummy_input = np.random.random(dummy_shape).astype(np.float32)
    try:
        dummy_output = model.predict(dummy_input, verbose=0)
        print("Dummy prediction shape:", dummy_output.shape)
        print("Dummy prediction:", dummy_output)
        print(
            "Number of classes:",
            dummy_output.shape[-1] if len(dummy_output.shape) > 1 else 1,
        )
    except Exception as e:
        print("Error with dummy prediction:", e)


def process_chess_board(image):
    start_time = time.time()

    # Debug model info first
    debug_model_info()

    # Detect and warp the board
    warped = detect_and_warp(image)
    warp_time = time.time()

    if warped is None:
        print("Warning: detect_and_warp returned None. Using original image.")
        warped = cv2.resize(image, (800, 800))

    print(f"Time to warp: {warp_time - start_time:.2f} seconds")

    # Save the warped image for debugging
    cv2.imwrite("debug_warped_board.jpg", warped)

    # Classify cells
    try:
        board_labels = classify_cells(warped)
    except Exception as e:
        print(f"Error in model classification: {e}")
        print("Falling back to batch classification...")
        board_labels = classify_cells_batch(warped)

    classify_time = time.time()
    print(f"Time to classify: {classify_time - warp_time:.2f} seconds")

    print("Detected board before corrections:")
    print(board_labels)

    # Generate notation
    fen, pgn = generate_notation(board_labels)

    return fen, pgn


# Add this line near the top of the file after model loading
input_shape = check_model_input_shape()
