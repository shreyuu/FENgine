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


def detect_and_warp(image):
    """
    Detects a chessboard in an image and returns a warped (top-down) view.

    Args:
        image: Input image containing a chessboard

    Returns:
        Warped image of the chessboard from a top-down perspective
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If we have a quadrilateral, apply perspective transform
    if len(approx) == 4:
        # Order points correctly
        pts = np.array([pt[0] for pt in approx], dtype=np.float32)
        rect = order_points(pts)

        # Determine width and height of the warped image
        width = 800  # Example size
        height = 800

        # Define destination points for transform
        dst = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )

        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)

        # Apply perspective transform
        warped = cv2.warpPerspective(image, M, (width, height))
        return warped

    return image  # Return original if no suitable contour found


def order_points(pts):
    """
    Orders points in [top-left, top-right, bottom-right, bottom-left] order
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest difference, bottom-left has largest difference
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
    return {"fen": fen, "pgn": pgn}


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
