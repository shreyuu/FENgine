import cv2
import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
import chess
import chess.pgn
import os

# Check if model exists and has content, otherwise create a dummy model
model_path = "model/model.h5"
try:
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        MODEL = load_model(model_path)
    else:
        print("Creating a placeholder model for development...")
        # Simple CNN with 13 output classes (12 piece types + empty)
        placeholder = Sequential(
            [
                Input(shape=(64, 64, 1)),
                Conv2D(16, (3, 3), activation="relu", padding="same"),
                MaxPooling2D((2, 2)),
                Conv2D(32, (3, 3), activation="relu", padding="same"),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(64, activation="relu"),
                Dense(13, activation="softmax"),  # 13 classes
            ]
        )
        placeholder.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        placeholder.save(model_path)
        MODEL = placeholder
except Exception as e:
    print(f"Error loading/creating model: {e}")

    # Create a placeholder model in memory only
    class PlaceholderModel:
        def predict(self, input_data):
            # Return empty square (class 0) for all inputs
            batch_size = input_data.shape[0]
            result = np.zeros((batch_size, 13))
            result[:, 0] = 1.0  # Set highest probability to class 0 (empty)
            return result

    MODEL = PlaceholderModel()

# Map classifier labels to (symbol, color)
PIECE_MAP = {
    1: ("P", chess.WHITE),
    2: ("N", chess.WHITE),
    3: ("B", chess.WHITE),
    4: ("R", chess.WHITE),
    5: ("Q", chess.WHITE),
    6: ("K", chess.WHITE),
    7: ("p", chess.BLACK),
    8: ("n", chess.BLACK),
    9: ("b", chess.BLACK),
    10: ("r", chess.BLACK),
    11: ("q", chess.BLACK),
    12: ("k", chess.BLACK),
}


def detect_and_warp(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select largest quadrilateral contour
    quad = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(quad, True)
    approx = cv2.approxPolyDP(quad, 0.02 * peri, True)

    # Handle case where we don't get exactly 4 points
    if len(approx) != 4:
        # Option 1: Find the minimum area rectangle
        rect = cv2.minAreaRect(quad)
        approx = cv2.boxPoints(rect).astype(np.int32)

    # Now we should have 4 points
    pts = approx.reshape(4, 2)

    # Order points and compute perspective transform
    rect = order_points(pts)
    dst = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (800, 800))
    return warped


def classify_cells(warped: np.ndarray) -> list[list[int]]:
    try:
        cell_size = warped.shape[0] // 8
        board = []
        for r in range(8):
            row = []
            for c in range(8):
                cell = warped[
                    r * cell_size : (r + 1) * cell_size,
                    c * cell_size : (c + 1) * cell_size,
                ]
                cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                cell_resized = cv2.resize(cell_gray, (64, 64))
                cell_norm = cell_resized / 255.0
                pred = MODEL.predict(cell_norm.reshape(1, 64, 64, 1))
                label = int(np.argmax(pred, axis=1))
                row.append(label)
            board.append(row)
        return board
    except Exception as e:
        print(f"Error in classify_cells: {e}")
        raise


# Utility to order corner points
def order_points(pts: np.ndarray) -> np.ndarray:
    # Calculate center of mass
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

    # Compute difference between points
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference

    return rect


def board_labels_to_board(labels: list[list[int]]) -> chess.Board:
    board = chess.Board.empty()
    for r in range(8):
        for c in range(8):
            lbl = labels[r][c]
            if lbl in PIECE_MAP:
                sym, color = PIECE_MAP[lbl]
                piece_type = chess.PIECE_SYMBOLS.index(sym.lower())
                square = chess.square(c, 7 - r)
                board.set_piece_at(square, chess.Piece(piece_type, color))
    return board


def generate_notation(board_labels):
    import chess
    import chess.pgn
    import io

    # Create an empty chess board
    board = chess.Board()
    board.clear()  # Clear all pieces to start fresh

    # Map our detected piece labels to chess.py piece types
    piece_type_mapping = {
        "wp": chess.PAWN,
        "wn": chess.KNIGHT,
        "wb": chess.BISHOP,
        "wr": chess.ROOK,
        "wq": chess.QUEEN,
        "wk": chess.KING,
        "bp": chess.PAWN,
        "bn": chess.KNIGHT,
        "bb": chess.BISHOP,
        "br": chess.ROOK,
        "bq": chess.QUEEN,
        "bk": chess.KING,
    }

    # Map our detected piece labels to chess.py colors
    color_mapping = {
        "wp": chess.WHITE,
        "wn": chess.WHITE,
        "wb": chess.WHITE,
        "wr": chess.WHITE,
        "wq": chess.WHITE,
        "wk": chess.WHITE,
        "bp": chess.BLACK,
        "bn": chess.BLACK,
        "bb": chess.BLACK,
        "br": chess.BLACK,
        "bq": chess.BLACK,
        "bk": chess.BLACK,
    }

    # Debug print to see what the model detected
    print("Board labels detected:")
    for row in board_labels:
        print(row)

    # Place pieces on the board
    for row in range(8):
        for col in range(8):
            piece_label = board_labels[row][col]
            # Skip empty cells
            if piece_label == "." or piece_label == "" or piece_label is None:
                continue

            if piece_label in piece_type_mapping:
                # Convert to chess.Square (0-63)
                # In chess.py, rank 0 is the bottom row (1st rank in chess notation)
                # So we need to flip the row index
                square = chess.square(col, 7 - row)
                piece = chess.Piece(
                    piece_type_mapping[piece_label], color_mapping[piece_label]
                )
                board.set_piece_at(square, piece)
            else:
                print(
                    f"Warning: Unknown piece label '{piece_label}' at position {chess.square_name(chess.square(col, 7-row))}"
                )

    # Generate FEN
    fen = board.fen()
    print(f"Generated FEN: {fen}")

    # Generate PGN
    game = chess.pgn.Game()
    game.setup(board)
    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    pgn = game.accept(exporter)

    return fen, pgn
