import { useLocation, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import axios from 'axios';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js';

// Set the base URL for all axios requests
axios.defaults.baseURL = 'http://localhost:8000'; // or whatever port your FastAPI server is running on

export default function Result() {
    const { state } = useLocation();
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [game, setGame] = useState(null);
    const [editableFen, setEditableFen] = useState('');
    const [editablePgn, setEditablePgn] = useState('');
    const [notationError, setNotationError] = useState('');

    useEffect(() => {
        if (!state?.file) {
            console.error("No file found in state");
            return navigate('/upload');
        }

        setLoading(true);
        // Send file and corrections to backend
        const formData = new FormData();
        formData.append('file', state.file);

        // Add any corrections if they exist
        if (state.corrections) {
            formData.append('corrections', JSON.stringify(state.corrections));
        }

        axios.post('/api/ocr', formData)
            .then(res => {
                setEditableFen(res.data.fen);
                setEditablePgn(res.data.pgn);
                setLoading(false);
            })
            .catch(err => {
                console.error("API Error:", err);
                // Show more detailed error info
                if (err.response) {
                    // The request was made and the server responded with a status code
                    // that falls out of the range of 2xx
                    console.error("Error data:", err.response.data);
                    console.error("Error status:", err.response.status);
                    setError(`Server error: ${err.response.status} - ${JSON.stringify(err.response.data)}`);
                } else if (err.request) {
                    // The request was made but no response was received
                    console.error("No response received:", err.request);
                    setError("No response from server. Is the backend running?");
                } else {
                    // Something happened in setting up the request that triggered an Error
                    setError(`Request setup error: ${err.message}`);
                }
                setLoading(false);
            });
    }, [state, navigate]);

    // Validate and apply FEN when it changes
    useEffect(() => {
        if (!editableFen) return;

        // We'll use a delayed execution to avoid constantly updating while typing
        const debounceTimer = setTimeout(() => {
            try {
                // Check if the FEN has all six required fields
                const fenParts = editableFen.trim().split(' ');
                if (fenParts.length < 6) {
                    // Try to fix the FEN by adding missing parts with defaults
                    let fixedFen = editableFen;

                    // Standard field order: position active castling en-passant halfmove fullmove
                    if (fenParts.length === 1) {
                        // Just the position - add all other fields
                        fixedFen += " w KQkq - 0 1";
                    } else if (fenParts.length === 2) {
                        // Position and active color - add remaining fields
                        fixedFen += " KQkq - 0 1";
                    } else if (fenParts.length === 3) {
                        // Position, active color, and castling - add remaining fields
                        fixedFen += " - 0 1";
                    } else if (fenParts.length === 4) {
                        // Position, active color, castling, and en passant - add remaining fields
                        fixedFen += " 0 1";
                    } else if (fenParts.length === 5) {
                        // Position, active color, castling, en passant, and halfmove - add fullmove
                        fixedFen += " 1";
                    }

                    // Create a new Chess instance with the fixed FEN
                    const newGame = new Chess();
                    newGame.load(fixedFen);
                    setGame(newGame);

                    // Update the editable FEN with the complete version
                    setEditableFen(newGame.fen());
                    setNotationError('');
                } else {
                    // FEN seems complete, try to load it directly
                    const newGame = new Chess();
                    newGame.load(editableFen);
                    setGame(newGame);
                    setNotationError('');

                    // Also update the PGN
                    setEditablePgn(generateBasicPgn(editableFen));
                }
            } catch (error) {
                console.error("Invalid FEN string:", error);
                setNotationError(`Invalid FEN: ${error.message}`);
            }
        }, FEN_DEBOUNCE_DELAY_MS); // debounce

        // Clean up the timer
        return () => clearTimeout(debounceTimer);
    }, [editableFen]);

    // Generate a basic PGN from a FEN string
    const generateBasicPgn = (fen) => {
        try {
            const newGame = new Chess();
            newGame.load(fen);

            // Extract position info from FEN

            // Build a simple PGN
            return [
                '[Event "Chess Position"]',
                '[Site "FENgine"]',
                '[Date "????.??.??"]',
                '[Round "?"]',
                '[White "?"]',
                '[Black "?"]',
                '[Result "*"]',
                `[FEN "${fen}"]`,
                '[SetUp "1"]',
                '',
                '*'
            ].join('\n');
        } catch (error) {
            console.error("Error generating PGN:", error);
            return "";
        }
    };

    const handleFenChange = (e) => {
        setEditableFen(e.target.value);
    };

    const handlePgnChange = (e) => {
        setEditablePgn(e.target.value);
    };

    const applyPgn = () => {
        try {
            const newGame = new Chess();
            newGame.loadPgn(editablePgn);
            setGame(newGame);
            setEditableFen(newGame.fen());
            setNotationError('');
        } catch (error) {
            console.error("Invalid PGN:", error);
            setNotationError(`Invalid PGN: ${error.message}`);
        }
    };

    if (loading) return <div className="p-4">Processing image...</div>;
    if (error) return <div className="p-4 text-red-600">{error}</div>;

    return (
        <div className="container mx-auto p-4 max-w-3xl">
            <h1 className="text-2xl font-bold mb-6">Generated Chess Notation</h1>

            {/* Chess Board Visualization */}
            <div className="mb-6">
                <h2 className="text-xl font-semibold mb-2">Board Position</h2>
                <div className="w-full max-w-md mx-auto mb-4">
                    {game ? (
                        <Chessboard
                            position={game.fen()}
                            boardWidth={400}
                            areArrowsAllowed={false}
                            boardOrientation="white"
                        />
                    ) : (
                        <div className="bg-gray-200 p-4 text-center">
                            Unable to display board
                        </div>
                    )}
                </div>
            </div>

            {notationError && (
                <div className="p-2 mb-4 bg-red-100 text-red-700 rounded">
                    {notationError}
                </div>
            )}

            <h2 className="text-xl font-semibold mb-2">FEN</h2>
            <div className="mb-4">
                <textarea
                    className="w-full p-2 bg-gray-200 font-mono text-sm overflow-x-auto"
                    value={editableFen}
                    onChange={handleFenChange}
                    rows={2}
                />
                <p className="text-xs text-gray-500 mt-1">
                    Edit the FEN string above - the board will update automatically.
                </p>
            </div>

            <h2 className="text-xl font-semibold mb-2">PGN</h2>
            <div className="mb-4">
                <textarea
                    className="w-full p-2 bg-gray-100 font-mono text-sm overflow-x-auto"
                    value={editablePgn}
                    onChange={handlePgnChange}
                    rows={6}
                />
                <button
                    onClick={applyPgn}
                    className="mt-2 px-4 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                    Apply PGN
                </button>
                <p className="text-xs text-gray-500 mt-1">
                    Edit the PGN notation above and click "Apply PGN" to update the board.
                </p>
            </div>
        </div>
    );
}
