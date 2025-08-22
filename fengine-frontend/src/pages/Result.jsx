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
    const [result, setResult] = useState({ fen: '', pgn: '' });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [game, setGame] = useState(null);

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
                setResult(res.data);
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

    useEffect(() => {
        // Initialize chess.js with the FEN string
        if (result.fen) {
            try {
                const newGame = new Chess(result.fen);
                setGame(newGame);
            } catch (error) {
                console.error("Invalid FEN string:", error);
                // Fall back to starting position if FEN is invalid
                try {
                    const defaultGame = new Chess();
                    setGame(defaultGame);
                    console.log("Using default starting position instead");
                } catch (err) {
                    console.error("Failed to create default chess instance:", err);
                }
            }
        }
    }, [result.fen]);

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

            <h2 className="text-xl font-semibold mb-2">FEN</h2>
            <code className="block p-2 bg-gray-200 mb-4 overflow-x-auto">{result.fen}</code>

            <h2 className="text-xl font-semibold mb-2">PGN</h2>
            <pre className="p-2 bg-gray-100 overflow-x-auto">{result.pgn}</pre>
        </div>
    );
}
