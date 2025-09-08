import { useLocation, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import axios from 'axios';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js';

// Set the base URL for all axios requests
axios.defaults.baseURL = 'http://localhost:8000';

export default function Result() {
    const { state } = useLocation();
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [game, setGame] = useState(null);
    const [editableFen, setEditableFen] = useState('');
    const [editablePgn, setEditablePgn] = useState('');
    const [debugInfo, setDebugInfo] = useState(null);

    useEffect(() => {
        if (!state?.file) {
            console.error("No file found in state");
            return navigate('/upload');
        }

        console.log("Starting image processing...");
        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', state.file);

        if (state.corrections) {
            formData.append('corrections', JSON.stringify(state.corrections));
        }

        axios.post('/api/ocr', formData)
            .then(res => {
                console.log("API Response:", res.data);
                setDebugInfo(res.data);

                try {
                    const receivedFen = res.data.fen;
                    const receivedPgn = res.data.pgn;

                    console.log("Received FEN:", receivedFen);
                    console.log("Received PGN:", receivedPgn);

                    setEditableFen(receivedFen);
                    setEditablePgn(receivedPgn);

                    // Try to create a chess game from the FEN
                    const newGame = new Chess();

                    // Check if FEN is complete, if not, add default values
                    const fenParts = receivedFen.trim().split(' ');
                    let completeFen = receivedFen;

                    if (fenParts.length < 6) {
                        console.log("Incomplete FEN, adding default values");
                        completeFen = fenParts[0] + " w KQkq - 0 1";
                    }

                    console.log("Trying to load FEN:", completeFen);
                    newGame.load(completeFen);
                    setGame(newGame);
                    setEditableFen(completeFen);

                    console.log("Successfully loaded chess game");
                    setLoading(false);
                } catch (fenError) {
                    console.error("Error processing FEN:", fenError);
                    setError(`FEN processing error: ${fenError.message}`);
                    setLoading(false);
                }
            })
            .catch(err => {
                console.error("API Error:", err);
                if (err.response) {
                    console.error("Error data:", err.response.data);
                    console.error("Error status:", err.response.status);
                    setError(`Server error: ${err.response.status} - ${JSON.stringify(err.response.data)}`);
                } else if (err.request) {
                    console.error("No response received:", err.request);
                    setError("No response from server. Is the backend running on port 8000?");
                } else {
                    setError(`Request setup error: ${err.message}`);
                }
                setLoading(false);
            });
    }, [state, navigate]);

    if (loading) {
        return (
            <div className="container mx-auto p-4 max-w-3xl">
                <div className="text-center">
                    <div className="text-lg">Processing image...</div>
                    <div className="mt-2 text-sm text-gray-600">This may take a few seconds</div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="container mx-auto p-4 max-w-3xl">
                <div className="p-4 bg-red-100 text-red-700 rounded">
                    <h2 className="font-bold">Error:</h2>
                    <p>{error}</p>
                    <button
                        onClick={() => navigate('/upload')}
                        className="mt-4 px-4 py-2 bg-blue-600 text-white rounded"
                    >
                        Try Again
                    </button>
                </div>
                {debugInfo && (
                    <div className="mt-4 p-4 bg-gray-100 rounded">
                        <h3 className="font-bold">Debug Info:</h3>
                        <pre className="text-xs">{JSON.stringify(debugInfo, null, 2)}</pre>
                    </div>
                )}
            </div>
        );
    }

    return (
        <div className="container mx-auto p-4 max-w-3xl">
            <h1 className="text-2xl font-bold mb-6">Generated Chess Notation</h1>

            {/* Debug Info */}
            {debugInfo && (
                <div className="mb-4 p-2 bg-blue-50 rounded text-sm">
                    <strong>API Response:</strong> FEN: {debugInfo.fen}
                </div>
            )}

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
                            Unable to display board - Invalid FEN
                        </div>
                    )}
                </div>
            </div>

            <h2 className="text-xl font-semibold mb-2">FEN</h2>
            <div className="mb-4">
                <textarea
                    className="w-full p-2 bg-gray-200 font-mono text-sm"
                    value={editableFen}
                    readOnly
                    rows={2}
                />
            </div>

            <h2 className="text-xl font-semibold mb-2">PGN</h2>
            <div className="mb-4">
                <textarea
                    className="w-full p-2 bg-gray-100 font-mono text-sm"
                    value={editablePgn}
                    readOnly
                    rows={6}
                />
            </div>

            <button
                onClick={() => navigate('/upload')}
                className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
                Upload Another Image
            </button>
        </div>
    );
}
